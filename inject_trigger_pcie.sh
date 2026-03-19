#!/bin/bash
# 将 trigger_pcie 注入 parsec.img，使 Guest 无需网络即可运行
# 用法: ./inject_trigger_pcie.sh [--no-compile] [SIZE]
#   --no-compile: 跳过编译，使用已有的 trigger_pcie（供 run_sweep.py 使用）
#   SIZE: Guest 侧矩阵规模，默认 1024
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 解析参数
SKIP_COMPILE=false
SIZE=1024
for arg in "$@"; do
    case "$arg" in
        --no-compile|-n) SKIP_COMPILE=true ;;
        [0-9]*) SIZE=$arg ;;
    esac
done

# 1. 编译 trigger_pcie（除非 --no-compile，需 libm5.a：cd util/m5 && scons build/x86）
if [ "$SKIP_COMPILE" = false ]; then
    echo "=== 编译 trigger_pcie ==="
    GEM5_ROOT="$(cd "$SCRIPT_DIR" && pwd)"
    gcc -O0 -static -mno-80387 -o trigger_pcie trigger_pcie.c \
        -I"$GEM5_ROOT/include" -L"$GEM5_ROOT/util/m5/build/x86/out" -lm5 \
        || { echo "编译失败，请先 cd util/m5 && scons build/x86 生成 libm5.a"; exit 1; }
else
    echo "=== 跳过编译，使用已有 trigger_pcie ==="
fi

# 1.5 生成 Guest 启动脚本，SIZE 控制矩阵规模
cat > script.sh <<EOF
#!/bin/bash
echo "--- CONFIGURING NETWORK ---"
echo "Skipping network setup; trigger_pcie is already injected locally."

echo "--- EXECUTING TEST ---"
# 优先使用镜像里已注入的本地 trigger_pcie，避免无网络时卡死在 wget。
# 仅当本地二进制缺失时，才尝试从宿主机下载一个新版本。
TRIGGER=""
[ -x /root/trigger_pcie ] && TRIGGER=/root/trigger_pcie
[ -n "\$TRIGGER" ] || [ ! -x /home/test_code/trigger_pcie ] || TRIGGER=/home/test_code/trigger_pcie
[ -n "\$TRIGGER" ] || [ ! -x ./trigger_pcie ] || TRIGGER=./trigger_pcie
if [ -z "\$TRIGGER" ]; then
    echo "--- DOWNLOADING CODE FROM HOST ---"
    TMP_TRIGGER=/tmp/trigger_pcie.download
    rm -f "\$TMP_TRIGGER"
    if wget -T 3 -t 1 -q http://10.0.2.2:8000/trigger_pcie -O "\$TMP_TRIGGER" 2>/dev/null || \
       wget -T 3 -t 1 http://10.0.2.2:8000/trigger_pcie -O "\$TMP_TRIGGER" 2>/dev/null; then
        if [ -s "\$TMP_TRIGGER" ]; then
            mv "\$TMP_TRIGGER" /root/trigger_pcie
            chmod +x /root/trigger_pcie
            TRIGGER=/root/trigger_pcie
        fi
    fi
    rm -f "\$TMP_TRIGGER"
fi
if [ -n "\$TRIGGER" ]; then
    echo "Using trigger binary: \$TRIGGER"
    "\$TRIGGER" 0x200000000 $SIZE
else
    echo "trigger_pcie not found. Run ./inject_trigger_pcie.sh first to add it to disk."
fi
echo "--- AUTO TEST FINISHED ---"
m5 exit
EOF
chmod +x script.sh

# 2. 挂载磁盘镜像并复制
echo "=== 注入到 parsec.img ==="
IMG="parsec.img"
MNT="/tmp/parsec_mnt_$$"
mkdir -p "$MNT"

# 使用 kpartx 或 guestmount（若可用）
if command -v kpartx &>/dev/null; then
    LOOP=$(sudo kpartx -av "$IMG" | grep -o 'loop[0-9]*p1' | head -1)
    LOOP_DEV="/dev/mapper/$LOOP"
    sudo mount "$LOOP_DEV" "$MNT"
    trap "sudo umount $MNT; sudo kpartx -dv $IMG; rmdir $MNT" EXIT
elif command -v guestmount &>/dev/null; then
    sudo guestmount -a "$IMG" -m /dev/sda1 "$MNT"
    trap "sudo guestunmount $MNT; rmdir $MNT" EXIT
else
    # 分区 1 起始扇区 2048，偏移 = 2048*512 = 1048576
    sudo mount -o loop,offset=1048576 "$IMG" "$MNT" || {
        echo "挂载失败。可安装: sudo apt install kpartx"
        rmdir "$MNT"
        exit 1
    }
    trap "sudo umount $MNT; rmdir $MNT" EXIT
fi

# 复制到 /root 和 /home/test_code
sudo mkdir -p "$MNT/home/test_code"
sudo cp trigger_pcie "$MNT/root/trigger_pcie"
sudo cp trigger_pcie "$MNT/home/test_code/trigger_pcie"
sudo cp script.sh "$MNT/root/script.sh"
sudo cp script.sh "$MNT/home/test_code/script.sh"
sudo chmod +x "$MNT/root/trigger_pcie" "$MNT/home/test_code/trigger_pcie"
sudo chmod +x "$MNT/root/script.sh" "$MNT/home/test_code/script.sh"
echo "已注入到 /root/trigger_pcie、/home/test_code/trigger_pcie 以及 script.sh"
echo "=== 完成 ==="
