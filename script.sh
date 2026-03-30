#!/bin/bash
echo "--- CONFIGURING NETWORK ---"
echo "Skipping network setup; trigger_gemm is already injected locally."

echo "--- EXECUTING TEST ---"
# 优先使用镜像里已注入的本地 trigger_gemm，避免无网络时卡死在 wget。
# 仅当本地二进制缺失时，才尝试从宿主机下载一个新版本。
TRIGGER=""
[ -x /root/trigger_gemm ] && TRIGGER=/root/trigger_gemm
[ -n "$TRIGGER" ] || [ ! -x /home/test_code/trigger_gemm ] || TRIGGER=/home/test_code/trigger_gemm
[ -n "$TRIGGER" ] || [ ! -x ./trigger_gemm ] || TRIGGER=./trigger_gemm
if [ -z "$TRIGGER" ]; then
    echo "--- DOWNLOADING CODE FROM HOST ---"
    TMP_TRIGGER=/tmp/trigger_gemm.download
    rm -f "$TMP_TRIGGER"
    if wget -T 3 -t 1 -q http://10.0.2.2:8000/trigger_gemm -O "$TMP_TRIGGER" 2>/dev/null ||        wget -T 3 -t 1 http://10.0.2.2:8000/trigger_gemm -O "$TMP_TRIGGER" 2>/dev/null; then
        if [ -s "$TMP_TRIGGER" ]; then
            mv "$TMP_TRIGGER" /root/trigger_gemm
            chmod +x /root/trigger_gemm
            TRIGGER=/root/trigger_gemm
        fi
    fi
    rm -f "$TMP_TRIGGER"
fi
if [ -n "$TRIGGER" ]; then
    echo "Using trigger binary: $TRIGGER"
    "$TRIGGER" 0x200000000 257
else
    echo "trigger_gemm not found. Run ./inject_trigger_gemm.sh first to add it to disk."
fi
echo "--- AUTO TEST FINISHED ---"
m5 exit
