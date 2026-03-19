#!/usr/bin/env python3
"""
自动化 Roofline 扫略脚本：改代码 → 编译注入 → 跑仿真 → 抓数据 → 算力学公式 → 生成 CSV

使用 m5_reset_stats / m5_dump_stats 切分纯净 ROI，仅从 stats.txt 最后一块提取延迟。
物理时钟频率 2.4 GHz，严格区分 Peak MAC Throughput 与 Effective System Throughput。

若 libm5.a 不存在，脚本会自动执行 scons 构建。
"""
import os
import re
import subprocess
import csv

# ================= 配置区 =================
GEM5_ROOT = "/home/jzx8091/SimCXL-main"
SIZES = [256, 512, 1024, 2048]
OUTPUT_DIR = "my_outputData"
C_FILE = "trigger_gemm.c"
CSV_FILE = os.path.join(OUTPUT_DIR, "roofline_data.csv")
RAW_TXT_FILE = os.path.join(OUTPUT_DIR, "summary_raw.txt")

# 物理时钟频率 (Hz)
CLOCK_FREQ_GHZ = 2.4

# 编译命令：需链接 m5ops 库（libm5.a 需在 util/m5 下 scons build/x86 生成）
COMPILE_CMD = (
    f"gcc -O3 -static -mno-80387 -o trigger_gemm trigger_gemm.c "
    f"-I{GEM5_ROOT}/include -L{GEM5_ROOT}/util/m5/build/x86/out -lm5"
)
# 注入命令（传 --no-compile 使用已编译的 binary，传 size 控制 Guest 侧矩阵规模）
INJECT_CMD = "sudo ./inject_trigger_gemm.sh"

# gem5 启动命令模板
GEM5_CMD_TEMPLATE = (
    f"./build/X86/gem5.opt -p {GEM5_ROOT}/src/python "
    "--debug-flags=MatrixFlow -d {outdir} "
    "configs/example/gem5_library/x86-cxl-type3-with-classic.py "
    "--is_asic True --cpu_type TIMING > {logfile} 2>&1"
)
# =========================================

CHUNK_SEPARATOR = "---------- Begin Simulation Statistics ----------"
LIBM5_PATH = os.path.join(GEM5_ROOT, "util/m5/build/x86/out/libm5.a")


def ensure_libm5():
    """若 libm5.a 不存在，自动在 util/m5 下执行 scons 构建"""
    if os.path.exists(LIBM5_PATH):
        return
    print("[*] libm5.a 未找到，正在自动构建...")
    m5_dir = os.path.join(GEM5_ROOT, "util/m5")
    rc = subprocess.run(
        ["scons", "build/x86/out/libm5.a"],
        cwd=m5_dir,
        capture_output=True,
        text=True,
    )
    if rc.returncode != 0:
        print(rc.stderr or rc.stdout)
        raise SystemExit(
            f"构建 libm5.a 失败。请手动执行: cd {m5_dir} && scons build/x86/out/libm5.a"
        )
    print("[*] libm5.a 构建完成")


def modify_c_file(size):
    """修改 C 文件中的默认矩阵规模（用于 inject 未传 size 时的 fallback）"""
    with open(C_FILE, "r") as f:
        content = f.read()
    new_content = re.sub(
        r"uint32_t matrix_size = \d+",
        f"uint32_t matrix_size = {size}",
        content,
    )
    with open(C_FILE, "w") as f:
        f.write(new_content)
    print(f"[*] 已将 {C_FILE} 中的默认矩阵大小修改为: {size}")


def extract_metrics(stats_file, log_file):
    """
    从 stats.txt 中提取纯净 ROI 数据。
    gem5 退出时可能追加空的收尾 stats 块，chunks[-1] 往往无效。
    改为寻找特征块：包含 totalComputeCycles > 0 的那一块（MatrixFlowEngine 真实执行）。
    """
    metrics = {
        "simSeconds": 0.0,
        "dmaRead": 0,
        "dmaWrite": 0,
        "computeCycles": 0,
    }

    if not os.path.exists(stats_file):
        return metrics

    with open(stats_file, "r") as f:
        stats_text = f.read()

    chunks = stats_text.split(CHUNK_SEPARATOR)

    # 寻找特征块：totalComputeCycles > 0 表示 MatrixFlowEngine 真实执行过
    target_chunk = ""
    for chunk in chunks:
        match = re.search(r"totalComputeCycles\s+(\d+)", chunk)
        if match and int(match.group(1)) > 0:
            target_chunk = chunk
            break

    if not target_chunk:
        target_chunk = chunks[-1] if chunks else ""

    target_chunk = target_chunk.strip()
    if not target_chunk:
        return metrics

    # 从特征块中提取
    match_time = re.search(r"simSeconds\s+([0-9.]+)", target_chunk)
    match_read = re.search(r"totalDmaBytesRead\s+(\d+)", target_chunk)
    match_write = re.search(r"totalDmaBytesWritten\s+(\d+)", target_chunk)
    match_cycles = re.search(r"totalComputeCycles\s+(\d+)", target_chunk)

    if match_time:
        metrics["simSeconds"] = float(match_time.group(1))
    if match_read:
        metrics["dmaRead"] = int(match_read.group(1))
    if match_write:
        metrics["dmaWrite"] = int(match_write.group(1))
    if match_cycles:
        metrics["computeCycles"] = int(match_cycles.group(1))

    # 若最后一块无 computeCycles，尝试从 terminal log 抓取（DPRINTF 输出）
    if metrics["computeCycles"] == 0 and os.path.exists(log_file):
        with open(log_file, "r") as f:
            log_text = f.read()
            match_cycles = re.search(r"computeCycles=(\d+)", log_text)
            if match_cycles:
                metrics["computeCycles"] = int(match_cycles.group(1))

    return metrics


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = []

    print(f"🚀 启动自动化 Sweep 测试，目标规模: {SIZES}")
    print(f"   物理时钟: {CLOCK_FREQ_GHZ} GHz，ROI 纯净延迟（特征块 totalComputeCycles>0）")
    print("=" * 50)

    ensure_libm5()

    for size in SIZES:
        print(f"\n>>> 正在测试矩阵规模: {size}x{size} <<<")

        # 1. 修改 C 文件默认值并编译
        modify_c_file(size)
        print("[*] 正在编译 trigger_gemm (需 libm5.a)...")
        subprocess.run(COMPILE_CMD, shell=True, check=True)

        # 2. 注入镜像（--no-compile 使用已编译 binary，传入 size 控制 Guest 侧）
        inject_cmd = f"{INJECT_CMD} --no-compile {size}"
        print(f"[*] 正在注入镜像 (script.sh 将执行 trigger_gemm 0x200000000 {size})...")
        subprocess.run(inject_cmd, shell=True, check=True)

        # 3. 启动 gem5
        m5out_dir = os.path.join(OUTPUT_DIR, f"m5out_{size}")
        log_file = os.path.join(OUTPUT_DIR, f"terminal_log_{size}.txt")
        cmd = GEM5_CMD_TEMPLATE.format(outdir=m5out_dir, logfile=log_file)

        print(f"[*] 正在运行 gem5 仿真（预计耗时较长），日志存入: {log_file}")
        subprocess.run(cmd, shell=True)

        # 4. 抓取与计算（仅从最后一块 stats 提取）
        stats_file = os.path.join(m5out_dir, "stats.txt")
        m = extract_metrics(stats_file, log_file)

        # --- 严格物理公式 ---
        flops = 2.0 * (size**3)
        roi_latency_s = m["simSeconds"]
        total_bytes = m["dmaRead"] + m["dmaWrite"]

        # Peak MAC Throughput: 理论峰值，仅基于 totalComputeCycles（不含 DMA stall）
        peak_gflops = (
            (flops / 1e9) / (m["computeCycles"] / (CLOCK_FREQ_GHZ * 1e9))
            if m["computeCycles"] > 0
            else 0.0
        )

        # Effective System Throughput: 诚实算力，包含 DMA 延迟
        effective_gflops = (flops / 1e9) / roi_latency_s if roi_latency_s > 0 else 0.0

        oi = flops / total_bytes if total_bytes > 0 else 0.0

        result_row = {
            "Size": size,
            "FLOPs": int(flops),
            "End-to-End ROI Latency (s)": round(roi_latency_s, 6),
            "totalDmaRead": m["dmaRead"],
            "totalDmaWrite": m["dmaWrite"],
            "totalComputeCycles": m["computeCycles"],
            "Operational Intensity (OI)": round(oi, 4),
            "Peak MAC Throughput (GFLOPS)": round(peak_gflops, 6),
            "Effective System Throughput (GFLOPS)": round(effective_gflops, 6),
        }
        results.append(result_row)
        print(
            f"[√] 测试完成! OI={result_row['Operational Intensity (OI)']}, "
            f"Peak={result_row['Peak MAC Throughput (GFLOPS)']:.4f} GFLOPS, "
            f"Effective={result_row['Effective System Throughput (GFLOPS)']:.4f} GFLOPS"
        )

    # 5. 保存到 CSV 和 TXT
    if results:
        headers = results[0].keys()
        with open(CSV_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(results)

        with open(RAW_TXT_FILE, "w") as f:
            for r in results:
                f.write(str(r) + "\n")

        print("\n" + "=" * 50)
        print(f"🎉 扫略测试全线竣工！数据已保存至: {CSV_FILE}")


if __name__ == "__main__":
    main()
