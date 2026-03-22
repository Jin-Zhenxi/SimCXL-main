#!/usr/bin/env python3
"""
ViT-inspired layer proxy sweep:
edit trigger_gemm.c preset defaults -> build/inject -> run gem5 -> collect
ROI stats and per-phase timings.
"""

import csv
import os
import re
import subprocess

# ================= 配置区 =================
GEM5_ROOT = "/home/jzx8091/SimCXL-main"
PRESETS = [
    {
        "name": "ViT-Base-like",
        "seq_len": 197,
        "hidden_dim": 768,
        "mlp_dim": 3072,
        "num_heads": 12,
    },
    {
        "name": "ViT-Large-like",
        "seq_len": 257,
        "hidden_dim": 1024,
        "mlp_dim": 4096,
        "num_heads": 16,
    },
    {
        "name": "ViT-Huge-like",
        "seq_len": 257,
        "hidden_dim": 1280,
        "mlp_dim": 5120,
        "num_heads": 16,
    },
]
OUTPUT_DIR = "my_outputData"
C_FILE = "trigger_gemm.c"
CSV_FILE = os.path.join(OUTPUT_DIR, "roofline_data.csv")
RAW_TXT_FILE = os.path.join(OUTPUT_DIR, "summary_raw.txt")
CLOCK_FREQ_GHZ = 2.4

COMPILE_CMD = (
    f"gcc -O3 -static -mno-80387 -o trigger_gemm trigger_gemm.c "
    f"-I{GEM5_ROOT}/include -L{GEM5_ROOT}/util/m5/build/x86/out -lm -lm5"
)
INJECT_CMD = "sudo ./inject_trigger_gemm.sh"
GEM5_CONFIG = "configs/example/gem5_library/x86-cxl-pcie-hbm-autofallback.py"
GEM5_CMD_TEMPLATE = (
    f"./build/X86/gem5.opt -p {GEM5_ROOT}/src/python "
    "--debug-flags=MatrixFlow -d {outdir} "
    f"{GEM5_CONFIG} "
    "--is_asic True --cpu_type TIMING > {logfile} 2>&1"
)
# =========================================

CHUNK_SEPARATOR = "---------- Begin Simulation Statistics ----------"
LIBM5_PATH = os.path.join(GEM5_ROOT, "util/m5/build/x86/out/libm5.a")


def ensure_libm5():
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


def modify_c_file(preset):
    with open(C_FILE, "r") as f:
        content = f.read()

    replacements = {
        r'uint32_t default_seq_len = \d+;':
            f"uint32_t default_seq_len = {preset['seq_len']};",
        r'uint32_t default_hidden_dim = \d+;':
            f"uint32_t default_hidden_dim = {preset['hidden_dim']};",
        r'uint32_t default_mlp_dim = \d+;':
            f"uint32_t default_mlp_dim = {preset['mlp_dim']};",
        r'uint32_t default_num_heads = \d+;':
            f"uint32_t default_num_heads = {preset['num_heads']};",
        r'\.preset_name = "[^"]+",':
            f'.preset_name = "{preset["name"]}",',
    }

    for pattern, repl in replacements.items():
        content = re.sub(pattern, repl, content)

    with open(C_FILE, "w") as f:
        f.write(content)

    print(
        "[*] 已写入预设: "
        f"{preset['name']} "
        f"(seq_len={preset['seq_len']}, hidden_dim={preset['hidden_dim']}, "
        f"mlp_dim={preset['mlp_dim']}, num_heads={preset['num_heads']})"
    )


def extract_metrics(stats_file, log_file):
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

    if metrics["computeCycles"] == 0 and os.path.exists(log_file):
        with open(log_file, "r") as f:
            log_text = f.read()
            match_cycles = re.search(r"computeCycles=(\d+)", log_text)
            if match_cycles:
                metrics["computeCycles"] = int(match_cycles.group(1))

    return metrics


def extract_gemm_active_time(log_file):
    if not os.path.exists(log_file):
        return 0.0

    start_ticks = []
    done_ticks = []
    with open(log_file, "r") as f:
        for line in f:
            match = re.search(r"^(\d+): .*startMatrixCompute:", line)
            if match:
                start_ticks.append(int(match.group(1)))
                continue

            match = re.search(
                r"^(\d+): .*MatrixFlow completion flag written:", line
            )
            if match:
                done_ticks.append(int(match.group(1)))

    pair_count = min(len(start_ticks), len(done_ticks))
    if pair_count == 0:
        return 0.0

    total_ticks = 0
    for idx in range(pair_count):
        if done_ticks[idx] > start_ticks[idx]:
            total_ticks += done_ticks[idx] - start_ticks[idx]

    return total_ticks / 1e12


def extract_phase_timings(log_file):
    timings = {
        "phase1_ms": 0.0,
        "phase2_d2h_ms": 0.0,
        "phase2_softmax_ms": 0.0,
        "phase2_layernorm_ms": 0.0,
        "phase2_gelu_ms": 0.0,
        "phase2_residual_ms": 0.0,
        "phase2_h2d_ms": 0.0,
        "phase2_total_ms": 0.0,
        "phase3_ms": 0.0,
        "end_to_end_ms": 0.0,
    }

    if not os.path.exists(log_file):
        return timings

    with open(log_file, "r") as f:
        text = f.read()

    for key in timings:
        match = re.search(rf"\[Timing\] {re.escape(key)}=([0-9.]+)", text)
        if match:
            timings[key] = float(match.group(1))

    return timings


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = []

    print("🚀 启动 ViT-inspired Layer Proxy Sweep")
    print("   物理时钟: 2.4 GHz，PCIe 64GB/s，zero packetization penalty")
    print(
        "   预设: "
        + ", ".join(
            f"{preset['name']}[S={preset['seq_len']},H={preset['hidden_dim']},"
            f"M={preset['mlp_dim']},heads={preset['num_heads']}]"
            for preset in PRESETS
        )
    )
    print("=" * 50)

    ensure_libm5()

    for preset in PRESETS:
        preset_tag = preset["name"].replace(" ", "_").replace("/", "_")
        seq_len = preset["seq_len"]

        print(f"\n>>> 正在测试预设: {preset['name']} <<<")
        modify_c_file(preset)

        print("[*] 正在编译 trigger_gemm (需 libm5.a + libm)...")
        subprocess.run(COMPILE_CMD, shell=True, check=True)

        inject_cmd = f"{INJECT_CMD} --no-compile {seq_len}"
        print(
            "[*] 正在注入镜像 "
            f"(script.sh 将执行 trigger_gemm 0x200000000 {seq_len})..."
        )
        subprocess.run(inject_cmd, shell=True, check=True)

        m5out_dir = os.path.join(OUTPUT_DIR, f"m5out_{preset_tag}")
        log_file = os.path.join(OUTPUT_DIR, f"terminal_log_{preset_tag}.txt")
        cmd = GEM5_CMD_TEMPLATE.format(outdir=m5out_dir, logfile=log_file)

        print(f"[*] 正在运行 gem5 仿真，日志存入: {log_file}")
        rc = subprocess.run(cmd, shell=True)
        if rc.returncode != 0:
            raise SystemExit(
                f"gem5 仿真失败，返回码={rc.returncode}。请检查日志: {log_file}"
            )

        stats_file = os.path.join(m5out_dir, "stats.txt")
        metrics = extract_metrics(stats_file, log_file)
        if (
            metrics["simSeconds"] == 0.0
            and metrics["dmaRead"] == 0
            and metrics["dmaWrite"] == 0
            and metrics["computeCycles"] == 0
        ):
            raise SystemExit(
                f"未从 {stats_file} 提取到有效 ROI 统计。请检查日志: {log_file}"
            )

        phase_timings = extract_phase_timings(log_file)
        flops = 4.0 * (seq_len**3)
        roi_latency_s = metrics["simSeconds"]
        gemm_active_s = extract_gemm_active_time(log_file)
        total_bytes = metrics["dmaRead"] + metrics["dmaWrite"]

        peak_gflops = (
            (flops / 1e9)
            / (metrics["computeCycles"] / (CLOCK_FREQ_GHZ * 1e9))
            if metrics["computeCycles"] > 0
            else 0.0
        )
        gemm_effective_peak_gflops = (
            (flops / 1e9) / gemm_active_s if gemm_active_s > 0 else 0.0
        )
        effective_gflops = (
            (flops / 1e9) / roi_latency_s if roi_latency_s > 0 else 0.0
        )
        oi = flops / total_bytes if total_bytes > 0 else 0.0

        result_row = {
            "Preset": preset["name"],
            "SeqLen": preset["seq_len"],
            "HiddenDim": preset["hidden_dim"],
            "MLPDim": preset["mlp_dim"],
            "NumHeads": preset["num_heads"],
            "GEMM Proxy Size": seq_len,
            "FLOPs": int(flops),
            "End-to-End ROI Latency (s)": round(roi_latency_s, 6),
            "end_to_end_ms": round(phase_timings["end_to_end_ms"], 6),
            "phase1_ms": round(phase_timings["phase1_ms"], 6),
            "phase2_d2h_ms": round(phase_timings["phase2_d2h_ms"], 6),
            "phase2_softmax_ms": round(phase_timings["phase2_softmax_ms"], 6),
            "phase2_layernorm_ms": round(
                phase_timings["phase2_layernorm_ms"], 6
            ),
            "phase2_gelu_ms": round(phase_timings["phase2_gelu_ms"], 6),
            "phase2_residual_ms": round(
                phase_timings["phase2_residual_ms"], 6
            ),
            "phase2_h2d_ms": round(phase_timings["phase2_h2d_ms"], 6),
            "phase2_total_ms": round(phase_timings["phase2_total_ms"], 6),
            "phase3_ms": round(phase_timings["phase3_ms"], 6),
            "totalDmaRead": metrics["dmaRead"],
            "totalDmaWrite": metrics["dmaWrite"],
            "totalComputeCycles": metrics["computeCycles"],
            "Operational Intensity (OI)": round(oi, 4),
            "Peak MAC Throughput (GFLOPS)": round(peak_gflops, 6),
            "Effective GEMM Peak (GFLOPS)": round(
                gemm_effective_peak_gflops, 6
            ),
            "Effective System Throughput (GFLOPS)": round(
                effective_gflops, 6
            ),
        }
        results.append(result_row)

        print(
            f"[√] {preset['name']} 完成! "
            f"ROI={result_row['End-to-End ROI Latency (s)']:.6f}s, "
            f"Phase2={result_row['phase2_total_ms']:.3f}ms, "
            f"GEMM-Effective={result_row['Effective GEMM Peak (GFLOPS)']:.3f} GFLOPS, "
            f"Effective={result_row['Effective System Throughput (GFLOPS)']:.3f} GFLOPS"
        )

    if results:
        headers = results[0].keys()
        with open(CSV_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(results)

        with open(RAW_TXT_FILE, "w") as f:
            for row in results:
                f.write(str(row) + "\n")

        print("\n" + "=" * 50)
        print(f"🎉 Sweep 完成！数据已保存至: {CSV_FILE}")


if __name__ == "__main__":
    main()
