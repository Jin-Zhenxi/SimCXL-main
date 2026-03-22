#!/usr/bin/env python3
"""
ViT-inspired layer proxy sweep for the CXL Type-3 + device-side DDR5 HDM branch.
This is not a full-model ViT runtime.
"""
import os
import re
import subprocess
import csv

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
DEFAULT_PHASE2_MODE = "staged_block"
DEFAULT_STAGED_BLOCK_BYTES = 1024

# 物理时钟频率 (Hz)
CLOCK_FREQ_GHZ = 2.4

# 编译命令：需链接 m5ops 库（libm5.a 需在 util/m5 下 scons build/x86 生成）
COMPILE_CMD = (
    f"gcc -O3 -static -mno-80387 -o trigger_gemm trigger_gemm.c "
    f"-I{GEM5_ROOT}/include -L{GEM5_ROOT}/util/m5/build/x86/out -lm -lm5"
)
INJECT_CMD = "sudo ./inject_trigger_gemm.sh"

# gem5 启动命令模板
GEM5_CMD_TEMPLATE = (
    f"./build/X86/gem5.opt -p {GEM5_ROOT}/src/python "
    "-d {outdir} "
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


def modify_c_file(preset):
    """写入 ViT-like preset 默认值。"""
    with open(C_FILE, "r") as f:
        content = f.read()

    replacements = {
        r"uint32_t default_seq_len = \d+;":
            f"uint32_t default_seq_len = {preset['seq_len']};",
        r"uint32_t default_hidden_dim = \d+;":
            f"uint32_t default_hidden_dim = {preset['hidden_dim']};",
        r"uint32_t default_mlp_dim = \d+;":
            f"uint32_t default_mlp_dim = {preset['mlp_dim']};",
        r"uint32_t default_num_heads = \d+;":
            f"uint32_t default_num_heads = {preset['num_heads']};",
        r'const char \*default_phase2_mode = "[^"]+";':
            f'const char *default_phase2_mode = "{DEFAULT_PHASE2_MODE}";',
        r"uint32_t default_staged_block_bytes = \d+;":
            f"uint32_t default_staged_block_bytes = {DEFAULT_STAGED_BLOCK_BYTES};",
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


def extract_gemm_active_time(log_file):
    phase = extract_phase_timings(log_file)
    return (phase["phase1_ms"] + phase["phase3_ms"]) / 1.0e3


def extract_phase_timings(serial_log_file):
    timing_keys = [
        "phase1_ms",
        "phase2_d2h_ms",
        "phase2_h2d_ms",
        "phase2_cxl_inplace_ms",
        "phase2_expand_ms",
        "phase2_softmax_ms",
        "phase2_layernorm_ms",
        "phase2_project_ms",
        "phase2_gelu_ms",
        "phase2_residual_ms",
        "phase2_compact_ms",
        "phase2_non_gemm_ms",
        "phase2_total_ms",
        "phase3_ms",
        "end_to_end_ms",
    ]
    byte_keys = [
        "host_mediated_copy_bytes",
        "staged_block_bytes",
        "gemm_tile_count_per_gemm",
        "gemm_tile_count_total",
        "descriptor_launch_count",
        "doorbell_launch_count",
        "phase1_poll_count",
        "phase3_poll_count",
        "cxl_inplace_access_bytes",
        "score_read_bytes",
        "score_write_bytes",
        "score_read_accesses",
        "score_write_accesses",
        "hidden_read_bytes",
        "hidden_write_bytes",
        "hidden_read_accesses",
        "hidden_write_accesses",
        "residual_read_bytes",
        "residual_write_bytes",
        "residual_read_accesses",
        "residual_write_accesses",
        "mlp_read_bytes",
        "mlp_write_bytes",
        "mlp_read_accesses",
        "mlp_write_accesses",
        "proxy_remote_footprint_bytes",
        "total_proxy_movement_bytes",
    ]

    values = {key: 0.0 for key in timing_keys}
    values.update({key: 0 for key in byte_keys})
    values["phase2_mode"] = "unknown"

    if not os.path.exists(serial_log_file):
        return values

    timing_patterns = {
        key: re.compile(rf"\[Timing\] {re.escape(key)}=([0-9.]+)")
        for key in timing_keys
    }
    byte_patterns = {
        key: re.compile(rf"\[Timing\] {re.escape(key)}=(\d+)")
        for key in byte_keys
    }
    mode_pattern = re.compile(r"\[Timing\] phase2_mode=([A-Za-z0-9_]+)")

    with open(serial_log_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            mode_match = mode_pattern.search(line)
            if mode_match:
                values["phase2_mode"] = mode_match.group(1)
            for key, pattern in timing_patterns.items():
                match = pattern.search(line)
                if match:
                    values[key] = float(match.group(1))

            for key, pattern in byte_patterns.items():
                match = pattern.search(line)
                if match:
                    values[key] = int(match.group(1))

    return values


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = []

    print("🚀 启动 ViT-inspired Layer Proxy Sweep")
    print("   架构: CXL Type-3 + 设备侧 DDR5 HDM + device-side MatrixFlow")
    print("   口径: timing-mode 已补丁验证；链路为 64 GB/s 级别原型")
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

        serial_log = os.path.join(m5out_dir, "board.pc.com_1.device")
        phase = extract_phase_timings(serial_log)
        proxy_gemm_flops = 4.0 * (seq_len**3)
        attention_gemm_flops = 4.0 * preset["seq_len"] * preset["seq_len"] * preset["hidden_dim"]
        mlp_gemm_flops = 4.0 * preset["seq_len"] * preset["hidden_dim"] * preset["mlp_dim"]
        softmax_flops = 5.0 * preset["seq_len"] * preset["seq_len"]
        layernorm_flops = 6.0 * preset["seq_len"] * preset["hidden_dim"]
        gelu_flops = 8.0 * preset["seq_len"] * preset["mlp_dim"]
        residual_flops = 1.0 * preset["seq_len"] * preset["hidden_dim"]
        non_gemm_flops = (
            softmax_flops + layernorm_flops + gelu_flops + residual_flops
        )
        total_layer_flops = attention_gemm_flops + mlp_gemm_flops + non_gemm_flops
        roi_latency_s = metrics["simSeconds"]
        gemm_active_s = extract_gemm_active_time(serial_log)
        dma_bytes = metrics["dmaRead"] + metrics["dmaWrite"]
        total_movement_bytes = dma_bytes + phase["total_proxy_movement_bytes"]
        compute_time_total_s = (
            (metrics["totalComputeCycles"] if "totalComputeCycles" in metrics else metrics["computeCycles"])
            * 2.0 / (CLOCK_FREQ_GHZ * 1e9)
            if metrics["computeCycles"] > 0
            else 0.0
        )
        gemm_wall_s = gemm_active_s
        gemm_compute_share = (
            100.0 * compute_time_total_s / gemm_wall_s if gemm_wall_s > 0 else 0.0
        )
        gemm_noncompute_share = 100.0 - gemm_compute_share if gemm_wall_s > 0 else 0.0

        peak_gflops = (
            ((proxy_gemm_flops / 2.0) / 1e9)
            / (metrics["computeCycles"] / (CLOCK_FREQ_GHZ * 1e9))
            if metrics["computeCycles"] > 0
            else 0.0
        )
        gemm_effective_gflops = (
            (proxy_gemm_flops / 1e9) / gemm_active_s
            if gemm_active_s > 0
            else 0.0
        )
        layer_gemm_normalized_gflops = (
            ((attention_gemm_flops + mlp_gemm_flops) / 1e9) / gemm_active_s
            if gemm_active_s > 0
            else 0.0
        )
        system_effective_gflops = (
            (total_layer_flops / 1e9) / roi_latency_s if roi_latency_s > 0 else 0.0
        )
        oi = total_layer_flops / total_movement_bytes if total_movement_bytes > 0 else 0.0

        def avg_size(byte_key, acc_key):
            acc = phase[acc_key]
            return phase[byte_key] / acc if acc > 0 else 0.0

        result_row = {
            "Preset": preset["name"],
            "SeqLen": preset["seq_len"],
            "HiddenDim": preset["hidden_dim"],
            "MLPDim": preset["mlp_dim"],
            "NumHeads": preset["num_heads"],
            "Phase2 Mode": phase["phase2_mode"],
            "Staged Block Bytes": phase["staged_block_bytes"],
            "GEMM Proxy Size": seq_len,
            "Proxy GEMM FLOPs": int(proxy_gemm_flops),
            "Attention GEMM FLOPs": int(attention_gemm_flops),
            "MLP GEMM FLOPs": int(mlp_gemm_flops),
            "Non-GEMM FLOPs": int(non_gemm_flops),
            "Total FLOPs": int(total_layer_flops),
            "End-to-End ROI Latency (s)": round(roi_latency_s, 6),
            "Total Latency (ms)": round(phase["end_to_end_ms"], 6),
            "Phase 2 Share of Total (%)": round(
                100.0 * phase["phase2_total_ms"] / phase["end_to_end_ms"]
                if phase["end_to_end_ms"] > 0
                else 0.0,
                4,
            ),
            "GEMM Tiles Per GEMM": phase["gemm_tile_count_per_gemm"],
            "GEMM Total Tiles": phase["gemm_tile_count_total"],
            "Phase-1 GEMM Time (ms)": round(phase["phase1_ms"], 6),
            "D2H Copy Time (ms)": round(phase["phase2_d2h_ms"], 6),
            "H2D Copy Time (ms)": round(phase["phase2_h2d_ms"], 6),
            "CXL In-Place Phase Time (ms)": round(
                phase["phase2_cxl_inplace_ms"], 6
            ),
            "Expand Time (ms)": round(phase["phase2_expand_ms"], 6),
            "Softmax Time (ms)": round(phase["phase2_softmax_ms"], 6),
            "LayerNorm Time (ms)": round(phase["phase2_layernorm_ms"], 6),
            "Project Time (ms)": round(phase["phase2_project_ms"], 6),
            "GeLU Time (ms)": round(phase["phase2_gelu_ms"], 6),
            "Residual Time (ms)": round(phase["phase2_residual_ms"], 6),
            "Compact Time (ms)": round(phase["phase2_compact_ms"], 6),
            "Non-GEMM Time (ms)": round(phase["phase2_non_gemm_ms"], 6),
            "Phase-2 Total (ms)": round(phase["phase2_total_ms"], 6),
            "Phase-3 GEMM Time (ms)": round(phase["phase3_ms"], 6),
            "Descriptor Launch Count": phase["descriptor_launch_count"],
            "Doorbell Launch Count": phase["doorbell_launch_count"],
            "Phase1 Poll Count": phase["phase1_poll_count"],
            "Phase3 Poll Count": phase["phase3_poll_count"],
            "Device DMA Read Bytes": metrics["dmaRead"],
            "Device DMA Write Bytes": metrics["dmaWrite"],
            "GEMM DMA Bytes": dma_bytes,
            "Host-Mediated Copy Bytes": phase["host_mediated_copy_bytes"],
            "CXL In-Place Access Bytes": phase["cxl_inplace_access_bytes"],
            "Score Read Bytes": phase["score_read_bytes"],
            "Score Write Bytes": phase["score_write_bytes"],
            "Score Read Accesses": phase["score_read_accesses"],
            "Score Write Accesses": phase["score_write_accesses"],
            "Score Avg Read Size (B/access)": round(avg_size("score_read_bytes", "score_read_accesses"), 4),
            "Score Avg Write Size (B/access)": round(avg_size("score_write_bytes", "score_write_accesses"), 4),
            "Hidden Read Bytes": phase["hidden_read_bytes"],
            "Hidden Write Bytes": phase["hidden_write_bytes"],
            "Hidden Read Accesses": phase["hidden_read_accesses"],
            "Hidden Write Accesses": phase["hidden_write_accesses"],
            "Hidden Avg Read Size (B/access)": round(avg_size("hidden_read_bytes", "hidden_read_accesses"), 4),
            "Hidden Avg Write Size (B/access)": round(avg_size("hidden_write_bytes", "hidden_write_accesses"), 4),
            "Residual Read Bytes": phase["residual_read_bytes"],
            "Residual Write Bytes": phase["residual_write_bytes"],
            "Residual Read Accesses": phase["residual_read_accesses"],
            "Residual Write Accesses": phase["residual_write_accesses"],
            "Residual Avg Read Size (B/access)": round(avg_size("residual_read_bytes", "residual_read_accesses"), 4),
            "Residual Avg Write Size (B/access)": round(avg_size("residual_write_bytes", "residual_write_accesses"), 4),
            "MLP Read Bytes": phase["mlp_read_bytes"],
            "MLP Write Bytes": phase["mlp_write_bytes"],
            "MLP Read Accesses": phase["mlp_read_accesses"],
            "MLP Write Accesses": phase["mlp_write_accesses"],
            "MLP Avg Read Size (B/access)": round(avg_size("mlp_read_bytes", "mlp_read_accesses"), 4),
            "MLP Avg Write Size (B/access)": round(avg_size("mlp_write_bytes", "mlp_write_accesses"), 4),
            "Proxy Remote Footprint Bytes": phase["proxy_remote_footprint_bytes"],
            "Phase2 Total Bytes": phase["total_proxy_movement_bytes"],
            "Total Data Movement Bytes": total_movement_bytes,
            "totalComputeCycles": metrics["computeCycles"],
            "GEMM Compute Share (%)": round(gemm_compute_share, 4),
            "GEMM Non-Compute Share (%)": round(gemm_noncompute_share, 4),
            "Operational Intensity (OI)": round(oi, 4),
            "Peak MAC Throughput (GFLOPS)": round(peak_gflops, 6),
            "GEMM-Effective (GFLOPS)": round(gemm_effective_gflops, 6),
            "Layer-GEMM-Normalized (GFLOPS)": round(
                layer_gemm_normalized_gflops, 6
            ),
            "End-to-End Effective (GFLOPS)": round(
                system_effective_gflops, 6
            ),
        }
        results.append(result_row)

        print(
            f"[√] {preset['name']} 完成! "
            f"Phase1={result_row['Phase-1 GEMM Time (ms)']:.3f}ms, "
            f"Phase2={result_row['Phase-2 Total (ms)']:.3f}ms, "
            f"GEMM-Effective={result_row['GEMM-Effective (GFLOPS)']:.3f} GFLOPS, "
            f"End-to-End Effective={result_row['End-to-End Effective (GFLOPS)']:.3f} GFLOPS"
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
        print(f"🎉 Sweep 完成！数据已保存至: {CSV_FILE}")


if __name__ == "__main__":
    main()
