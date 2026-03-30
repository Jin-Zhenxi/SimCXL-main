#!/usr/bin/env python3
"""
ViT-inspired layer proxy sweep for the CXL Type-3 + device-side DDR5 HDM branch.
This is not a full-model ViT runtime.
"""
import csv
import math
import os
import re
import shutil
import subprocess
from collections import Counter

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
]
OUTPUT_DIR = "my_outputData"
C_FILE = "trigger_gemm.c"
CSV_FILE = os.path.join(OUTPUT_DIR, "roofline_data.csv")
RAW_TXT_FILE = os.path.join(OUTPUT_DIR, "summary_raw.txt")
DEFAULT_PHASE2_MODE = "staged_block"
DEFAULT_STAGED_BLOCK_BYTES = 1024
DEFAULT_MIN_READ_REQUEST_BYTES = 64
MAX_INVALID_RUN_RETRIES = 3
PREFETCH_CONFIGS = [
    {
        "label": "tail_baseline",
        "mode": "b_only",
        "trigger": "compute_launch",
        "rows_a": 0,
        "rows_b": 32,
        "rows_b_by_preset": {
            "ViT-Base-like": 48,
            "ViT-Large-like": 128,
        },
        "carry_over_max_rows": 0,
        "carry_over_inherit_inflight": 0,
        "hole_fill_lead_rows": 32,
        "writec_overlap_b_issue_budget_rows": 0,
    },
]

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
    with open(C_FILE) as f:
        content = f.read()

    replacements = {
        r"uint32_t default_seq_len = \d+;": f"uint32_t default_seq_len = {preset['seq_len']};",
        r"uint32_t default_hidden_dim = \d+;": f"uint32_t default_hidden_dim = {preset['hidden_dim']};",
        r"uint32_t default_mlp_dim = \d+;": f"uint32_t default_mlp_dim = {preset['mlp_dim']};",
        r"uint32_t default_num_heads = \d+;": f"uint32_t default_num_heads = {preset['num_heads']};",
        r'const char \*default_phase2_mode = "[^"]+";': f'const char *default_phase2_mode = "{DEFAULT_PHASE2_MODE}";',
        r"uint32_t default_staged_block_bytes = \d+;": f"uint32_t default_staged_block_bytes = {DEFAULT_STAGED_BLOCK_BYTES};",
        r'\.preset_name = "[^"]+",': f'.preset_name = "{preset["name"]}",',
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

    with open(stats_file) as f:
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
        with open(log_file) as f:
            log_text = f.read()
            match_cycles = re.search(r"computeCycles=(\d+)", log_text)
            if match_cycles:
                metrics["computeCycles"] = int(match_cycles.group(1))

    return metrics


def extract_bus_dma_stats(stats_file):
    stats = {
        "bus_dma_pkt_count": 0,
        "bus_dma_pkt_bytes": 0,
        "mem_rsp_gap_mean_cycles": 0.0,
        "mem_rsp_gap_stdev_cycles": 0.0,
        "mem_rsp_gap_min_cycles": 0.0,
        "mem_rsp_gap_max_cycles": 0.0,
    }
    if not os.path.exists(stats_file):
        return stats

    patterns = {
        "bus_dma_pkt_count": r"cxl_mem_bus\.pktCount_.*matrix_engine\.dma::total\s+(\d+)",
        "bus_dma_pkt_bytes": r"cxl_mem_bus\.pktSize_.*matrix_engine\.dma::total\s+(\d+)",
        "mem_rsp_gap_mean_cycles": r"memToCXLCtrlRsp::mean\s+([0-9.]+)",
        "mem_rsp_gap_stdev_cycles": r"memToCXLCtrlRsp::stdev\s+([0-9.]+)",
        "mem_rsp_gap_min_cycles": r"memToCXLCtrlRsp::min_value\s+([0-9.]+)",
        "mem_rsp_gap_max_cycles": r"memToCXLCtrlRsp::max_value\s+([0-9.]+)",
    }

    text = open(stats_file, encoding="utf-8", errors="ignore").read()
    for key, pattern in patterns.items():
        m = re.search(pattern, text)
        if not m:
            continue
        stats[key] = (
            float(m.group(1)) if "." in m.group(1) else int(m.group(1))
        )
    return stats


def extract_prefetch_stats(stats_file):
    stats = {
        "next_prefetch_issue_count": 0,
        "next_k_prefetch_issue_count": 0,
        "next_output_prefetch_issue_count": 0,
        "next_prefetch_hit_count": 0,
        "next_output_prefetch_hit_count": 0,
        "next_prefetch_fallback_count": 0,
        "next_output_prefetch_fallback_count": 0,
        "next_prefetch_late_completion_count": 0,
        "next_output_prefetch_late_completion_count": 0,
        "next_prefetch_discard_count": 0,
        "prefetched_b_rows_consumed": 0,
        "fallback_b_rows_fetched": 0,
        "next_output_prefetch_rows_issued": 0,
        "next_k_prefetch_rows_issued": 0,
        "next_output_prefetch_defer_count": 0,
        "next_output_headstart_cycles": 0,
        "next_output_first_issue_to_boundary_cycles": 0,
        "next_output_rows_ready_at_boundary": 0,
        "next_output_consumed_before_fallback_rows": 0,
        "carry_over_rows_at_boundary": 0,
        "carry_over_inflight_rows_at_boundary": 0,
        "carry_over_rows_consumed_post_boundary": 0,
        "normal_fetch_hole_rows": 0,
        "duplicate_b_row_fetch_avoided": 0,
        "duplicate_b_row_fetch_detected": 0,
        "carry_over_late_completion_count": 0,
        "normal_fetch_deferred_by_carry": 0,
        "writec_overlap_cycles": 0,
        "writec_overlap_enabled_count": 0,
        "writec_overlap_success_count": 0,
        "next_output_progress_during_writec": 0,
        "b_rows_issued_during_writec": 0,
        "writec_blocked_b_issue_count": 0,
    }
    if not os.path.exists(stats_file):
        return stats

    patterns = {
        "next_prefetch_issue_count": r"matrix_engine\.nextPrefetchIssueCount\s+(\d+)",
        "next_k_prefetch_issue_count": r"matrix_engine\.nextKPrefetchIssueCount\s+(\d+)",
        "next_output_prefetch_issue_count": r"matrix_engine\.nextOutputPrefetchIssueCount\s+(\d+)",
        "next_prefetch_hit_count": r"matrix_engine\.nextPrefetchHitCount\s+(\d+)",
        "next_output_prefetch_hit_count": r"matrix_engine\.nextOutputPrefetchHitCount\s+(\d+)",
        "next_prefetch_fallback_count": r"matrix_engine\.nextPrefetchFallbackCount\s+(\d+)",
        "next_output_prefetch_fallback_count": r"matrix_engine\.nextOutputPrefetchFallbackCount\s+(\d+)",
        "next_prefetch_late_completion_count": r"matrix_engine\.nextPrefetchLateCompletionCount\s+(\d+)",
        "next_output_prefetch_late_completion_count": r"matrix_engine\.nextOutputPrefetchLateCompletionCount\s+(\d+)",
        "next_prefetch_discard_count": r"matrix_engine\.nextPrefetchDiscardCount\s+(\d+)",
        "prefetched_b_rows_consumed": r"matrix_engine\.prefetchedBRowsConsumed\s+(\d+)",
        "fallback_b_rows_fetched": r"matrix_engine\.fallbackBRowsFetched\s+(\d+)",
        "next_output_prefetch_rows_issued": r"matrix_engine\.nextOutputPrefetchRowsIssued\s+(\d+)",
        "next_k_prefetch_rows_issued": r"matrix_engine\.nextKPrefetchRowsIssued\s+(\d+)",
        "next_output_prefetch_defer_count": r"matrix_engine\.nextOutputPrefetchDeferCount\s+(\d+)",
        "next_output_headstart_cycles": r"matrix_engine\.nextOutputHeadstartCycles\s+(\d+)",
        "next_output_first_issue_to_boundary_cycles": r"matrix_engine\.nextOutputFirstIssueToBoundaryCycles\s+(\d+)",
        "next_output_rows_ready_at_boundary": r"matrix_engine\.nextOutputRowsReadyAtBoundary\s+(\d+)",
        "next_output_consumed_before_fallback_rows": r"matrix_engine\.nextOutputConsumedBeforeFallbackRows\s+(\d+)",
        "carry_over_rows_at_boundary": r"matrix_engine\.carryOverRowsAtBoundary\s+(\d+)",
        "carry_over_inflight_rows_at_boundary": r"matrix_engine\.carryOverInflightRowsAtBoundary\s+(\d+)",
        "carry_over_rows_consumed_post_boundary": r"matrix_engine\.carryOverRowsConsumedPostBoundary\s+(\d+)",
        "normal_fetch_hole_rows": r"matrix_engine\.normalFetchHoleRows\s+(\d+)",
        "duplicate_b_row_fetch_avoided": r"matrix_engine\.duplicateBRowFetchAvoided\s+(\d+)",
        "duplicate_b_row_fetch_detected": r"matrix_engine\.duplicateBRowFetchDetected\s+(\d+)",
        "carry_over_late_completion_count": r"matrix_engine\.carryOverLateCompletionCount\s+(\d+)",
        "normal_fetch_deferred_by_carry": r"matrix_engine\.normalFetchDeferredByCarry\s+(\d+)",
        "writec_overlap_cycles": r"matrix_engine\.writeCOverlapCycles\s+(\d+)",
        "writec_overlap_enabled_count": r"matrix_engine\.writeCOverlapEnabledCount\s+(\d+)",
        "writec_overlap_success_count": r"matrix_engine\.writeCOverlapSuccessCount\s+(\d+)",
        "next_output_progress_during_writec": r"matrix_engine\.nextOutputProgressDuringWriteC\s+(\d+)",
        "b_rows_issued_during_writec": r"matrix_engine\.bRowsIssuedDuringWriteC\s+(\d+)",
        "writec_blocked_b_issue_count": r"matrix_engine\.writeCBlockedBIssueCount\s+(\d+)",
    }
    text = open(stats_file, encoding="utf-8", errors="ignore").read()
    for key, pattern in patterns.items():
        matches = re.findall(pattern, text)
        if matches:
            stats[key] = int(matches[-1])
    return stats


def histogram_to_str(counter):
    parts = []
    for size in sorted(counter):
        parts.append(f"{size}B:{counter[size]}")
    return "; ".join(parts)


def compute_request_formation(preset):
    seq_len = preset["seq_len"]
    elem_bytes = 4
    tile_dim = 128
    min_read = DEFAULT_MIN_READ_REQUEST_BYTES
    raw = Counter()
    packetized = Counter()

    def add_packetized(req_bytes):
        remaining = req_bytes
        while remaining > 0:
            chunk = min(64, remaining)
            packetized[chunk] += 1
            remaining -= chunk

    # One GEMM only; run_sweep separately reports per-gemm and total counts.
    raw[40] += 1  # descriptor read
    add_packetized(40)
    raw[8] += 1  # completion flag write
    add_packetized(8)

    for i in range(0, seq_len, tile_dim):
        cur_m = min(tile_dim, seq_len - i)
        for j in range(0, seq_len, tile_dim):
            cur_n = min(tile_dim, seq_len - j)
            for k in range(0, seq_len, tile_dim):
                cur_k = min(tile_dim, seq_len - k)

                # A read rows
                row_a = cur_k * elem_bytes
                req_a = max(row_a, min_read) if row_a < min_read else row_a
                raw[row_a] += cur_m
                for _ in range(cur_m):
                    add_packetized(req_a)

                # B read rows
                row_b = cur_n * elem_bytes
                req_b = max(row_b, min_read) if row_b < min_read else row_b
                raw[row_b] += cur_k
                for _ in range(cur_k):
                    add_packetized(req_b)

                # C write rows at end of k-loop only, no flooring yet
                if k + cur_k >= seq_len:
                    row_c = cur_n * elem_bytes
                    raw[row_c] += cur_m
                    for _ in range(cur_m):
                        add_packetized(row_c)

    raw_total_count = sum(raw.values())
    raw_total_bytes = sum(size * count for size, count in raw.items())
    pkt_total_count = sum(packetized.values())
    pkt_total_bytes = sum(size * count for size, count in packetized.items())
    return {
        "raw_hist": histogram_to_str(raw),
        "raw_count": raw_total_count,
        "raw_bytes": raw_total_bytes,
        "raw_avg_size_B": (raw_total_bytes / raw_total_count)
        if raw_total_count
        else 0.0,
        "packet_hist_approx": histogram_to_str(packetized),
        "packet_count_approx": pkt_total_count,
        "packet_bytes_approx": pkt_total_bytes,
        "packet_avg_size_B_approx": (pkt_total_bytes / pkt_total_count)
        if pkt_total_count
        else 0.0,
    }


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

    with open(serial_log_file, encoding="utf-8", errors="ignore") as f:
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


def serial_log_has_benchmark(serial_log_file):
    if not os.path.exists(serial_log_file):
        return False
    text = open(serial_log_file, encoding="utf-8", errors="ignore").read()
    return (
        "ViT-inspired Layer Proxy Benchmark" in text
        and "[Phase 1] GEMM1 on device HDM" in text
    )


def write_partial_results(results):
    if not results:
        return
    headers = results[0].keys()
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results)

    with open(RAW_TXT_FILE, "w") as f:
        for r in results:
            f.write(str(r) + "\n")


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
    print(
        "   预取配置: "
        + ", ".join(
            f"{cfg['label']}[mode={cfg['mode']},trigger={cfg['trigger']},"
            f"rowsA={cfg['rows_a']},rowsB={cfg.get('rows_b_by_preset', cfg['rows_b'])},"
            f"carryMax={cfg.get('carry_over_max_rows', 0)},"
            f"carryInflight={cfg.get('carry_over_inherit_inflight', 0)},"
            f"holeLead={cfg.get('hole_fill_lead_rows', 0)},"
            f"writeCBudget={cfg.get('writec_overlap_b_issue_budget_rows', 0)}]"
            for cfg in PREFETCH_CONFIGS
        )
    )
    print("=" * 50)

    ensure_libm5()

    for cfg in PREFETCH_CONFIGS:
        print(
            f"\n=== 预取配置: {cfg['label']} "
            f"(mode={cfg['mode']}, trigger={cfg['trigger']}, "
            f"rowsA={cfg['rows_a']}, rowsB={cfg['rows_b']}) ==="
        )
        for preset in PRESETS:
            rows_b = cfg.get("rows_b_by_preset", {}).get(
                preset["name"], cfg["rows_b"]
            )
            preset_tag = preset["name"].replace(" ", "_").replace("/", "_")
            run_tag = f"{preset_tag}__{cfg['label']}"
            seq_len = preset["seq_len"]

            print(f"\n>>> 正在测试预设: {preset['name']} / 配置: {cfg['label']} <<<")
            modify_c_file(preset)

            print("[*] 正在编译 trigger_gemm (需 libm5.a + libm)...")
            subprocess.run(COMPILE_CMD, shell=True, check=True)

            inject_cmd = f"{INJECT_CMD} --no-compile {seq_len}"
            print(
                "[*] 正在注入镜像 "
                f"(script.sh 将执行 trigger_gemm 0x200000000 {seq_len})..."
            )
            subprocess.run(inject_cmd, shell=True, check=True)

            env = os.environ.copy()
            env["MATRIXFLOW_NEXT_PREFETCH_MODE"] = cfg["mode"]
            env["MATRIXFLOW_NEXT_PREFETCH_TRIGGER"] = cfg["trigger"]
            env["MATRIXFLOW_NEXT_PREFETCH_ROWS_A"] = str(cfg["rows_a"])
            env["MATRIXFLOW_NEXT_PREFETCH_ROWS_B"] = str(rows_b)
            env["MATRIXFLOW_CARRY_OVER_MAX_ROWS"] = str(
                cfg.get("carry_over_max_rows", 0)
            )
            env["MATRIXFLOW_CARRY_OVER_INHERIT_INFLIGHT"] = str(
                cfg.get("carry_over_inherit_inflight", 0)
            )
            env["MATRIXFLOW_HOLE_FILL_LEAD_ROWS"] = str(
                cfg.get("hole_fill_lead_rows", 0)
            )
            env["MATRIXFLOW_WRITEC_OVERLAP_B_ISSUE_BUDGET_ROWS"] = str(
                cfg.get("writec_overlap_b_issue_budget_rows", 0)
            )

            metrics = None
            bus_dma = None
            prefetch = None
            serial_log = None
            for attempt in range(1, MAX_INVALID_RUN_RETRIES + 1):
                m5out_dir = os.path.join(OUTPUT_DIR, f"m5out_{run_tag}")
                log_file = os.path.join(
                    OUTPUT_DIR, f"terminal_log_{run_tag}.txt"
                )
                if os.path.exists(m5out_dir):
                    shutil.rmtree(m5out_dir)
                if os.path.exists(log_file):
                    os.remove(log_file)

                cmd = GEM5_CMD_TEMPLATE.format(
                    outdir=m5out_dir, logfile=log_file
                )
                print(
                    f"[*] 正在运行 gem5 仿真，日志存入: {log_file} "
                    f"(attempt {attempt}/{MAX_INVALID_RUN_RETRIES})"
                )
                rc = subprocess.run(cmd, shell=True, env=env)
                if rc.returncode != 0:
                    raise SystemExit(
                        f"gem5 仿真失败，返回码={rc.returncode}。请检查日志: {log_file}"
                    )

                stats_file = os.path.join(m5out_dir, "stats.txt")
                serial_log = os.path.join(m5out_dir, "board.pc.com_1.device")
                metrics = extract_metrics(stats_file, log_file)
                bus_dma = extract_bus_dma_stats(stats_file)
                prefetch = extract_prefetch_stats(stats_file)
                valid_metrics = not (
                    metrics["simSeconds"] == 0.0
                    and metrics["dmaRead"] == 0
                    and metrics["dmaWrite"] == 0
                    and metrics["computeCycles"] == 0
                )
                valid_serial = serial_log_has_benchmark(serial_log)
                if valid_metrics and valid_serial:
                    break

                print(
                    "[!] 无效 run：未进入 benchmark 或未写出有效 ROI 统计，" f"准备重试 {run_tag}"
                )
            else:
                raise SystemExit(
                    f"多次重试后仍未拿到有效结果: {run_tag}。" f"请检查日志: {log_file}"
                )

            phase = extract_phase_timings(serial_log)
            formation = compute_request_formation(preset)
            proxy_gemm_flops = 4.0 * (seq_len**3)
            attention_gemm_flops = (
                4.0
                * preset["seq_len"]
                * preset["seq_len"]
                * preset["hidden_dim"]
            )
            mlp_gemm_flops = (
                4.0
                * preset["seq_len"]
                * preset["hidden_dim"]
                * preset["mlp_dim"]
            )
            softmax_flops = 5.0 * preset["seq_len"] * preset["seq_len"]
            layernorm_flops = 6.0 * preset["seq_len"] * preset["hidden_dim"]
            gelu_flops = 8.0 * preset["seq_len"] * preset["mlp_dim"]
            residual_flops = 1.0 * preset["seq_len"] * preset["hidden_dim"]
            non_gemm_flops = (
                softmax_flops + layernorm_flops + gelu_flops + residual_flops
            )
            total_layer_flops = (
                attention_gemm_flops + mlp_gemm_flops + non_gemm_flops
            )
            roi_latency_s = metrics["simSeconds"]
            gemm_active_s = extract_gemm_active_time(serial_log)
            dma_bytes = metrics["dmaRead"] + metrics["dmaWrite"]
            total_movement_bytes = (
                dma_bytes + phase["total_proxy_movement_bytes"]
            )
            compute_time_total_s = (
                (
                    metrics["totalComputeCycles"]
                    if "totalComputeCycles" in metrics
                    else metrics["computeCycles"]
                )
                * 2.0
                / (CLOCK_FREQ_GHZ * 1e9)
                if metrics["computeCycles"] > 0
                else 0.0
            )
            gemm_wall_s = gemm_active_s
            gemm_compute_share = (
                100.0 * compute_time_total_s / gemm_wall_s
                if gemm_wall_s > 0
                else 0.0
            )
            gemm_noncompute_share = (
                100.0 - gemm_compute_share if gemm_wall_s > 0 else 0.0
            )

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
            gemm_dma_bw_gbps = (
                (dma_bytes / 1e9) / gemm_active_s if gemm_active_s > 0 else 0.0
            )
            layer_gemm_normalized_gflops = (
                ((attention_gemm_flops + mlp_gemm_flops) / 1e9) / gemm_active_s
                if gemm_active_s > 0
                else 0.0
            )
            system_effective_gflops = (
                (total_layer_flops / 1e9) / roi_latency_s
                if roi_latency_s > 0
                else 0.0
            )
            oi = (
                total_layer_flops / total_movement_bytes
                if total_movement_bytes > 0
                else 0.0
            )

            def avg_size(byte_key, acc_key):
                acc = phase[acc_key]
                return phase[byte_key] / acc if acc > 0 else 0.0

            result_row = {
                "Preset": preset["name"],
                "Prefetch Label": cfg["label"],
                "Prefetch Mode": cfg["mode"],
                "Prefetch Trigger": cfg["trigger"],
                "Prefetch Rows A": cfg["rows_a"],
                "Prefetch Rows B": rows_b,
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
                "next_prefetch_issue_count": prefetch[
                    "next_prefetch_issue_count"
                ],
                "next_k_prefetch_issue_count": prefetch[
                    "next_k_prefetch_issue_count"
                ],
                "next_output_prefetch_issue_count": prefetch[
                    "next_output_prefetch_issue_count"
                ],
                "next_prefetch_hit_count": prefetch["next_prefetch_hit_count"],
                "next_output_prefetch_hit_count": prefetch[
                    "next_output_prefetch_hit_count"
                ],
                "next_prefetch_fallback_count": prefetch[
                    "next_prefetch_fallback_count"
                ],
                "next_output_prefetch_fallback_count": prefetch[
                    "next_output_prefetch_fallback_count"
                ],
                "next_prefetch_late_completion_count": prefetch[
                    "next_prefetch_late_completion_count"
                ],
                "next_output_prefetch_late_completion_count": prefetch[
                    "next_output_prefetch_late_completion_count"
                ],
                "next_prefetch_discard_count": prefetch[
                    "next_prefetch_discard_count"
                ],
                "prefetched_b_rows_consumed": prefetch[
                    "prefetched_b_rows_consumed"
                ],
                "fallback_b_rows_fetched": prefetch["fallback_b_rows_fetched"],
                "next_output_prefetch_rows_issued": prefetch[
                    "next_output_prefetch_rows_issued"
                ],
                "next_k_prefetch_rows_issued": prefetch[
                    "next_k_prefetch_rows_issued"
                ],
                "next_output_prefetch_defer_count": prefetch[
                    "next_output_prefetch_defer_count"
                ],
                "next_output_headstart_cycles": prefetch[
                    "next_output_headstart_cycles"
                ],
                "next_output_first_issue_to_boundary_cycles": prefetch[
                    "next_output_first_issue_to_boundary_cycles"
                ],
                "next_output_rows_ready_at_boundary": prefetch[
                    "next_output_rows_ready_at_boundary"
                ],
                "next_output_consumed_before_fallback_rows": prefetch[
                    "next_output_consumed_before_fallback_rows"
                ],
                "carry_over_rows_at_boundary": prefetch[
                    "carry_over_rows_at_boundary"
                ],
                "carry_over_inflight_rows_at_boundary": prefetch[
                    "carry_over_inflight_rows_at_boundary"
                ],
                "carry_over_rows_consumed_post_boundary": prefetch[
                    "carry_over_rows_consumed_post_boundary"
                ],
                "normal_fetch_hole_rows": prefetch["normal_fetch_hole_rows"],
                "duplicate_b_row_fetch_avoided": prefetch[
                    "duplicate_b_row_fetch_avoided"
                ],
                "duplicate_b_row_fetch_detected": prefetch[
                    "duplicate_b_row_fetch_detected"
                ],
                "carry_over_late_completion_count": prefetch[
                    "carry_over_late_completion_count"
                ],
                "normal_fetch_deferred_by_carry": prefetch[
                    "normal_fetch_deferred_by_carry"
                ],
                "writec_overlap_cycles": prefetch["writec_overlap_cycles"],
                "writec_overlap_enabled_count": prefetch[
                    "writec_overlap_enabled_count"
                ],
                "writec_overlap_success_count": prefetch[
                    "writec_overlap_success_count"
                ],
                "next_output_progress_during_writec": prefetch[
                    "next_output_progress_during_writec"
                ],
                "b_rows_issued_during_writec": prefetch[
                    "b_rows_issued_during_writec"
                ],
                "writec_blocked_b_issue_count": prefetch[
                    "writec_blocked_b_issue_count"
                ],
                "Raw DMA Action Histogram": formation["raw_hist"],
                "Raw DMA Action Count": formation["raw_count"],
                "Raw DMA Action Bytes": formation["raw_bytes"],
                "Raw DMA Action Avg Size (B)": round(
                    formation["raw_avg_size_B"], 4
                ),
                "Approx 64B Packet Histogram": formation["packet_hist_approx"],
                "Approx Packet Count": formation["packet_count_approx"],
                "Approx Packet Bytes": formation["packet_bytes_approx"],
                "Approx Packet Avg Size (B)": round(
                    formation["packet_avg_size_B_approx"], 4
                ),
                "Bus DMA Packet Count": bus_dma["bus_dma_pkt_count"],
                "Bus DMA Packet Bytes": bus_dma["bus_dma_pkt_bytes"],
                "Bus Avg Packet Size (B)": round(
                    (
                        bus_dma["bus_dma_pkt_bytes"]
                        / bus_dma["bus_dma_pkt_count"]
                    )
                    if bus_dma["bus_dma_pkt_count"] > 0
                    else 0.0,
                    4,
                ),
                "Bus Avg Request Payload Size (B)": round(
                    (
                        bus_dma["bus_dma_pkt_bytes"]
                        / (bus_dma["bus_dma_pkt_count"] / 2.0)
                    )
                    if bus_dma["bus_dma_pkt_count"] > 0
                    else 0.0,
                    4,
                ),
                "memToCXLCtrlRsp mean (cycles)": round(
                    float(bus_dma["mem_rsp_gap_mean_cycles"]), 6
                ),
                "memToCXLCtrlRsp stdev (cycles)": round(
                    float(bus_dma["mem_rsp_gap_stdev_cycles"]), 6
                ),
                "memToCXLCtrlRsp min (cycles)": round(
                    float(bus_dma["mem_rsp_gap_min_cycles"]), 6
                ),
                "memToCXLCtrlRsp max (cycles)": round(
                    float(bus_dma["mem_rsp_gap_max_cycles"]), 6
                ),
                "Host-Mediated Copy Bytes": phase["host_mediated_copy_bytes"],
                "CXL In-Place Access Bytes": phase["cxl_inplace_access_bytes"],
                "Score Read Bytes": phase["score_read_bytes"],
                "Score Write Bytes": phase["score_write_bytes"],
                "Score Read Accesses": phase["score_read_accesses"],
                "Score Write Accesses": phase["score_write_accesses"],
                "Score Avg Read Size (B/access)": round(
                    avg_size("score_read_bytes", "score_read_accesses"), 4
                ),
                "Score Avg Write Size (B/access)": round(
                    avg_size("score_write_bytes", "score_write_accesses"), 4
                ),
                "Hidden Read Bytes": phase["hidden_read_bytes"],
                "Hidden Write Bytes": phase["hidden_write_bytes"],
                "Hidden Read Accesses": phase["hidden_read_accesses"],
                "Hidden Write Accesses": phase["hidden_write_accesses"],
                "Hidden Avg Read Size (B/access)": round(
                    avg_size("hidden_read_bytes", "hidden_read_accesses"), 4
                ),
                "Hidden Avg Write Size (B/access)": round(
                    avg_size("hidden_write_bytes", "hidden_write_accesses"), 4
                ),
                "Residual Read Bytes": phase["residual_read_bytes"],
                "Residual Write Bytes": phase["residual_write_bytes"],
                "Residual Read Accesses": phase["residual_read_accesses"],
                "Residual Write Accesses": phase["residual_write_accesses"],
                "Residual Avg Read Size (B/access)": round(
                    avg_size("residual_read_bytes", "residual_read_accesses"),
                    4,
                ),
                "Residual Avg Write Size (B/access)": round(
                    avg_size(
                        "residual_write_bytes", "residual_write_accesses"
                    ),
                    4,
                ),
                "MLP Read Bytes": phase["mlp_read_bytes"],
                "MLP Write Bytes": phase["mlp_write_bytes"],
                "MLP Read Accesses": phase["mlp_read_accesses"],
                "MLP Write Accesses": phase["mlp_write_accesses"],
                "MLP Avg Read Size (B/access)": round(
                    avg_size("mlp_read_bytes", "mlp_read_accesses"), 4
                ),
                "MLP Avg Write Size (B/access)": round(
                    avg_size("mlp_write_bytes", "mlp_write_accesses"), 4
                ),
                "Proxy Remote Footprint Bytes": phase[
                    "proxy_remote_footprint_bytes"
                ],
                "Phase2 Total Bytes": phase["total_proxy_movement_bytes"],
                "Total Data Movement Bytes": total_movement_bytes,
                "totalComputeCycles": metrics["computeCycles"],
                "GEMM Compute Share (%)": round(gemm_compute_share, 4),
                "GEMM Non-Compute Share (%)": round(gemm_noncompute_share, 4),
                "Operational Intensity (OI)": round(oi, 4),
                "Peak MAC Throughput (GFLOPS)": round(peak_gflops, 6),
                "GEMM DMA BW (GB/s)": round(gemm_dma_bw_gbps, 6),
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
                f"[√] {preset['name']} / {cfg['label']} 完成! "
                f"Phase1={result_row['Phase-1 GEMM Time (ms)']:.3f}ms, "
                f"Phase2={result_row['Phase-2 Total (ms)']:.3f}ms, "
                f"GEMM-Effective={result_row['GEMM-Effective (GFLOPS)']:.3f} GFLOPS, "
                f"End-to-End Effective={result_row['End-to-End Effective (GFLOPS)']:.3f} GFLOPS"
            )
            write_partial_results(results)

    # 5. 保存到 CSV 和 TXT
    if results:
        write_partial_results(results)
        print("\n" + "=" * 50)
        print(f"🎉 Sweep 完成！数据已保存至: {CSV_FILE}")


if __name__ == "__main__":
    main()
