#!/usr/bin/env python3
"""
Carry-over sensitivity analysis around the fixed single-path tail baselines:
- ViT-Base-like @ rows_B = 48
- ViT-Large-like @ rows_B = 128

This is not another rows_B search. It perturbs only carry-over mechanism knobs:
- carry_over_max_rows
- carry_over_inherit_inflight
"""

import csv
import os
import shutil
import subprocess
from pathlib import Path

import run_sweep as rs

OUTPUT_DIR = "my_outputData"
CSV_FILE = os.path.join(OUTPUT_DIR, "carry_sensitivity.csv")
BEST_FILE = os.path.join(OUTPUT_DIR, "carry_sensitivity_best.csv")

PRESETS = [
    {
        "name": "ViT-Base-like",
        "seq_len": 197,
        "hidden_dim": 768,
        "mlp_dim": 3072,
        "num_heads": 12,
        "rows_b": 48,
        "carry_caps": [16, 32, 48, 64, 0],
    },
    {
        "name": "ViT-Large-like",
        "seq_len": 257,
        "hidden_dim": 1024,
        "mlp_dim": 4096,
        "num_heads": 16,
        "rows_b": 128,
        "carry_caps": [32, 64, 96, 128, 0],
    },
]

INHERIT_INFLIGHT_OPTIONS = [0, 1]
MAX_INVALID_RUN_RETRIES = 3


def write_results(path, rows):
    if not rows:
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=fieldnames, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def score_row(row):
    return (
        float(row["GEMM Compute Share (%)"]),
        float(row["GEMM-Effective (GFLOPS)"]),
        -float(row["Phase1/3 time (ms)"]),
        float(row["carry_over_rows_consumed_post_boundary"]),
        -float(row["fallback_b_rows_fetched"]),
    )


def pick_best_rows(rows):
    by_preset = {}
    for row in rows:
        by_preset.setdefault(row["Preset"], []).append(row)
    best = []
    for preset, items in by_preset.items():
        items.sort(key=score_row, reverse=True)
        best.append(items[0])
    return best


def run_one(preset, carry_cap, inherit_inflight):
    cfg_label = f"carry_cap_{carry_cap if carry_cap != 0 else 'all'}__inherit_{inherit_inflight}"
    preset_tag = preset["name"].replace(" ", "_").replace("/", "_")
    run_tag = f"{preset_tag}__{cfg_label}"

    rs.modify_c_file(preset)

    print(
        f"\n>>> Carry sensitivity: {preset['name']} "
        f"rows_b={preset['rows_b']} carry_cap={carry_cap} "
        f"inherit_inflight={inherit_inflight} <<<"
    )
    print("[*] 正在编译 trigger_gemm...")
    subprocess.run(rs.COMPILE_CMD, shell=True, check=True)

    inject_cmd = f"{rs.INJECT_CMD} --no-compile {preset['seq_len']}"
    print(f"[*] 正在注入镜像 (seq_len={preset['seq_len']})...")
    subprocess.run(inject_cmd, shell=True, check=True)

    env = os.environ.copy()
    env["MATRIXFLOW_NEXT_PREFETCH_MODE"] = "b_only"
    env["MATRIXFLOW_NEXT_PREFETCH_TRIGGER"] = "compute_launch"
    env["MATRIXFLOW_NEXT_PREFETCH_ROWS_A"] = "0"
    env["MATRIXFLOW_NEXT_PREFETCH_ROWS_B"] = str(preset["rows_b"])
    env["MATRIXFLOW_CARRY_OVER_MAX_ROWS"] = str(carry_cap)
    env["MATRIXFLOW_CARRY_OVER_INHERIT_INFLIGHT"] = str(inherit_inflight)

    for attempt in range(1, MAX_INVALID_RUN_RETRIES + 1):
        m5out_dir = os.path.join(OUTPUT_DIR, f"m5out_{run_tag}")
        log_file = os.path.join(OUTPUT_DIR, f"terminal_log_{run_tag}.txt")
        if os.path.exists(m5out_dir):
            shutil.rmtree(m5out_dir)
        if os.path.exists(log_file):
            os.remove(log_file)

        cmd = rs.GEM5_CMD_TEMPLATE.format(outdir=m5out_dir, logfile=log_file)
        print(
            f"[*] 正在运行 gem5 仿真，日志存入: {log_file} (attempt {attempt}/{MAX_INVALID_RUN_RETRIES})"
        )
        rc = subprocess.run(cmd, shell=True, env=env)
        if rc.returncode != 0:
            raise SystemExit(
                f"gem5 仿真失败，返回码={rc.returncode}。请检查日志: {log_file}"
            )

        stats_file = os.path.join(m5out_dir, "stats.txt")
        serial_log = os.path.join(m5out_dir, "board.pc.com_1.device")
        metrics = rs.extract_metrics(stats_file, log_file)
        bus_dma = rs.extract_bus_dma_stats(stats_file)
        prefetch = rs.extract_prefetch_stats(stats_file)
        valid_metrics = not (
            metrics["simSeconds"] == 0.0
            and metrics["dmaRead"] == 0
            and metrics["dmaWrite"] == 0
            and metrics["computeCycles"] == 0
        )
        valid_serial = rs.serial_log_has_benchmark(serial_log)
        if valid_metrics and valid_serial:
            break
        print(f"[!] 无效 run：{run_tag}，准备重试")
    else:
        raise SystemExit(f"多次重试后仍未拿到有效结果: {run_tag}")

    phase = rs.extract_phase_timings(serial_log)
    gemm_active_s = rs.extract_gemm_active_time(serial_log)
    dma_bytes = metrics["dmaRead"] + metrics["dmaWrite"]
    compute_time_total_s = (
        metrics["computeCycles"] * 2.0 / (rs.CLOCK_FREQ_GHZ * 1e9)
        if metrics["computeCycles"] > 0
        else 0.0
    )
    gemm_compute_share = (
        100.0 * compute_time_total_s / gemm_active_s
        if gemm_active_s > 0
        else 0.0
    )
    gemm_effective_gflops = (
        (4.0 * (preset["seq_len"] ** 3) / 1e9) / gemm_active_s
        if gemm_active_s > 0
        else 0.0
    )
    gemm_dma_bw = (
        (dma_bytes / 1e9) / gemm_active_s if gemm_active_s > 0 else 0.0
    )

    return {
        "Preset": preset["name"],
        "SeqLen": preset["seq_len"],
        "Prefetch Rows B": preset["rows_b"],
        "carry_over_max_rows": carry_cap,
        "carry_over_inherit_inflight": inherit_inflight,
        "Phase1/3 time (ms)": round(
            phase["phase1_ms"] + phase["phase3_ms"], 6
        ),
        "Phase1 Poll Count": phase["phase1_poll_count"],
        "Phase3 Poll Count": phase["phase3_poll_count"],
        "GEMM DMA Bytes": dma_bytes,
        "GEMM DMA BW (GB/s)": round(gemm_dma_bw, 6),
        "GEMM Compute Share (%)": round(gemm_compute_share, 4),
        "GEMM-Effective (GFLOPS)": round(gemm_effective_gflops, 6),
        "next_output_prefetch_issue_count": prefetch[
            "next_output_prefetch_issue_count"
        ],
        "next_output_prefetch_hit_count": prefetch[
            "next_output_prefetch_hit_count"
        ],
        "next_output_rows_ready_at_boundary": prefetch[
            "next_output_rows_ready_at_boundary"
        ],
        "next_output_consumed_before_fallback_rows": prefetch[
            "next_output_consumed_before_fallback_rows"
        ],
        "prefetched_b_rows_consumed": prefetch["prefetched_b_rows_consumed"],
        "fallback_b_rows_fetched": prefetch["fallback_b_rows_fetched"],
        "next_output_prefetch_late_completion_count": prefetch[
            "next_output_prefetch_late_completion_count"
        ],
        "carry_over_rows_at_boundary": prefetch["carry_over_rows_at_boundary"],
        "carry_over_inflight_rows_at_boundary": prefetch[
            "carry_over_inflight_rows_at_boundary"
        ],
        "carry_over_rows_consumed_post_boundary": prefetch[
            "carry_over_rows_consumed_post_boundary"
        ],
        "normal_fetch_hole_rows": prefetch["normal_fetch_hole_rows"],
        "duplicate_b_row_fetch_detected": prefetch[
            "duplicate_b_row_fetch_detected"
        ],
        "carry_over_late_completion_count": prefetch[
            "carry_over_late_completion_count"
        ],
        "Bus Avg Request Payload Size (B)": round(
            (
                bus_dma["bus_dma_pkt_bytes"]
                / (bus_dma["bus_dma_pkt_count"] / 2.0)
            )
            if bus_dma["bus_dma_pkt_count"] > 0
            else 0.0,
            4,
        ),
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rs.ensure_libm5()

    print("🚀 Carry-over Sensitivity Analysis")
    print("   固定 baseline: Base@48, Large@128")
    print("   机制旋钮: carry_over_max_rows, carry_over_inherit_inflight")
    print("=" * 60)

    rows = []
    for preset in PRESETS:
        for carry_cap in preset["carry_caps"]:
            for inherit in INHERIT_INFLIGHT_OPTIONS:
                row = run_one(preset, carry_cap, inherit)
                rows.append(row)
                write_results(CSV_FILE, rows)
                print(
                    f"[√] {preset['name']} cap={carry_cap} inherit={inherit} "
                    f"share={row['GEMM Compute Share (%)']:.4f}% "
                    f"gemm={row['GEMM-Effective (GFLOPS)']:.3f} GFLOPS"
                )

    write_results(BEST_FILE, pick_best_rows(rows))
    print("\n" + "=" * 60)
    print(f"✅ 敏感性分析完成，结果写入: {CSV_FILE}")
    print(f"✅ 最优摘要写入: {BEST_FILE}")


if __name__ == "__main__":
    main()
