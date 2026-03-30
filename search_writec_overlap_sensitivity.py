#!/usr/bin/env python3
"""
Minimal single-dimension sensitivity analysis for the third structural step.

Goal:
- Keep the second-knife baseline fixed
- Only perturb how many *new* B rows WriteC overlap may issue
- Use this once to show whether the third knife is still a main gain source

Fixed baseline:
- ViT-Large-like @ rows_B = 128
- carry_over_max_rows = 0
- carry_over_inherit_inflight = 0
- hole_fill_lead_rows = 32

Only perturbs:
- writec_overlap_b_issue_budget_rows
"""

import csv
import os
import shutil
import subprocess

import run_sweep as rs

OUTPUT_DIR = "my_outputData"
CSV_FILE = os.path.join(OUTPUT_DIR, "writec_overlap_sensitivity.csv")
BEST_FILE = os.path.join(OUTPUT_DIR, "writec_overlap_sensitivity_best.csv")

PRESET = {
    "name": "ViT-Large-like",
    "seq_len": 257,
    "hidden_dim": 1024,
    "mlp_dim": 4096,
    "num_heads": 16,
    "rows_b": 128,
}

WRITE_C_BUDGET_ROWS = [0, 1, 2, 4]
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


def load_existing_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def row_key(writec_budget):
    return str(writec_budget)


def score_row(row):
    return (
        float(row["GEMM Compute Share (%)"]),
        float(row["GEMM-Effective (GFLOPS)"]),
        -float(row["Phase1/3 time (ms)"]),
        -float(row["next_prefetch_late_completion_count"]),
        -float(row["next_output_prefetch_late_completion_count"]),
        -float(row["next_prefetch_discard_count"]),
    )


def pick_best_rows(rows):
    if not rows:
        return []
    items = sorted(rows, key=score_row, reverse=True)
    return [items[0]]


def run_one(writec_budget):
    run_tag = (
        f"{PRESET['name'].replace(' ', '_').replace('/', '_')}"
        f"__writec_budget_{writec_budget}"
    )

    rs.modify_c_file(PRESET)

    print(
        f"\n>>> WriteC overlap sensitivity: {PRESET['name']} "
        f"rows_b={PRESET['rows_b']} writec_budget={writec_budget} <<<"
    )
    print("[*] 正在编译 trigger_gemm...")
    subprocess.run(rs.COMPILE_CMD, shell=True, check=True)

    inject_cmd = f"{rs.INJECT_CMD} --no-compile {PRESET['seq_len']}"
    print(f"[*] 正在注入镜像 (seq_len={PRESET['seq_len']})...")
    subprocess.run(inject_cmd, shell=True, check=True)

    env = os.environ.copy()
    env["MATRIXFLOW_NEXT_PREFETCH_MODE"] = "b_only"
    env["MATRIXFLOW_NEXT_PREFETCH_TRIGGER"] = "compute_launch"
    env["MATRIXFLOW_NEXT_PREFETCH_ROWS_A"] = "0"
    env["MATRIXFLOW_NEXT_PREFETCH_ROWS_B"] = str(PRESET["rows_b"])
    env["MATRIXFLOW_CARRY_OVER_MAX_ROWS"] = "0"
    env["MATRIXFLOW_CARRY_OVER_INHERIT_INFLIGHT"] = "0"
    env["MATRIXFLOW_HOLE_FILL_LEAD_ROWS"] = "32"
    env["MATRIXFLOW_WRITEC_OVERLAP_B_ISSUE_BUDGET_ROWS"] = str(writec_budget)

    for attempt in range(1, MAX_INVALID_RUN_RETRIES + 1):
        m5out_dir = os.path.join(OUTPUT_DIR, f"m5out_{run_tag}")
        log_file = os.path.join(OUTPUT_DIR, f"terminal_log_{run_tag}.txt")
        if os.path.exists(m5out_dir):
            shutil.rmtree(m5out_dir)
        if os.path.exists(log_file):
            os.remove(log_file)

        cmd = rs.GEM5_CMD_TEMPLATE.format(outdir=m5out_dir, logfile=log_file)
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
        (4.0 * (PRESET["seq_len"] ** 3) / 1e9) / gemm_active_s
        if gemm_active_s > 0
        else 0.0
    )
    gemm_dma_bw = (
        (dma_bytes / 1e9) / gemm_active_s if gemm_active_s > 0 else 0.0
    )

    return {
        "Preset": PRESET["name"],
        "SeqLen": PRESET["seq_len"],
        "Prefetch Rows B": PRESET["rows_b"],
        "carry_over_max_rows": 0,
        "carry_over_inherit_inflight": 0,
        "hole_fill_lead_rows": 32,
        "writec_overlap_b_issue_budget_rows": writec_budget,
        "Phase1/3 time (ms)": round(
            phase["phase1_ms"] + phase["phase3_ms"], 6
        ),
        "Phase1 Poll Count": phase["phase1_poll_count"],
        "Phase3 Poll Count": phase["phase3_poll_count"],
        "GEMM DMA Bytes": dma_bytes,
        "GEMM DMA BW (GB/s)": round(gemm_dma_bw, 6),
        "GEMM Compute Share (%)": round(gemm_compute_share, 4),
        "GEMM-Effective (GFLOPS)": round(gemm_effective_gflops, 6),
        "next_prefetch_late_completion_count": prefetch[
            "next_prefetch_late_completion_count"
        ],
        "next_output_prefetch_late_completion_count": prefetch[
            "next_output_prefetch_late_completion_count"
        ],
        "next_prefetch_discard_count": prefetch["next_prefetch_discard_count"],
        "prefetched_b_rows_consumed": prefetch["prefetched_b_rows_consumed"],
        "fallback_b_rows_fetched": prefetch["fallback_b_rows_fetched"],
        "carry_over_rows_consumed_post_boundary": prefetch[
            "carry_over_rows_consumed_post_boundary"
        ],
        "normal_fetch_hole_rows": prefetch["normal_fetch_hole_rows"],
        "writec_overlap_enabled_count": prefetch[
            "writec_overlap_enabled_count"
        ],
        "writec_overlap_success_count": prefetch[
            "writec_overlap_success_count"
        ],
        "next_output_progress_during_writec": prefetch[
            "next_output_progress_during_writec"
        ],
        "b_rows_issued_during_writec": prefetch["b_rows_issued_during_writec"],
        "writec_blocked_b_issue_count": prefetch[
            "writec_blocked_b_issue_count"
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

    print("🚀 WriteC Overlap Sensitivity Analysis")
    print("   固定 baseline: Large@128")
    print("   固定 carry-over: max_rows=0, inherit_inflight=0")
    print("   固定 hole-filling: hole_fill_lead_rows=32")
    print("   单维度旋钮: writec_overlap_b_issue_budget_rows")
    print("=" * 60)

    rows = load_existing_rows(CSV_FILE)
    completed = {
        row_key(row["writec_overlap_b_issue_budget_rows"]) for row in rows
    }
    for budget in WRITE_C_BUDGET_ROWS:
        if row_key(budget) in completed:
            print(f"[-] 跳过已完成点: writec_budget={budget}")
            continue
        row = run_one(budget)
        rows.append(row)
        completed.add(row_key(budget))
        write_results(CSV_FILE, rows)
        print(
            f"[√] writec_budget={budget} "
            f"share={row['GEMM Compute Share (%)']:.4f}% "
            f"gemm={row['GEMM-Effective (GFLOPS)']:.3f} GFLOPS"
        )

    write_results(BEST_FILE, pick_best_rows(rows))
    print("\n" + "=" * 60)
    print(f"✅ 敏感性分析完成，结果写入: {CSV_FILE}")
    print(f"✅ 最优摘要写入: {BEST_FILE}")


if __name__ == "__main__":
    main()
