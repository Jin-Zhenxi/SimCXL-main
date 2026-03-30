#!/usr/bin/env python3
"""
Small-tile GEMM kernel search for MatrixFlow/CXL+HDM.

Goal:
- keep the mainline mechanism fixed to B-only + compute_launch
- search a richer rows_b space on representative small/square kernel sizes
- answer whether the local optimum is robust across tile shapes, not only ViT presets
"""

import csv
import os
import shutil
import subprocess
from collections import defaultdict

import run_sweep as rs

OUTPUT_DIR = "my_outputData"
CSV_FILE = os.path.join(OUTPUT_DIR, "small_tile_search.csv")
BEST_FILE = os.path.join(OUTPUT_DIR, "small_tile_search_best.csv")

# Search set = task-relevant ViT presets + representative square kernel shapes.
# This lets us answer both:
# 1) what is best for the current ViT/CXL mainline
# 2) whether the local optimum is robust beyond just ViT presets
SEARCH_PRESETS = [
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
        "name": "Kernel-S128",
        "seq_len": 128,
        "hidden_dim": 1024,
        "mlp_dim": 4096,
        "num_heads": 16,
    },
    {
        "name": "Kernel-S160",
        "seq_len": 160,
        "hidden_dim": 1024,
        "mlp_dim": 4096,
        "num_heads": 16,
    },
    {
        "name": "Kernel-S192",
        "seq_len": 192,
        "hidden_dim": 1024,
        "mlp_dim": 4096,
        "num_heads": 16,
    },
    {
        "name": "Kernel-S224",
        "seq_len": 224,
        "hidden_dim": 1024,
        "mlp_dim": 4096,
        "num_heads": 16,
    },
    {
        "name": "Kernel-S256",
        "seq_len": 256,
        "hidden_dim": 1024,
        "mlp_dim": 4096,
        "num_heads": 16,
    },
    {
        "name": "Kernel-S320",
        "seq_len": 320,
        "hidden_dim": 1024,
        "mlp_dim": 4096,
        "num_heads": 16,
    },
]

# Two-stage search:
# 1. coarse grid to find the broad basin
# 2. local refinement around the best coarse candidates
COARSE_ROWS_B_CANDIDATES = [8, 12, 16, 24, 32, 48, 64, 96, 128]
REFINE_DELTAS = (-16, -12, -8, -4, 4, 8, 12, 16)
MIN_ROWS_B = 4
MAX_ROWS_B = 128

FIXED_CONFIG = {
    "mode": "b_only",
    "trigger": "compute_launch",
    "rows_a": 0,
}

MAX_INVALID_RUN_RETRIES = 3


def write_results(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def pick_best_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["Preset"]].append(row)

    best_rows = []
    for preset, items in grouped.items():
        items.sort(
            key=lambda r: (
                float(r["GEMM Compute Share (%)"]),
                float(r["GEMM-Effective (GFLOPS)"]),
                -float(r["Phase1/3 time (ms)"]),
            ),
            reverse=True,
        )
        best_rows.append(items[0])

    robust = defaultdict(list)
    for row in rows:
        robust[int(row["Prefetch Rows B"])].append(
            float(row["GEMM Compute Share (%)"])
        )

    robust_rows = []
    for rows_b, shares in sorted(robust.items()):
        robust_rows.append(
            {
                "Preset": "ROBUST",
                "Prefetch Rows B": rows_b,
                "Mean GEMM Compute Share (%)": round(
                    sum(shares) / len(shares), 4
                ),
                "Num Presets": len(shares),
            }
        )

    robust_rows.sort(
        key=lambda r: r["Mean GEMM Compute Share (%)"], reverse=True
    )
    return best_rows, robust_rows


def score_row(row):
    """
    Search objective:
    - primary: GEMM Compute Share
    - secondary: GEMM-Effective
    - tertiary: lower Phase1/3 time
    - slight preference for lower DMA bytes when the top metrics are close
    """
    return (
        float(row["GEMM Compute Share (%)"]),
        float(row["GEMM-Effective (GFLOPS)"]),
        -float(row["Phase1/3 time (ms)"]),
        -float(row["GEMM DMA Bytes"]),
    )


def clamp_rows_b(value):
    return max(MIN_ROWS_B, min(MAX_ROWS_B, int(value)))


def build_refine_candidates(best_rows_b_values):
    candidates = set()
    for base in best_rows_b_values:
        for delta in REFINE_DELTAS:
            candidates.add(clamp_rows_b(base + delta))
    return sorted(candidates)


def run_one(preset, rows_b):
    cfg = {
        "label": f"smalltile_compute_launch_{rows_b}",
        "mode": FIXED_CONFIG["mode"],
        "trigger": FIXED_CONFIG["trigger"],
        "rows_a": FIXED_CONFIG["rows_a"],
        "rows_b": rows_b,
    }

    preset_tag = preset["name"].replace(" ", "_").replace("/", "_")
    run_tag = f"{preset_tag}__{cfg['label']}"

    rs.modify_c_file(preset)

    print(
        f"\n>>> Small-tile search: {preset['name']} "
        f"(S={preset['seq_len']}) rows_b={rows_b} <<<"
    )
    print("[*] 正在编译 trigger_gemm...")
    subprocess.run(rs.COMPILE_CMD, shell=True, check=True)

    inject_cmd = f"{rs.INJECT_CMD} --no-compile {preset['seq_len']}"
    print(f"[*] 正在注入镜像 (seq_len={preset['seq_len']})...")
    subprocess.run(inject_cmd, shell=True, check=True)

    env = os.environ.copy()
    env["MATRIXFLOW_NEXT_PREFETCH_MODE"] = cfg["mode"]
    env["MATRIXFLOW_NEXT_PREFETCH_TRIGGER"] = cfg["trigger"]
    env["MATRIXFLOW_NEXT_PREFETCH_ROWS_A"] = str(cfg["rows_a"])
    env["MATRIXFLOW_NEXT_PREFETCH_ROWS_B"] = str(cfg["rows_b"])

    metrics = None
    bus_dma = None
    prefetch = None
    serial_log = None
    log_file = None

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
        "Preset Class": "ViT"
        if preset["name"].startswith("ViT-")
        else "Kernel",
        "Preset": preset["name"],
        "SeqLen": preset["seq_len"],
        "HiddenDim": preset["hidden_dim"],
        "MLPDim": preset["mlp_dim"],
        "NumHeads": preset["num_heads"],
        "Prefetch Mode": cfg["mode"],
        "Prefetch Trigger": cfg["trigger"],
        "Prefetch Rows B": cfg["rows_b"],
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
        "next_output_prefetch_rows_issued": prefetch[
            "next_output_prefetch_rows_issued"
        ],
        "carry_over_rows_at_boundary": prefetch["carry_over_rows_at_boundary"],
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

    print("🚀 Small-Tile GEMM Kernel Search")
    print("   主线固定: B-only + compute_launch")
    print("   搜索维度: rows_b (两阶段: 粗扫 + 局部细化)")
    print("   搜索集: " + ", ".join(p["name"] for p in SEARCH_PRESETS))
    print(
        "   粗扫 rows_b: " + ", ".join(str(v) for v in COARSE_ROWS_B_CANDIDATES)
    )
    print("=" * 60)

    rows = []
    tried = set()

    # Stage 1: coarse search
    for preset in SEARCH_PRESETS:
        for rows_b in COARSE_ROWS_B_CANDIDATES:
            tried.add((preset["name"], rows_b))
            row = run_one(preset, rows_b)
            rows.append(row)
            write_results(CSV_FILE, rows)
            print(
                f"[√][coarse] {preset['name']} rows_b={rows_b} "
                f"share={row['GEMM Compute Share (%)']:.4f}% "
                f"gemm={row['GEMM-Effective (GFLOPS)']:.3f} GFLOPS"
            )

    # Stage 2: per-preset local refinement around the two best coarse points
    print("\n" + "=" * 60)
    print("🔎 进入局部细化阶段")
    coarse_by_preset = defaultdict(list)
    for row in rows:
        coarse_by_preset[row["Preset"]].append(row)

    for preset in SEARCH_PRESETS:
        coarse_items = sorted(
            coarse_by_preset[preset["name"]],
            key=score_row,
            reverse=True,
        )
        seed_rows_b = [
            int(item["Prefetch Rows B"]) for item in coarse_items[:2]
        ]
        refine_candidates = build_refine_candidates(seed_rows_b)
        print(
            f"[*] {preset['name']} 粗扫最佳 seed={seed_rows_b}, "
            f"细化 rows_b={refine_candidates}"
        )
        for rows_b in refine_candidates:
            key = (preset["name"], rows_b)
            if key in tried:
                continue
            tried.add(key)
            row = run_one(preset, rows_b)
            rows.append(row)
            write_results(CSV_FILE, rows)
            print(
                f"[√][refine] {preset['name']} rows_b={rows_b} "
                f"share={row['GEMM Compute Share (%)']:.4f}% "
                f"gemm={row['GEMM-Effective (GFLOPS)']:.3f} GFLOPS"
            )

    best_rows, robust_rows = pick_best_rows(rows)
    write_results(BEST_FILE, best_rows + robust_rows)

    print("\n" + "=" * 60)
    print(f"✅ 搜索完成，结果写入: {CSV_FILE}")
    print(f"✅ 最优摘要写入: {BEST_FILE}")


if __name__ == "__main__":
    main()
