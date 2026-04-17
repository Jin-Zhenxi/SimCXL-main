#!/usr/bin/env python3
"""Run pure_gemm across the current run_sweep CXL architecture set."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path

from pure_gemm_cxl_run import (
    CLOCK_FREQ_GHZ,
    COMPILE_CMD,
    GEM5_BIN,
    GEM5_ROOT,
    GEM5_SCRIPT,
    OUTPUT_DIR,
    ensure_libm5,
    extract_phase_timings,
    extract_roi_metrics,
    patch_metrics_from_debug_log,
    run_cmd,
    run_gem5_with_progress,
    serial_log_has_pure_gemm,
)
from run_sweep import (
    DEFAULT_MIN_READ_REQUEST_BYTES,
    PREFETCH_CONFIGS,
    RUN_ONLY_LABELS,
    extract_gemm_active_time,
)

SUMMARY_CSV = OUTPUT_DIR / "pure_gemm_runsweep_cxl_summary.csv"
SUMMARY_TXT = OUTPUT_DIR / "pure_gemm_runsweep_cxl_summary.txt"
DEFAULT_ANCHOR_PRESET = "ViT-Large-like"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run pure_gemm over baseline + current run_sweep architecture labels."
        )
    )
    parser.add_argument(
        "--matrix-size",
        "--matrix_size",
        "--matrixflow-size",
        "--matrixflow_size",
        dest="matrix_size",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--device-link-gbs",
        "--device_link_gbs",
        dest="device_link_gbs",
        type=int,
        choices=[16, 32, 64, 128, 256],
        default=32,
    )
    parser.add_argument(
        "--cpu-type",
        "--cpu_type",
        dest="cpu_type",
        choices=["TIMING", "O3"],
        default="TIMING",
    )
    parser.add_argument(
        "--rows-b-anchor-preset",
        dest="rows_b_anchor_preset",
        default=DEFAULT_ANCHOR_PRESET,
        help=(
            "Which run_sweep preset anchor to use when a config defines "
            "rows_b_by_preset. Default matches the current active sweep preset."
        ),
    )
    parser.add_argument(
        "--labels",
        help=(
            "Comma-separated subset of labels to run. "
            "Default is baseline_serial plus current RUN_ONLY_LABELS."
        ),
    )
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--skip-inject", action="store_true")
    parser.add_argument("--keep-existing", action="store_true")
    return parser.parse_args()


def active_configs(requested_labels: str | None) -> list[dict]:
    if requested_labels:
        requested = {
            label.strip()
            for label in requested_labels.split(",")
            if label.strip()
        }
    else:
        requested = {"baseline_serial", *RUN_ONLY_LABELS}

    return [cfg for cfg in PREFETCH_CONFIGS if cfg["label"] in requested]


def rows_b_for_cfg(cfg: dict, anchor_preset: str) -> int:
    return cfg.get("rows_b_by_preset", {}).get(anchor_preset, cfg["rows_b"])


def build_env(cfg: dict, rows_b: int) -> dict[str, str]:
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
    env["MATRIXFLOW_VIP_B_ROWS_CAPACITY"] = str(
        cfg.get("vip_b_rows_capacity", 0)
    )
    env["MATRIXFLOW_MHOT_B_ROWS_CAPACITY"] = str(
        cfg.get("mhot_b_rows_capacity", 0)
    )
    env["MATRIXFLOW_COVERAGE_SHADOW_ROWS_CAPACITY"] = str(
        cfg.get("coverage_shadow_rows_capacity", 0)
    )
    env["MATRIXFLOW_COVERAGE_GATHER_MIN_ISSUE_BUDGET"] = str(
        cfg.get("coverage_gather_min_issue_budget", 0)
    )
    env["MATRIXFLOW_AB_SCHEDULER_MODE"] = cfg.get(
        "ab_scheduler_mode", "baseline"
    )
    env["MATRIXFLOW_AB_A_MIN_CREDIT_ROWS"] = str(
        cfg.get("ab_a_min_credit_rows", 16)
    )
    env["MATRIXFLOW_AB_B_BIAS"] = str(cfg.get("ab_bias_b", 1))
    env["MATRIXFLOW_AB_W_URGENCY"] = str(cfg.get("ab_weight_urgency", 4))
    env["MATRIXFLOW_AB_W_DEFICIT"] = str(cfg.get("ab_weight_deficit", 3))
    env["MATRIXFLOW_AB_W_REUSE"] = str(cfg.get("ab_weight_reuse", 1))
    env["MATRIXFLOW_AB_W_FALLBACK_RISK"] = str(
        cfg.get("ab_weight_fallback_risk", 2)
    )
    env["MATRIXFLOW_AB_MIN_LAUNCH_ROWS_A"] = str(
        cfg.get("ab_min_launch_rows_a", 0)
    )
    env["MATRIXFLOW_AB_MIN_LAUNCH_ROWS_B"] = str(
        cfg.get("ab_min_launch_rows_b", 0)
    )
    env["MATRIXFLOW_AB_CURRENT_PROTECTED_B_QUOTA_ROWS"] = str(
        cfg.get("ab_current_protected_b_quota_rows", 4)
    )
    env["MATRIXFLOW_MIN_READ_REQUEST_BYTES"] = str(
        DEFAULT_MIN_READ_REQUEST_BYTES
    )
    return env


def summarize_case(
    label: str,
    rows_b: int,
    matrix_size: int,
    device_link_gbs: int,
    run_status: str,
    stats_file: Path,
    terminal_log: Path,
    serial_log: Path,
    m5out_dir: Path,
) -> dict:
    metrics = extract_roi_metrics(stats_file)
    metrics = patch_metrics_from_debug_log(metrics, terminal_log)
    phase = extract_phase_timings(serial_log)
    gemm_active_s = extract_gemm_active_time(str(terminal_log))
    gemm_wall_s = (
        gemm_active_s
        if gemm_active_s > 0
        else (phase["phase1_ms"] / 1.0e3 if phase["phase1_ms"] > 0 else 0.0)
    )
    flops = 2.0 * (matrix_size**3)
    compute_time_total_s = (
        metrics["computeCycles"] / (CLOCK_FREQ_GHZ * 1e9)
        if metrics["computeCycles"] > 0
        else 0.0
    )
    workload_ok = (
        serial_log_has_pure_gemm(serial_log)
        and metrics["dmaRead"] > 0
        and metrics["computeCycles"] > 0
    )
    gemm_effective = (flops / 1e9) / gemm_wall_s if gemm_wall_s > 0 else 0.0
    compute_share = (
        100.0 * compute_time_total_s / gemm_wall_s if gemm_wall_s > 0 else 0.0
    )

    if run_status == "ok" and workload_ok and gemm_active_s <= 0.0:
        run_status = "invalid_missing_active_time"
        workload_ok = False
    if run_status == "ok" and workload_ok and compute_share > 105.0:
        run_status = "invalid_inconsistent_timing"
        workload_ok = False
    if run_status == "ok" and not workload_ok:
        run_status = "invalid_false_completion"

    return {
        "Label": label,
        "Workload": "pure_gemm",
        "Preset": f"pure_gemm_{matrix_size}",
        "Run Tag": f"pure_gemm_{matrix_size}_cxl_{device_link_gbs}g__{label}",
        "Device Link (GB/s)": device_link_gbs,
        "Matrix Size": matrix_size,
        "RowsB Anchor Preset": args.rows_b_anchor_preset,
        "RowsB": rows_b,
        "Run Status": run_status,
        "Workload OK": workload_ok,
        "phase1_ms": round(phase["phase1_ms"], 6),
        "phase2_total_ms": round(phase["phase2_total_ms"], 6),
        "end_to_end_ms": round(phase["end_to_end_ms"], 6),
        "GEMM Active Time (s)": round(gemm_active_s, 9),
        "GEMM-Effective (GFLOPS)": round(
            gemm_effective if workload_ok else 0.0, 6
        ),
        "ComputeShare (%)": round(compute_share if workload_ok else 0.0, 6),
        "dmaRead": int(metrics["dmaRead"]),
        "dmaWrite": int(metrics["dmaWrite"]),
        "computeCycles": int(metrics["computeCycles"]),
        "m5out_dir": str(m5out_dir),
        "terminal_log": str(terminal_log),
        "serial_log": str(serial_log),
    }


def write_summary(rows: list[dict]) -> None:
    if not rows:
        SUMMARY_TXT.write_text("no rows\n", encoding="utf-8")
        return

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = []
    for row in rows:
        lines.append(
            f"{row['Label']}: phase1_ms={row['phase1_ms']}, "
            f"GEMM-Effective={row['GEMM-Effective (GFLOPS)']} GFLOPS, "
            f"ComputeShare={row['ComputeShare (%)']}%, "
            f"status={row['Run Status']}"
        )
    SUMMARY_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    global args
    args = parse_args()
    ensure_libm5()
    OUTPUT_DIR.mkdir(exist_ok=True)

    cfgs = active_configs(args.labels)
    if not cfgs:
        raise SystemExit("没有匹配到任何配置标签。")

    print(
        "[*] 本轮 pure_gemm 2048/自定义矩阵将复用 run_sweep 架构集: "
        + ", ".join(cfg["label"] for cfg in cfgs)
    )
    print(
        f"[*] rows_b anchor preset = {args.rows_b_anchor_preset}, "
        f"device_link_gbs = {args.device_link_gbs}, matrix_size = {args.matrix_size}"
    )

    if not args.skip_compile:
        run_cmd(COMPILE_CMD)

    if not args.skip_inject:
        run_cmd(
            f"sudo ./inject_trigger_gemm.sh --no-compile {args.matrix_size}"
        )

    rows = []
    for cfg in cfgs:
        label = cfg["label"]
        rows_b = rows_b_for_cfg(cfg, args.rows_b_anchor_preset)
        env = build_env(cfg, rows_b)
        run_tag = f"pure_gemm_{args.matrix_size}_cxl_{args.device_link_gbs}g__{label}"
        m5out_dir = OUTPUT_DIR / f"m5out_{run_tag}"
        terminal_log = OUTPUT_DIR / f"terminal_log_{run_tag}.txt"
        serial_log = m5out_dir / "board.pc.com_1.device"
        stats_file = m5out_dir / "stats.txt"

        if not args.keep_existing:
            if m5out_dir.exists():
                shutil.rmtree(m5out_dir)
            if terminal_log.exists():
                terminal_log.unlink()

        run_status = "ok"
        reuse_existing = (
            args.keep_existing
            and terminal_log.exists()
            and serial_log.exists()
            and stats_file.exists()
        )
        if reuse_existing:
            print(f"[*] 复用已有结果: {run_tag}")
        else:
            cmd = (
                f"{GEM5_BIN} -p {GEM5_ROOT / 'src/python'} "
                "--debug-flags=MatrixFlowTiming "
                f"-d {m5out_dir} "
                f"{GEM5_SCRIPT} "
                "--is_asic True "
                f"--cpu_type {args.cpu_type} "
                "--no-network "
                f"--matrixflow_size {args.matrix_size} "
                "--matrixflow_workload pure_gemm "
                f"--device-link-gbs {args.device_link_gbs} "
                "--allow-local-trigger "
                f"> {terminal_log} 2>&1"
            )
            rc = run_gem5_with_progress(cmd, serial_log, terminal_log, env=env)
            if rc != 0:
                run_status = f"failed({rc})"

        row = summarize_case(
            label=label,
            rows_b=rows_b,
            matrix_size=args.matrix_size,
            device_link_gbs=args.device_link_gbs,
            run_status=run_status,
            stats_file=stats_file,
            terminal_log=terminal_log,
            serial_log=serial_log,
            m5out_dir=m5out_dir,
        )
        rows.append(row)
        print(
            f"[{'√' if row['Workload OK'] else '!'}] {label}: "
            f"phase1_ms={row['phase1_ms']}, "
            f"GEMM-Effective={row['GEMM-Effective (GFLOPS)']} GFLOPS, "
            f"ComputeShare={row['ComputeShare (%)']}%, "
            f"status={row['Run Status']}"
        )

    write_summary(rows)
    print(f"[done] 汇总已写入 {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
