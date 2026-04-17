#!/usr/bin/env python3
"""Oracle-style autopsy for irregular pure_gemm sizes around 257."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import shutil
from dataclasses import dataclass
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
    patch_metrics_from_debug_log,
    run_cmd,
    run_gem5_with_progress,
    serial_log_has_pure_gemm,
)
from run_sweep import (
    DEFAULT_MIN_READ_REQUEST_BYTES,
    extract_gemm_active_time,
)

DEFAULT_SIZES = [240, 248, 255, 256, 257, 258, 272]
SUMMARY_RAW_CSV = OUTPUT_DIR / "pure_gemm257_oracle_autopsy_raw.csv"
SUMMARY_ANALYSIS_CSV = OUTPUT_DIR / "pure_gemm257_oracle_autopsy.csv"
SUMMARY_REPORT_TXT = OUTPUT_DIR / "pure_gemm257_oracle_autopsy_report.txt"
CHUNK_SEPARATOR = "---------- Begin Simulation Statistics ----------"
TILE_DIM = 128
ARRAY_DIM = 16
ELEM_BYTES = 4


@dataclass(frozen=True)
class OracleCase:
    label: str
    mode: str
    trigger: str
    rows_a: int
    rows_b: int
    carry_over_max_rows: int
    carry_over_inherit_inflight: int
    hole_fill_lead_rows: int
    writec_overlap_b_issue_budget_rows: int
    vip_b_rows_capacity: int
    mhot_b_rows_capacity: int
    coverage_shadow_rows_capacity: int
    coverage_gather_min_issue_budget: int
    ab_scheduler_mode: str
    ab_a_min_credit_rows: int = 16
    ab_bias_b: int = 1
    ab_weight_urgency: int = 4
    ab_weight_deficit: int = 3
    ab_weight_reuse: int = 1
    ab_weight_fallback_risk: int = 2
    ab_min_launch_rows_a: int = 0
    ab_min_launch_rows_b: int = 0
    ab_current_protected_b_quota_rows: int = 4
    note: str = ""


ORACLE_CASES = [
    OracleCase(
        label="irregular_gemm_oracle_clean_baseline",
        mode="b_only",
        trigger="b_ready",
        rows_a=0,
        rows_b=32,
        carry_over_max_rows=0,
        carry_over_inherit_inflight=0,
        hole_fill_lead_rows=0,
        writec_overlap_b_issue_budget_rows=0,
        vip_b_rows_capacity=0,
        mhot_b_rows_capacity=0,
        coverage_shadow_rows_capacity=0,
        coverage_gather_min_issue_budget=0,
        ab_scheduler_mode="baseline",
        note=(
            "Conservative clean baseline for irregular GEMM: no VIP/M-hot/"
            "coverage, smaller rows_b, lower speculative depth."
        ),
    ),
    OracleCase(
        label="baseline_serial",
        mode="b_only",
        trigger="compute_launch",
        rows_a=0,
        rows_b=128,
        carry_over_max_rows=0,
        carry_over_inherit_inflight=0,
        hole_fill_lead_rows=32,
        writec_overlap_b_issue_budget_rows=0,
        vip_b_rows_capacity=0,
        mhot_b_rows_capacity=0,
        coverage_shadow_rows_capacity=0,
        coverage_gather_min_issue_budget=0,
        ab_scheduler_mode="baseline",
        note="Current simple mainline baseline from run_sweep.",
    ),
    OracleCase(
        label="irregular_gemm_oracle_vip_only",
        mode="b_only",
        trigger="b_ready",
        rows_a=0,
        rows_b=32,
        carry_over_max_rows=0,
        carry_over_inherit_inflight=0,
        hole_fill_lead_rows=0,
        writec_overlap_b_issue_budget_rows=0,
        vip_b_rows_capacity=64,
        mhot_b_rows_capacity=0,
        coverage_shadow_rows_capacity=0,
        coverage_gather_min_issue_budget=0,
        ab_scheduler_mode="b_vip_rescue_recurrence_aware_v3",
        note="Clean baseline plus VIP rescue only.",
    ),
    OracleCase(
        label="irregular_gemm_oracle_mhot_only",
        mode="b_only",
        trigger="b_ready",
        rows_a=0,
        rows_b=32,
        carry_over_max_rows=0,
        carry_over_inherit_inflight=0,
        hole_fill_lead_rows=0,
        writec_overlap_b_issue_budget_rows=0,
        vip_b_rows_capacity=64,
        mhot_b_rows_capacity=128,
        coverage_shadow_rows_capacity=0,
        coverage_gather_min_issue_budget=0,
        ab_scheduler_mode="b_mhot_gap_aware_next_cut",
        note="Clean baseline plus VIP + M-hot.",
    ),
    OracleCase(
        label="irregular_gemm_oracle_gather_only",
        mode="b_only",
        trigger="b_ready",
        rows_a=0,
        rows_b=32,
        carry_over_max_rows=0,
        carry_over_inherit_inflight=0,
        hole_fill_lead_rows=0,
        writec_overlap_b_issue_budget_rows=0,
        vip_b_rows_capacity=64,
        mhot_b_rows_capacity=128,
        coverage_shadow_rows_capacity=128,
        coverage_gather_min_issue_budget=0,
        ab_scheduler_mode="b_coverage_2d_gather_pingpong_first_cut",
        note="Clean baseline plus VIP + M-hot + gather/shadow.",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and analyze irregular pure_gemm oracle cases."
    )
    parser.add_argument(
        "--sizes",
        default=",".join(str(v) for v in DEFAULT_SIZES),
        help="Comma-separated matrix sizes to sweep.",
    )
    parser.add_argument(
        "--cases",
        help=(
            "Comma-separated case labels. Default runs the full oracle set: "
            + ", ".join(case.label for case in ORACLE_CASES)
        ),
    )
    parser.add_argument(
        "--device-link-gbs",
        dest="device_link_gbs",
        type=int,
        choices=[16, 32, 64, 128, 256],
        default=32,
    )
    parser.add_argument(
        "--cpu-type",
        dest="cpu_type",
        choices=["TIMING", "O3"],
        default="TIMING",
    )
    parser.add_argument(
        "--pure-gemm-pack-tail-align16",
        "--pure_gemm_pack_tail_align16",
        dest="pure_gemm_pack_tail_align16",
        action="store_true",
    )
    parser.add_argument(
        "--pure-gemm-peeled-rect-v1",
        "--pure_gemm_peeled_rect_v1",
        dest="pure_gemm_peeled_rect_v1",
        action="store_true",
    )
    parser.add_argument(
        "--pure-gemm-peeled-rect-v2-right-edge-rectified",
        "--pure_gemm_peeled_rect_v2_right_edge_rectified",
        dest="pure_gemm_peeled_rect_v2_right_edge_rectified",
        action="store_true",
    )
    parser.add_argument(
        "--pure-gemm-peeled-rect-batched-single-doorbell-first-cut",
        "--pure_gemm_peeled_rect_batched_single_doorbell_first_cut",
        dest="pure_gemm_peeled_rect_batched_single_doorbell_first_cut",
        action="store_true",
    )
    parser.add_argument(
        "--irregular-gemm-b-tail-scratchpad-output-hold-first-cut",
        "--irregular_gemm_b_tail_scratchpad_output_hold_first_cut",
        dest="irregular_gemm_b_tail_scratchpad_output_hold_first_cut",
        action="store_true",
    )
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--skip-inject", action="store_true")
    parser.add_argument("--keep-existing", action="store_true")
    return parser.parse_args()


def parse_sizes(raw: str) -> list[int]:
    sizes = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            sizes.append(int(token))
    if not sizes:
        raise SystemExit("至少需要一个 matrix size。")
    return sizes


def selected_cases(raw: str | None) -> list[OracleCase]:
    if not raw:
        return ORACLE_CASES
    requested = {token.strip() for token in raw.split(",") if token.strip()}
    cases = [case for case in ORACLE_CASES if case.label in requested]
    if not cases:
        raise SystemExit("没有匹配到任何 oracle case。")
    return cases


def build_env(case: OracleCase) -> dict[str, str]:
    env = os.environ.copy()
    env["MATRIXFLOW_NEXT_PREFETCH_MODE"] = case.mode
    env["MATRIXFLOW_NEXT_PREFETCH_TRIGGER"] = case.trigger
    env["MATRIXFLOW_NEXT_PREFETCH_ROWS_A"] = str(case.rows_a)
    env["MATRIXFLOW_NEXT_PREFETCH_ROWS_B"] = str(case.rows_b)
    env["MATRIXFLOW_CARRY_OVER_MAX_ROWS"] = str(case.carry_over_max_rows)
    env["MATRIXFLOW_CARRY_OVER_INHERIT_INFLIGHT"] = str(
        case.carry_over_inherit_inflight
    )
    env["MATRIXFLOW_HOLE_FILL_LEAD_ROWS"] = str(case.hole_fill_lead_rows)
    env["MATRIXFLOW_WRITEC_OVERLAP_B_ISSUE_BUDGET_ROWS"] = str(
        case.writec_overlap_b_issue_budget_rows
    )
    env["MATRIXFLOW_VIP_B_ROWS_CAPACITY"] = str(case.vip_b_rows_capacity)
    env["MATRIXFLOW_MHOT_B_ROWS_CAPACITY"] = str(case.mhot_b_rows_capacity)
    env["MATRIXFLOW_COVERAGE_SHADOW_ROWS_CAPACITY"] = str(
        case.coverage_shadow_rows_capacity
    )
    env["MATRIXFLOW_COVERAGE_GATHER_MIN_ISSUE_BUDGET"] = str(
        case.coverage_gather_min_issue_budget
    )
    env["MATRIXFLOW_AB_SCHEDULER_MODE"] = case.ab_scheduler_mode
    env["MATRIXFLOW_AB_A_MIN_CREDIT_ROWS"] = str(case.ab_a_min_credit_rows)
    env["MATRIXFLOW_AB_B_BIAS"] = str(case.ab_bias_b)
    env["MATRIXFLOW_AB_W_URGENCY"] = str(case.ab_weight_urgency)
    env["MATRIXFLOW_AB_W_DEFICIT"] = str(case.ab_weight_deficit)
    env["MATRIXFLOW_AB_W_REUSE"] = str(case.ab_weight_reuse)
    env["MATRIXFLOW_AB_W_FALLBACK_RISK"] = str(case.ab_weight_fallback_risk)
    env["MATRIXFLOW_AB_MIN_LAUNCH_ROWS_A"] = str(case.ab_min_launch_rows_a)
    env["MATRIXFLOW_AB_MIN_LAUNCH_ROWS_B"] = str(case.ab_min_launch_rows_b)
    env["MATRIXFLOW_AB_CURRENT_PROTECTED_B_QUOTA_ROWS"] = str(
        case.ab_current_protected_b_quota_rows
    )
    env["MATRIXFLOW_MIN_READ_REQUEST_BYTES"] = str(
        DEFAULT_MIN_READ_REQUEST_BYTES
    )
    return env


def stats_active_chunk(stats_file: Path) -> str:
    if not stats_file.exists():
        return ""
    text = stats_file.read_text(encoding="utf-8", errors="ignore")
    chunks = text.split(CHUNK_SEPARATOR)
    for chunk in chunks:
        match = re.search(r"totalComputeCycles\s+(\d+)", chunk)
        if match and int(match.group(1)) > 0:
            return chunk
    return chunks[-1] if chunks else ""


def extract_scalar(chunk: str, pattern: str) -> float:
    match = re.search(pattern, chunk, re.MULTILINE)
    return float(match.group(1)) if match else 0.0


def sum_all(chunk: str, pattern: str) -> float:
    return float(
        sum(int(value) for value in re.findall(pattern, chunk, re.MULTILINE))
    )


def extract_engine_stats(stats_file: Path) -> dict[str, float]:
    chunk = stats_active_chunk(stats_file)
    stats = {
        "dmaRead": extract_scalar(chunk, r"totalDmaBytesRead\s+(\d+)"),
        "dmaWrite": extract_scalar(chunk, r"totalDmaBytesWritten\s+(\d+)"),
        "computeCycles": extract_scalar(chunk, r"totalComputeCycles\s+(\d+)"),
        "fallback_b_rows_fetched": extract_scalar(
            chunk, r"fallbackBRowsFetched\s+(\d+)"
        ),
        "normal_fetch_hole_rows": extract_scalar(
            chunk, r"normalFetchHoleRows\s+(\d+)"
        ),
        "vip_rescue_hit_count": extract_scalar(
            chunk, r"vipRescueHitCount\s+(\d+)"
        ),
        "mhot_hit_count": extract_scalar(chunk, r"mhotHitCount\s+(\d+)"),
        "coverage_gather_issue_count": extract_scalar(
            chunk, r"coverageGatherIssueCount\s+(\d+)"
        ),
        "coverage_shadow_hit_count": extract_scalar(
            chunk, r"coverageShadowHitCount\s+(\d+)"
        ),
        "a_path_stall_waiting_for_b": extract_scalar(
            chunk, r"aPathStallWaitingForB\s+(\d+)"
        ),
        "b_path_stall_waiting_for_a": extract_scalar(
            chunk, r"bPathStallWaitingForA\s+(\d+)"
        ),
        "rx_a_stall_cycles": extract_scalar(chunk, r"rxAStallCycles\s+(\d+)"),
        "rx_b_stall_cycles": extract_scalar(chunk, r"rxBStallCycles\s+(\d+)"),
        "writec_overlap_cycles": extract_scalar(
            chunk, r"writeCOverlapCycles\s+(\d+)"
        ),
        "dram_num_reads": sum_all(
            chunk,
            r"dram\.numReads::pc\.south_bridge\.cxl_device\.matrix_engine\s+(\d+)",
        ),
        "dram_num_writes": sum_all(
            chunk,
            r"dram\.numWrites::pc\.south_bridge\.cxl_device\.matrix_engine\s+(\d+)",
        ),
        "bridge_req_send_succeed": extract_scalar(
            chunk, r"board\.cxl_bridge\.reqSendSucceed\s+(\d+)"
        ),
        "engine_metrics_source": "stats",
    }
    return stats


def patch_engine_stats_from_terminal_log(
    stats: dict[str, float], terminal_log: Path
) -> dict[str, float]:
    if not terminal_log.exists():
        return stats
    if (
        stats["dmaRead"] > 0
        and stats["dmaWrite"] > 0
        and stats["computeCycles"] > 0
    ):
        return stats

    last_match: re.Match[str] | None = None
    pattern = re.compile(
        r"MatrixFlow completion flag written:.*"
        r"dmaRead=(\d+)\s+dmaWrite=(\d+)\s+computeCycles=(\d+)"
    )
    with terminal_log.open(encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            match = pattern.search(line)
            if match:
                last_match = match

    if last_match:
        stats["dmaRead"] = float(last_match.group(1))
        stats["dmaWrite"] = float(last_match.group(2))
        stats["computeCycles"] = float(last_match.group(3))
        stats["engine_metrics_source"] = "terminal_fallback"
    return stats


def extract_raw_timings(serial_log: Path) -> dict[str, float]:
    raw = {
        "phase1_begin_ns": 0.0,
        "phase1_end_ns": 0.0,
        "phase2_begin_ns": 0.0,
        "phase2_end_ns": 0.0,
        "phase3_begin_ns": 0.0,
        "phase3_end_ns": 0.0,
        "total_end_ns": 0.0,
    }
    if not serial_log.exists():
        return raw
    patterns = {
        key: re.compile(rf"\[TimingRaw\] {re.escape(key)}=(\d+)")
        for key in raw
    }
    with serial_log.open(encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            for key, pattern in patterns.items():
                match = pattern.search(line)
                if match:
                    raw[key] = float(match.group(1))
    return raw


def extract_start_done_ticks(terminal_log: Path) -> dict[str, float]:
    result = {
        "start_tick_ps": 0.0,
        "done_tick_ps": 0.0,
        "completion_count": 0.0,
    }
    if not terminal_log.exists():
        return result
    start_pat = re.compile(r"^(\d+): .*startMatrixCompute:")
    done_pat = re.compile(r"^(\d+): .*MatrixFlow completion flag written:")
    with terminal_log.open(encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            start_match = start_pat.search(line)
            if start_match and result["start_tick_ps"] == 0.0:
                result["start_tick_ps"] = float(start_match.group(1))
            done_match = done_pat.search(line)
            if done_match:
                result["completion_count"] += 1.0
                if result["done_tick_ps"] == 0.0:
                    result["done_tick_ps"] = float(done_match.group(1))
    return result


def align_up(value: int, granularity: int) -> int:
    return ((value + granularity - 1) // granularity) * granularity


def tile_sizes(n: int, tile: int) -> list[int]:
    dim = math.ceil(n / tile)
    sizes = []
    remaining = n
    for _ in range(dim):
        this = min(tile, remaining)
        sizes.append(this)
        remaining -= this
    return sizes


def derive_geometry_metrics(
    matrix_size: int,
    actual_dma_bytes_total: float,
    compute_cycles: float,
    gemm_active_s: float,
    phase1_begin_ns: float,
    phase1_end_ns: float,
    start_tick_ps: float,
    done_tick_ps: float,
    min_request_bytes: int = DEFAULT_MIN_READ_REQUEST_BYTES,
) -> dict[str, float]:
    dims = tile_sizes(matrix_size, TILE_DIM)
    total_tiles = len(dims) ** 3
    tail_tiles = 0
    useful_macs = 0
    padded_macs = 0
    tail_useful_macs = 0
    tail_padded_macs = 0
    useful_read_bytes = 0
    useful_write_bytes = 0
    row_rounded_bytes = 0
    small_request_count = 0
    row_request_count_oracle = 0

    for mi in dims:
        for nj in dims:
            useful_write_bytes += mi * nj * ELEM_BYTES
            write_row_bytes = nj * ELEM_BYTES
            row_rounded_bytes += mi * align_up(
                write_row_bytes, min_request_bytes
            )
            row_request_count_oracle += mi
            if write_row_bytes < min_request_bytes:
                small_request_count += mi
            for kk in dims:
                is_tail = (mi < TILE_DIM) or (nj < TILE_DIM) or (kk < TILE_DIM)
                if is_tail:
                    tail_tiles += 1

                useful = mi * nj * kk
                padded = (
                    align_up(mi, ARRAY_DIM)
                    * align_up(nj, ARRAY_DIM)
                    * align_up(kk, ARRAY_DIM)
                )
                useful_macs += useful
                padded_macs += padded
                if is_tail:
                    tail_useful_macs += useful
                    tail_padded_macs += padded

                useful_read_bytes += (mi * kk + kk * nj) * ELEM_BYTES

                a_row_bytes = kk * ELEM_BYTES
                b_row_bytes = nj * ELEM_BYTES
                row_rounded_bytes += mi * align_up(
                    a_row_bytes, min_request_bytes
                )
                row_rounded_bytes += kk * align_up(
                    b_row_bytes, min_request_bytes
                )
                row_request_count_oracle += mi + kk
                if a_row_bytes < min_request_bytes:
                    small_request_count += mi
                if b_row_bytes < min_request_bytes:
                    small_request_count += kk

    useful_dma_bytes = useful_read_bytes + useful_write_bytes
    tail_dma_overfetch_bytes = max(row_rounded_bytes - useful_dma_bytes, 0)
    dma_overfetch_bytes = max(actual_dma_bytes_total - useful_dma_bytes, 0)
    tail_tile_ratio = tail_tiles / total_tiles if total_tiles else 0.0
    useful_mac_ratio = useful_macs / padded_macs if padded_macs else 0.0
    padded_mac_ratio = 1.0 - useful_mac_ratio if padded_macs else 0.0
    tail_tile_utilization_avg = (
        tail_useful_macs / tail_padded_macs if tail_padded_macs else 1.0
    )
    gemm_active_cycles = gemm_active_s * CLOCK_FREQ_GHZ * 1e9
    compute_stall_cycles = max(gemm_active_cycles - compute_cycles, 0.0)
    compute_share = (
        compute_cycles / gemm_active_cycles if gemm_active_cycles > 0 else 0.0
    )
    pe_active_ratio = compute_share * useful_mac_ratio

    launch_overhead_cycles = 0.0
    completion_overhead_cycles = 0.0
    if phase1_begin_ns > 0 and start_tick_ps > 0:
        launch_overhead_cycles = max(
            ((start_tick_ps / 1000.0) - phase1_begin_ns) * CLOCK_FREQ_GHZ, 0.0
        )
    if phase1_end_ns > 0 and done_tick_ps > 0:
        completion_overhead_cycles = max(
            (phase1_end_ns - (done_tick_ps / 1000.0)) * CLOCK_FREQ_GHZ, 0.0
        )

    return {
        "tail_tile_count": float(tail_tiles),
        "tail_tile_ratio": tail_tile_ratio,
        "edge_tile_count": float(tail_tiles),
        "useful_mac_ratio": useful_mac_ratio,
        "padded_mac_ratio": padded_mac_ratio,
        "tail_compute_waste": float(max(padded_macs - useful_macs, 0)),
        "dma_useful_bytes": float(useful_dma_bytes),
        "tail_dma_overfetch_bytes": float(tail_dma_overfetch_bytes),
        "dma_overfetch_bytes": float(dma_overfetch_bytes),
        "row_request_count_oracle": float(row_request_count_oracle),
        "small_request_count": float(small_request_count),
        "pe_active_ratio": pe_active_ratio,
        "tile_utilization_avg": useful_mac_ratio,
        "tail_tile_utilization_avg": tail_tile_utilization_avg,
        "compute_stall_cycles": compute_stall_cycles,
        "effective_flops_per_launched_tile": (
            (2.0 * useful_macs) / total_tiles if total_tiles else 0.0
        ),
        "launch_overhead_cycles": launch_overhead_cycles,
        "completion_overhead_cycles": completion_overhead_cycles,
    }


def run_one_case(
    case: OracleCase,
    matrix_size: int,
    device_link_gbs: int,
    cpu_type: str,
    keep_existing: bool,
    pure_gemm_pack_tail_align16: bool,
    pure_gemm_peeled_rect_v1: bool,
    pure_gemm_peeled_rect_v2_right_edge_rectified: bool,
    pure_gemm_peeled_rect_batched_single_doorbell_first_cut: bool,
    irregular_gemm_b_tail_scratchpad_output_hold_first_cut: bool,
) -> tuple[str, Path, Path, Path]:
    run_tag = f"{case.label}__pure_gemm_{matrix_size}_cxl_{device_link_gbs}g"
    if irregular_gemm_b_tail_scratchpad_output_hold_first_cut:
        run_tag += "__tail01"
    elif pure_gemm_peeled_rect_batched_single_doorbell_first_cut:
        run_tag += "__peeled_batch1"
    elif pure_gemm_peeled_rect_v2_right_edge_rectified:
        run_tag += "__peeledv2_rightedge"
    elif pure_gemm_peeled_rect_v1:
        run_tag += "__peeledv1"
    elif pure_gemm_pack_tail_align16:
        run_tag += "__pack16"
    m5out_dir = OUTPUT_DIR / f"m5out_{run_tag}"
    terminal_log = OUTPUT_DIR / f"terminal_log_{run_tag}.txt"
    serial_log = m5out_dir / "board.pc.com_1.device"
    stats_file = m5out_dir / "stats.txt"

    if not keep_existing:
        if m5out_dir.exists():
            shutil.rmtree(m5out_dir)
        if terminal_log.exists():
            terminal_log.unlink()

    run_status = "ok"
    reuse_existing = (
        keep_existing
        and terminal_log.exists()
        and serial_log.exists()
        and stats_file.exists()
    )
    if not reuse_existing:
        env = build_env(case)
        cmd = (
            f"{GEM5_BIN} -p {GEM5_ROOT / 'src/python'} "
            "--debug-flags=MatrixFlowTiming "
            f"-d {m5out_dir} "
            f"{GEM5_SCRIPT} "
            "--is_asic True "
            f"--cpu_type {cpu_type} "
            "--no-network "
            f"--matrixflow_size {matrix_size} "
            "--matrixflow_workload pure_gemm "
            f"--device-link-gbs {device_link_gbs} "
            f"{'--irregular-gemm-b-tail-scratchpad-output-hold-first-cut ' if irregular_gemm_b_tail_scratchpad_output_hold_first_cut else ''}"
            f"{'--pure-gemm-peeled-rect-batched-single-doorbell-first-cut ' if (pure_gemm_peeled_rect_batched_single_doorbell_first_cut and not irregular_gemm_b_tail_scratchpad_output_hold_first_cut) else ''}"
            f"{'--pure-gemm-peeled-rect-v2-right-edge-rectified ' if (pure_gemm_peeled_rect_v2_right_edge_rectified and not pure_gemm_peeled_rect_batched_single_doorbell_first_cut and not irregular_gemm_b_tail_scratchpad_output_hold_first_cut) else ''}"
            f"{'--pure-gemm-peeled-rect-v1 ' if (pure_gemm_peeled_rect_v1 and not pure_gemm_peeled_rect_v2_right_edge_rectified and not pure_gemm_peeled_rect_batched_single_doorbell_first_cut and not irregular_gemm_b_tail_scratchpad_output_hold_first_cut) else ''}"
            f"{'--pure-gemm-pack-tail-align16 ' if (pure_gemm_pack_tail_align16 and not pure_gemm_peeled_rect_v1 and not pure_gemm_peeled_rect_v2_right_edge_rectified and not pure_gemm_peeled_rect_batched_single_doorbell_first_cut and not irregular_gemm_b_tail_scratchpad_output_hold_first_cut) else ''}"
            "--allow-local-trigger "
            f"> {terminal_log} 2>&1"
        )
        rc = run_gem5_with_progress(cmd, serial_log, terminal_log, env=env)
        if rc != 0:
            run_status = f"failed({rc})"
    return run_status, m5out_dir, terminal_log, serial_log


def build_row(
    case: OracleCase,
    matrix_size: int,
    device_link_gbs: int,
    run_status: str,
    m5out_dir: Path,
    terminal_log: Path,
    serial_log: Path,
    pure_gemm_pack_tail_align16: bool,
    pure_gemm_peeled_rect_v1: bool,
    pure_gemm_peeled_rect_v2_right_edge_rectified: bool,
    pure_gemm_peeled_rect_batched_single_doorbell_first_cut: bool,
    irregular_gemm_b_tail_scratchpad_output_hold_first_cut: bool,
) -> dict:
    stats_file = m5out_dir / "stats.txt"
    engine = extract_engine_stats(stats_file)
    engine = patch_engine_stats_from_terminal_log(engine, terminal_log)
    phase = extract_phase_timings(serial_log)
    raw = extract_raw_timings(serial_log)
    tick_info = extract_start_done_ticks(terminal_log)
    gemm_active_s = extract_gemm_active_time(str(terminal_log))
    workload_ok = (
        serial_log_has_pure_gemm(serial_log)
        and engine["dmaRead"] > 0
        and engine["computeCycles"] > 0
        and gemm_active_s > 0.0
    )

    actual_dma_total = engine["dmaRead"] + engine["dmaWrite"]
    request_count = engine["dram_num_reads"] + engine["dram_num_writes"]
    avg_request_payload = (
        actual_dma_total / request_count if request_count > 0 else 0.0
    )
    geom = derive_geometry_metrics(
        matrix_size=matrix_size,
        actual_dma_bytes_total=actual_dma_total,
        compute_cycles=engine["computeCycles"],
        gemm_active_s=gemm_active_s,
        phase1_begin_ns=raw["phase1_begin_ns"],
        phase1_end_ns=raw["phase1_end_ns"],
        start_tick_ps=tick_info["start_tick_ps"],
        done_tick_ps=tick_info["done_tick_ps"],
    )
    gemm_effective = (
        (2.0 * (matrix_size**3) / 1e9) / gemm_active_s
        if gemm_active_s > 0
        else 0.0
    )

    return {
        "Case": case.label,
        "Case Note": case.note,
        "Device Link (GB/s)": device_link_gbs,
        "Matrix Size": matrix_size,
        "Tail Pack Align16": pure_gemm_pack_tail_align16,
        "Peeled Rect V1": pure_gemm_peeled_rect_v1,
        "Peeled Rect V2 Right Edge Rectified": (
            pure_gemm_peeled_rect_v2_right_edge_rectified
        ),
        "Peeled Rect Batched Single Doorbell First Cut": (
            pure_gemm_peeled_rect_batched_single_doorbell_first_cut
        ),
        "Irregular GEMM B-tail Scratchpad Output Hold First Cut": (
            irregular_gemm_b_tail_scratchpad_output_hold_first_cut
        ),
        "Run Status": run_status,
        "Workload OK": workload_ok,
        "engine_metrics_source": engine["engine_metrics_source"],
        "phase1_ms": round(phase["phase1_ms"], 6),
        "phase2_total_ms": round(phase["phase2_total_ms"], 6),
        "end_to_end_ms": round(phase["end_to_end_ms"], 6),
        "GEMM Active Time (s)": round(gemm_active_s, 9),
        "GEMM-Effective (GFLOPS)": round(
            gemm_effective if workload_ok else 0.0, 6
        ),
        "tail_tile_count": int(geom["tail_tile_count"]),
        "tail_tile_ratio": round(geom["tail_tile_ratio"], 6),
        "edge_tile_count": int(geom["edge_tile_count"]),
        "useful_mac_ratio": round(geom["useful_mac_ratio"], 6),
        "padded_mac_ratio": round(geom["padded_mac_ratio"], 6),
        "tail_compute_waste": int(geom["tail_compute_waste"]),
        "dma_bytes_total": int(actual_dma_total),
        "dma_useful_bytes": int(geom["dma_useful_bytes"]),
        "dma_overfetch_bytes": int(geom["dma_overfetch_bytes"]),
        "tail_dma_overfetch_bytes": int(geom["tail_dma_overfetch_bytes"]),
        "request_count": int(request_count),
        "avg_request_payload": round(avg_request_payload, 6),
        "small_request_count": int(geom["small_request_count"]),
        "row_request_count_oracle": int(geom["row_request_count_oracle"]),
        "doorbell_launch_count": int(phase.get("doorbell_launch_count", 0.0)),
        "descriptor_count": int(phase.get("descriptor_launch_count", 0.0)),
        "completion_count": int(tick_info["completion_count"]),
        "phase1_poll_count": int(phase.get("phase1_poll_count", 0.0)),
        "phase3_poll_count": int(phase.get("phase3_poll_count", 0.0)),
        "launch_overhead_cycles": round(geom["launch_overhead_cycles"], 2),
        "completion_overhead_cycles": round(
            geom["completion_overhead_cycles"], 2
        ),
        "pe_active_ratio": round(geom["pe_active_ratio"], 6),
        "tile_utilization_avg": round(geom["tile_utilization_avg"], 6),
        "tail_tile_utilization_avg": round(
            geom["tail_tile_utilization_avg"], 6
        ),
        "compute_stall_cycles": round(geom["compute_stall_cycles"], 2),
        "effective_flops_per_launched_tile": round(
            geom["effective_flops_per_launched_tile"], 2
        ),
        "vip_rescue_hit_count": int(engine["vip_rescue_hit_count"]),
        "mhot_hit_count": int(engine["mhot_hit_count"]),
        "coverage_gather_issue_count": int(
            engine["coverage_gather_issue_count"]
        ),
        "coverage_shadow_hit_count": int(engine["coverage_shadow_hit_count"]),
        "fallback_b_rows_fetched": int(engine["fallback_b_rows_fetched"]),
        "normal_fetch_hole_rows": int(engine["normal_fetch_hole_rows"]),
        "a_path_stall_waiting_for_b": int(
            engine["a_path_stall_waiting_for_b"]
        ),
        "b_path_stall_waiting_for_a": int(
            engine["b_path_stall_waiting_for_a"]
        ),
        "rx_a_stall_cycles": int(engine["rx_a_stall_cycles"]),
        "rx_b_stall_cycles": int(engine["rx_b_stall_cycles"]),
        "m5out_dir": str(m5out_dir),
        "terminal_log": str(terminal_log),
        "serial_log": str(serial_log),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_report(rows: list[dict]) -> str:
    if not rows:
        return "no rows\n"

    by_case_size = {(r["Case"], int(r["Matrix Size"])): r for r in rows}

    lines = []
    lines.append("pure_gemm257_oracle_autopsy")
    lines.append("")
    lines.append("1. 各矩阵尺寸对照表（clean baseline）")
    lines.append(
        "Case,Size,GEMM-Effective,phase1_ms,tail_tile_ratio,useful_mac_ratio,"
        "dma_overfetch_bytes,tail_dma_overfetch_bytes,request_count,"
        "avg_request_payload,small_request_count,pe_active_ratio"
    )
    for size in sorted(
        {
            int(r["Matrix Size"])
            for r in rows
            if r["Case"] == "irregular_gemm_oracle_clean_baseline"
        }
    ):
        row = by_case_size.get(("irregular_gemm_oracle_clean_baseline", size))
        if not row:
            continue
        lines.append(
            f"{row['Case']},{size},{row['GEMM-Effective (GFLOPS)']},"
            f"{row['phase1_ms']},{row['tail_tile_ratio']},"
            f"{row['useful_mac_ratio']},{row['dma_overfetch_bytes']},"
            f"{row['tail_dma_overfetch_bytes']},{row['request_count']},"
            f"{row['avg_request_payload']},{row['small_request_count']},"
            f"{row['pe_active_ratio']}"
        )

    row256 = by_case_size.get(("irregular_gemm_oracle_clean_baseline", 256))
    row257 = by_case_size.get(("irregular_gemm_oracle_clean_baseline", 257))
    row272 = by_case_size.get(("irregular_gemm_oracle_clean_baseline", 272))
    if row256 and row257 and row272:
        lines.append("")
        lines.append("2. 256 vs 257 vs 272 关键差异（clean baseline）")
        lines.append(
            f"256 -> GEMM {row256['GEMM-Effective (GFLOPS)']} GFLOPS, "
            f"useful_mac_ratio={row256['useful_mac_ratio']}, "
            f"tail_tile_ratio={row256['tail_tile_ratio']}, "
            f"dma_overfetch_bytes={row256['dma_overfetch_bytes']}"
        )
        lines.append(
            f"257 -> GEMM {row257['GEMM-Effective (GFLOPS)']} GFLOPS, "
            f"useful_mac_ratio={row257['useful_mac_ratio']}, "
            f"tail_tile_ratio={row257['tail_tile_ratio']}, "
            f"dma_overfetch_bytes={row257['dma_overfetch_bytes']}, "
            f"tail_dma_overfetch_bytes={row257['tail_dma_overfetch_bytes']}, "
            f"small_request_count={row257['small_request_count']}"
        )
        lines.append(
            f"272 -> GEMM {row272['GEMM-Effective (GFLOPS)']} GFLOPS, "
            f"useful_mac_ratio={row272['useful_mac_ratio']}, "
            f"tail_tile_ratio={row272['tail_tile_ratio']}, "
            f"dma_overfetch_bytes={row272['dma_overfetch_bytes']}"
        )

        lines.append("")
        lines.append("3. pure_gemm257 五类死因（以 clean baseline 为准）")
        lines.append(
            f"Tail/padding: tail_tile_ratio={row257['tail_tile_ratio']}, "
            f"useful_mac_ratio={row257['useful_mac_ratio']}, "
            f"tail_compute_waste={row257['tail_compute_waste']}"
        )
        lines.append(
            f"DMA: dma_overfetch_bytes={row257['dma_overfetch_bytes']}, "
            f"tail_dma_overfetch_bytes={row257['tail_dma_overfetch_bytes']}, "
            f"request_count={row257['request_count']}, "
            f"avg_request_payload={row257['avg_request_payload']}, "
            f"small_request_count={row257['small_request_count']}"
        )
        lines.append(
            f"Control plane: descriptor_count={row257['descriptor_count']}, "
            f"doorbell_launch_count={row257['doorbell_launch_count']}, "
            f"completion_count={row257['completion_count']}, "
            f"phase1_poll_count={row257['phase1_poll_count']}, "
            f"launch_overhead_cycles={row257['launch_overhead_cycles']}, "
            f"completion_overhead_cycles={row257['completion_overhead_cycles']}"
        )
        lines.append(
            f"Array utilization: pe_active_ratio={row257['pe_active_ratio']}, "
            f"tile_utilization_avg={row257['tile_utilization_avg']}, "
            f"tail_tile_utilization_avg={row257['tail_tile_utilization_avg']}, "
            f"compute_stall_cycles={row257['compute_stall_cycles']}"
        )
        lines.append(
            f"Sidecar noise: vip_rescue_hit_count={row257['vip_rescue_hit_count']}, "
            f"mhot_hit_count={row257['mhot_hit_count']}, "
            f"coverage_gather_issue_count={row257['coverage_gather_issue_count']}, "
            f"coverage_shadow_hit_count={row257['coverage_shadow_hit_count']}, "
            f"fallback_b_rows_fetched={row257['fallback_b_rows_fetched']}, "
            f"normal_fetch_hole_rows={row257['normal_fetch_hole_rows']}"
        )

        lines.append("")
        lines.append("4. 结构必然 vs 实现额外损失")
        lines.append(
            "结构必然: useful_mac_ratio 低于 1、tail_tile_ratio > 0、"
            "tail_dma_overfetch_bytes > 0 代表 257 相对 256 的几何/粒度天然损失。"
        )
        lines.append(
            "实现额外: dma_overfetch_bytes 高于 tail_dma_overfetch_bytes、"
            "compute_stall_cycles 偏大、以及 sidecar/fallback 统计上升，"
            "代表当前实现额外制造的损失。"
        )

        lines.append("")
        lines.append("5. irregular_gemm mode 参数建议（基于 clean baseline 方向）")
        lines.append("- 默认关闭 VIP / coverage gather / aggressive hole filling。")
        lines.append("- 默认弱化或关闭 M-hot。")
        lines.append(
            "- 将 next_prefetch_trigger 收紧到 b_ready，降低 speculative depth。"
        )
        lines.append("- 将 rows_b 收紧到 32 或更小，先以少搬运为目标。")
        lines.append("- 保留最简单主线，优先保证 irregular GEMM 不被 sidecar 添乱。")

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    sizes = parse_sizes(args.sizes)
    cases = selected_cases(args.cases)
    ensure_libm5()
    OUTPUT_DIR.mkdir(exist_ok=True)

    print(
        "[*] pure_gemm257_oracle_autopsy: cases="
        + ", ".join(case.label for case in cases)
    )
    print("[*] sizes=" + ", ".join(str(v) for v in sizes))

    if not args.skip_compile:
        run_cmd(COMPILE_CMD)
    if not args.skip_inject:
        run_cmd("sudo ./inject_trigger_gemm.sh --no-compile 257")

    rows = []
    for size in sizes:
        for case in cases:
            print(f"\n=== {case.label} / pure_gemm{size} ===")
            run_status, m5out_dir, terminal_log, serial_log = run_one_case(
                case=case,
                matrix_size=size,
                device_link_gbs=args.device_link_gbs,
                cpu_type=args.cpu_type,
                keep_existing=args.keep_existing,
                pure_gemm_pack_tail_align16=args.pure_gemm_pack_tail_align16,
                pure_gemm_peeled_rect_v1=args.pure_gemm_peeled_rect_v1,
                pure_gemm_peeled_rect_v2_right_edge_rectified=(
                    args.pure_gemm_peeled_rect_v2_right_edge_rectified
                ),
                pure_gemm_peeled_rect_batched_single_doorbell_first_cut=(
                    args.pure_gemm_peeled_rect_batched_single_doorbell_first_cut
                ),
                irregular_gemm_b_tail_scratchpad_output_hold_first_cut=(
                    args.irregular_gemm_b_tail_scratchpad_output_hold_first_cut
                ),
            )
            row = build_row(
                case=case,
                matrix_size=size,
                device_link_gbs=args.device_link_gbs,
                run_status=run_status,
                m5out_dir=m5out_dir,
                terminal_log=terminal_log,
                serial_log=serial_log,
                pure_gemm_pack_tail_align16=args.pure_gemm_pack_tail_align16,
                pure_gemm_peeled_rect_v1=args.pure_gemm_peeled_rect_v1,
                pure_gemm_peeled_rect_v2_right_edge_rectified=(
                    args.pure_gemm_peeled_rect_v2_right_edge_rectified
                ),
                pure_gemm_peeled_rect_batched_single_doorbell_first_cut=(
                    args.pure_gemm_peeled_rect_batched_single_doorbell_first_cut
                ),
                irregular_gemm_b_tail_scratchpad_output_hold_first_cut=(
                    args.irregular_gemm_b_tail_scratchpad_output_hold_first_cut
                ),
            )
            rows.append(row)
            print(
                f"[{'√' if row['Workload OK'] else '!'}] {case.label} / {size}: "
                f"GEMM={row['GEMM-Effective (GFLOPS)']} GFLOPS, "
                f"phase1_ms={row['phase1_ms']}, "
                f"dma_overfetch={row['dma_overfetch_bytes']}, "
                f"status={row['Run Status']}"
            )

    write_csv(SUMMARY_RAW_CSV, rows)
    write_csv(SUMMARY_ANALYSIS_CSV, rows)
    SUMMARY_REPORT_TXT.write_text(build_report(rows), encoding="utf-8")
    print(f"[done] raw -> {SUMMARY_RAW_CSV}")
    print(f"[done] analysis -> {SUMMARY_ANALYSIS_CSV}")
    print(f"[done] report -> {SUMMARY_REPORT_TXT}")


if __name__ == "__main__":
    main()
