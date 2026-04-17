#!/usr/bin/env python3
"""Run pure_gemm on the CXL Type-3 DDR5 HDM baseline with debug-based metrics."""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from run_sweep import extract_gemm_active_time

GEM5_ROOT = Path("/home/jzx8091/SimCXL-main")
OUTPUT_DIR = GEM5_ROOT / "my_outputData"
TRIGGER_BIN = GEM5_ROOT / "trigger_gemm"
TRIGGER_SRC = GEM5_ROOT / "trigger_gemm.c"
GEM5_BIN = GEM5_ROOT / "build/X86/gem5.opt"
GEM5_SCRIPT = (
    "configs/example/gem5_library/x86-cxl-type3-with-classic-matrixflow.py"
)
LIBM5_PATH = GEM5_ROOT / "util/m5/build/x86/out/libm5.a"
COMPILE_CMD = (
    f"gcc -O3 -static -mno-80387 -o {TRIGGER_BIN} {TRIGGER_SRC} "
    f"-I{GEM5_ROOT}/include -L{GEM5_ROOT}/util/m5/build/x86/out -lm -lm5"
)
CHUNK_SEPARATOR = "---------- Begin Simulation Statistics ----------"
CLOCK_FREQ_GHZ = 2.4
SUMMARY_CSV = OUTPUT_DIR / "pure_gemm_cxl_summary.csv"
SUMMARY_TXT = OUTPUT_DIR / "pure_gemm_cxl_summary.txt"
GEM5_TICKS_PER_US = 1.0e6


def ensure_libm5() -> None:
    if LIBM5_PATH.exists():
        return
    raise SystemExit(
        "libm5.a 未找到。请先执行: "
        f"cd {GEM5_ROOT / 'util/m5'} && scons build/x86/out/libm5.a"
    )


def run_cmd(cmd: str) -> None:
    print(f"[*] {cmd}")
    subprocess.run(cmd, shell=True, cwd=GEM5_ROOT, check=True)


def run_gem5_with_progress(
    cmd: str,
    serial_log: Path,
    terminal_log: Path,
    env: dict[str, str] | None = None,
) -> int:
    print(f"[*] {cmd}")
    print(
        "[*] 仿真已启动。2048 这条在 CXL 上通常需要几十秒宿主机时间；"
        "看到 boot 日志但暂时没进 benchmark 是正常的。"
    )
    proc = subprocess.Popen(cmd, shell=True, cwd=GEM5_ROOT, env=env)
    start = time.time()
    last_report = 0.0
    benchmark_seen = False

    while True:
        rc = proc.poll()
        now = time.time()
        if now - last_report >= 5.0:
            last_report = now
            serial_size = (
                serial_log.stat().st_size if serial_log.exists() else 0
            )
            term_size = (
                terminal_log.stat().st_size if terminal_log.exists() else 0
            )
            phase = "booting"
            panic_seen = False
            if serial_log.exists():
                try:
                    with serial_log.open(
                        encoding="utf-8", errors="ignore"
                    ) as fh:
                        for line in fh:
                            if "Using trigger binary:" in line:
                                phase = "trigger_started"
                                benchmark_seen = True
                            elif "[Phase 1] GEMM1 on device HDM..." in line:
                                phase = "phase1_running"
                                benchmark_seen = True
                            elif "MatrixFlow benchmark finished" in line:
                                phase = "benchmark_finished"
                                benchmark_seen = True
                except OSError:
                    pass
            if terminal_log.exists():
                try:
                    with terminal_log.open(
                        encoding="utf-8", errors="ignore"
                    ) as fh:
                        for line in fh:
                            if (
                                " panic:" in line
                                or "Program aborted at tick 0" in line
                                or "fatal:" in line
                            ):
                                panic_seen = True
                                break
                except OSError:
                    pass

            elapsed = now - start
            print(
                f"[*] 进度: {phase}, elapsed={elapsed:.1f}s, "
                f"serial_log={serial_size}B, terminal_log={term_size}B"
            )
            if panic_seen:
                print(
                    "[!] terminal log detected panic/fatal, stopping this run."
                )
                proc.terminate()
                try:
                    proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5.0)
                return proc.returncode if proc.returncode is not None else 1

        if rc is not None:
            if benchmark_seen:
                print(f"[*] gem5 进程结束，返回码={rc}")
            else:
                print(f"[!] gem5 进程结束，返回码={rc}；本次运行在 benchmark 启动前就结束了。")
            return rc

        time.sleep(1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CXL pure_gemm with corrected debug-based metrics."
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
    parser.add_argument(
        "--irregular-gemm-fused-edges-completion-optimized-first-cut",
        "--irregular_gemm_fused_edges_completion_optimized_first_cut",
        dest="irregular_gemm_fused_edges_completion_optimized_first_cut",
        action="store_true",
    )
    parser.add_argument(
        "--irregular-gemm-fused-right-edge-clean-timing-first-cut",
        "--irregular_gemm_fused_right_edge_clean_timing_first_cut",
        dest="irregular_gemm_fused_right_edge_clean_timing_first_cut",
        action="store_true",
    )
    parser.add_argument(
        "--irregular-gemm-residual-device-overhead-autopsy-first-cut",
        "--irregular_gemm_residual_device_overhead_autopsy_first_cut",
        dest="irregular_gemm_residual_device_overhead_autopsy_first_cut",
        action="store_true",
    )
    parser.add_argument(
        "--irregular-gemm-no-wait-fused-right-first-cut",
        "--irregular_gemm_no_wait_fused_right_first_cut",
        dest="irregular_gemm_no_wait_fused_right_first_cut",
        action="store_true",
    )
    parser.add_argument(
        "--irregular-gemm-no-wait-fused-bottom-first-cut",
        "--irregular_gemm_no_wait_fused_bottom_first_cut",
        dest="irregular_gemm_no_wait_fused_bottom_first_cut",
        action="store_true",
    )
    parser.add_argument(
        "--irregular-gemm-single-fused-descriptor-corner-collapse-first-cut",
        "--irregular_gemm_single_fused_descriptor_corner_collapse_first_cut",
        dest="irregular_gemm_single_fused_descriptor_corner_collapse_first_cut",
        action="store_true",
    )
    parser.add_argument(
        "--irregular-gemm-final-completion-chain-autopsy-first-cut",
        "--irregular_gemm_final_completion_chain_autopsy_first_cut",
        dest="irregular_gemm_final_completion_chain_autopsy_first_cut",
        action="store_true",
    )
    parser.add_argument(
        "--irregular-gemm-boundary-only-hold-early-body-writeback-first-cut",
        "--irregular_gemm_boundary_only_hold_early_body_writeback_first_cut",
        dest="irregular_gemm_boundary_only_hold_early_body_writeback_first_cut",
        action="store_true",
    )
    parser.add_argument(
        "--irregular-gemm-static-output-tile-classifier-boundary-hold-first-cut",
        "--irregular_gemm_static_output_tile_classifier_boundary_hold_first_cut",
        dest="irregular_gemm_static_output_tile_classifier_boundary_hold_first_cut",
        action="store_true",
    )
    parser.add_argument(
        "--irregular-gemm-boundary-writeback-coalescing-first-cut",
        "--irregular_gemm_boundary_writeback_coalescing_first_cut",
        dest="irregular_gemm_boundary_writeback_coalescing_first_cut",
        action="store_true",
    )
    parser.add_argument(
        "--irregular-gemm-streaming-body-writeback-first-cut",
        "--irregular_gemm_streaming_body_writeback_first_cut",
        dest="irregular_gemm_streaming_body_writeback_first_cut",
        action="store_true",
    )
    parser.add_argument(
        "--interior-writeback-stripe-rows",
        "--interior_writeback_stripe_rows",
        dest="interior_writeback_stripe_rows",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--max-outstanding-stripes",
        "--max_outstanding_stripes",
        dest="max_outstanding_stripes",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--boundary-right-writeback-bytes",
        "--boundary_right_writeback_bytes",
        dest="boundary_right_writeback_bytes",
        type=int,
        default=4,
    )
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--skip-inject", action="store_true")
    parser.add_argument("--keep-existing", action="store_true")
    return parser.parse_args()


def extract_roi_metrics(stats_file: Path) -> dict[str, float]:
    metrics = {
        "simSeconds": 0.0,
        "dmaRead": 0.0,
        "dmaWrite": 0.0,
        "computeCycles": 0.0,
    }
    if not stats_file.exists():
        return metrics

    text = stats_file.read_text(encoding="utf-8", errors="ignore")
    chunks = text.split(CHUNK_SEPARATOR)
    target = ""
    for chunk in chunks:
        match = re.search(r"totalComputeCycles\s+(\d+)", chunk)
        if match and int(match.group(1)) > 0:
            target = chunk
            break
    if not target and chunks:
        target = chunks[-1]

    for key, pattern in {
        "simSeconds": r"simSeconds\s+([0-9.]+)",
        "dmaRead": r"totalDmaBytesRead\s+(\d+)",
        "dmaWrite": r"totalDmaBytesWritten\s+(\d+)",
        "computeCycles": r"totalComputeCycles\s+(\d+)",
    }.items():
        match = re.search(pattern, target)
        if match:
            metrics[key] = float(match.group(1))

    return metrics


def patch_metrics_from_debug_log(
    metrics: dict[str, float], terminal_log: Path
) -> dict[str, float]:
    if not terminal_log.exists():
        return metrics

    subproblem_pat = re.compile(
        r"MatrixFlow subproblem done: .*"
        r"batch=(\d+) index=(\d+) .*"
        r"dmaRead=(\d+) dmaWrite=(\d+) computeCycles=(\d+)"
    )
    completion_pat = re.compile(
        r"MatrixFlow completion flag written: .*"
        r"dmaRead=(\d+) dmaWrite=(\d+) computeCycles=(\d+)"
    )
    sane_limit = float(1 << 50)
    sane_subproblems: dict[tuple[int, int], dict[str, float]] = {}
    sane_completion_totals = {
        "dmaRead": 0.0,
        "dmaWrite": 0.0,
        "computeCycles": 0.0,
    }
    saw_subproblem = False
    saw_completion = False

    with terminal_log.open(encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            match = subproblem_pat.search(line)
            if match:
                saw_subproblem = True
                batch_id = int(match.group(1))
                index = int(match.group(2))
                dma_read = float(match.group(3))
                dma_write = float(match.group(4))
                compute_cycles = float(match.group(5))
                if (
                    dma_read < sane_limit
                    and dma_write < sane_limit
                    and compute_cycles < sane_limit
                ):
                    sane_subproblems[(batch_id, index)] = {
                        "dmaRead": dma_read,
                        "dmaWrite": dma_write,
                        "computeCycles": compute_cycles,
                    }
                continue

            match = completion_pat.search(line)
            if match:
                saw_completion = True
                dma_read = float(match.group(1))
                dma_write = float(match.group(2))
                compute_cycles = float(match.group(3))
                if (
                    dma_read < sane_limit
                    and dma_write < sane_limit
                    and compute_cycles < sane_limit
                ):
                    sane_completion_totals["dmaRead"] += dma_read
                    sane_completion_totals["dmaWrite"] += dma_write
                    sane_completion_totals["computeCycles"] += compute_cycles

    if sane_subproblems:
        totals = {"dmaRead": 0.0, "dmaWrite": 0.0, "computeCycles": 0.0}
        for entry in sane_subproblems.values():
            totals["dmaRead"] += entry["dmaRead"]
            totals["dmaWrite"] += entry["dmaWrite"]
            totals["computeCycles"] += entry["computeCycles"]
        metrics["dmaRead"] = totals["dmaRead"]
        metrics["computeCycles"] = totals["computeCycles"]
        metrics["dmaWrite"] = max(metrics["dmaWrite"], totals["dmaWrite"])
    elif saw_completion and sane_completion_totals["computeCycles"] > 0:
        metrics.update(sane_completion_totals)
    elif saw_subproblem or saw_completion:
        metrics["dmaRead"] = max(metrics["dmaRead"], 0.0)
        metrics["dmaWrite"] = max(metrics["dmaWrite"], 0.0)
        metrics["computeCycles"] = max(metrics["computeCycles"], 0.0)
    return metrics


def extract_phase_timings(serial_log: Path) -> dict[str, float]:
    keys = [
        "phase1_ms",
        "clean_phase1_device_us",
        "phase2_total_ms",
        "phase3_ms",
        "end_to_end_ms",
        "gemm_tile_count_total",
        "phase1_poll_count",
        "phase3_poll_count",
        "adaptivePollBackoffCount",
    ]
    values = {key: 0.0 for key in keys}
    values["workload"] = "unknown"
    if not serial_log.exists():
        return values

    workload_pattern = re.compile(r"\[Timing\] workload=([A-Za-z0-9_]+)")
    patterns = {
        key: re.compile(rf"\[Timing\] {re.escape(key)}=([0-9.]+)")
        for key in keys
    }
    with serial_log.open(encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            workload_match = workload_pattern.search(line)
            if workload_match:
                values["workload"] = workload_match.group(1)
            for key, pattern in patterns.items():
                match = pattern.search(line)
                if match:
                    values[key] = float(match.group(1))
    return values


def extract_residual_autopsy(
    terminal_log: Path, clean_device_us: float
) -> dict[str, float | bool]:
    result: dict[str, float | bool] = {
        "deviceWindowBeginTick": 0.0,
        "deviceWindowEndTick": 0.0,
        "deviceCompletionFlagWriteTick": 0.0,
        "finalUsefulWorkDoneTick": 0.0,
        "descriptorDecodeOverheadUs": 0.0,
        "batchTransitionGapUs": 0.0,
        "tailDependencyWaitUs": 0.0,
        "dmaDrainWaitUs": 0.0,
        "deviceFinalCompletionOverheadUs": 0.0,
        "body_active_us": 0.0,
        "right_edge_active_us": 0.0,
        "bottom_edge_active_us": 0.0,
        "corner_final_active_us": 0.0,
        "sum_subproblem_active_us": 0.0,
        "residualDeviceOverheadUs": 0.0,
        "engineDeviceWindowUs": 0.0,
        "bodyPostGapUs": 0.0,
        "rightEdgeNoOpGapUs": 0.0,
        "bottomEdgePreStartGapUs": 0.0,
        "cornerFinalGapUs": 0.0,
        "batchStepCount": 0.0,
        "batchNoOpStepCount": 0.0,
        "batchTransitionCount": 0.0,
        "fusedRightEdgeAReuseFired": False,
        "rightEdgeDescriptorNoOp": False,
        "tailScratchpadLoadBTail": False,
        "tailScratchpadHitBTail": False,
    }
    if not terminal_log.exists():
        return result

    device_begin_pat = re.compile(r"deviceWindowBegin: tick=(\d+)")
    device_end_pat = re.compile(r"deviceWindowEnd: tick=(\d+)")
    completion_pat = re.compile(r"deviceCompletionFlagWrite: tick=(\d+)")
    final_useful_pat = re.compile(r"finalUsefulWorkDone: index=\d+ tick=(\d+)")
    active_pat = re.compile(
        r"batchStepActiveWindow: index=(\d+) first=\d+ last=\d+ cycles=(\d+)"
    )
    summary_pat = re.compile(
        r"residualDeviceOverheadSummary: windowCycles=(\d+) "
        r"activeCycles=(\d+) residualCycles=(\d+) "
        r"decodeCycles=(\d+) transitionCycles=(\d+) "
        r"tailWaitCycles=(\d+) dmaDrainCycles=(\d+) "
        r"finalCompletionCycles=(\d+) "
        r"bodyActiveCycles=(\d+) rightActiveCycles=(\d+) "
        r"bottomActiveCycles=(\d+) cornerActiveCycles=(\d+) "
        r"batchSteps=(\d+) noOpSteps=(\d+) transitions=(\d+)"
    )
    subproblem_done_pat = re.compile(
        r"MatrixFlow subproblem done: .*dims=\((\d+),(\d+),(\d+)\).*batch=(\d+) index=(\d+)"
    )

    with terminal_log.open(encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if "fusedRightEdgeAReuse fired" in line:
                result["fusedRightEdgeAReuseFired"] = True
            if "rightEdgeDescriptor converted to no-op" in line:
                result["rightEdgeDescriptorNoOp"] = True
            if "tailScratchpadLoad(B-tail)" in line:
                result["tailScratchpadLoadBTail"] = True
            if "tailScratchpadHit(B-tail)" in line:
                result["tailScratchpadHitBTail"] = True

            match = device_begin_pat.search(line)
            if match:
                result["deviceWindowBeginTick"] = float(match.group(1))
                continue
            match = device_end_pat.search(line)
            if match:
                result["deviceWindowEndTick"] = float(match.group(1))
                continue
            match = completion_pat.search(line)
            if match:
                result["deviceCompletionFlagWriteTick"] = float(match.group(1))
                continue
            match = final_useful_pat.search(line)
            if match:
                result["finalUsefulWorkDoneTick"] = float(match.group(1))
                continue
            match = active_pat.search(line)
            if match:
                index = int(match.group(1))
                active_us = float(match.group(2)) / (CLOCK_FREQ_GHZ * 1e3)
                if index == 0:
                    result["body_active_us"] = active_us
                elif index == 1:
                    result["right_edge_active_us"] = active_us
                elif index == 2:
                    result["bottom_edge_active_us"] = active_us
                elif index == 3:
                    result["corner_final_active_us"] = active_us
                continue
            match = summary_pat.search(line)
            if match:
                window_cycles = float(match.group(1))
                active_cycles = float(match.group(2))
                residual_cycles = float(match.group(3))
                result["engineDeviceWindowUs"] = window_cycles / (
                    CLOCK_FREQ_GHZ * 1e3
                )
                result["sum_subproblem_active_us"] = active_cycles / (
                    CLOCK_FREQ_GHZ * 1e3
                )
                result["residualDeviceOverheadUs"] = residual_cycles / (
                    CLOCK_FREQ_GHZ * 1e3
                )
                result["descriptorDecodeOverheadUs"] = float(
                    match.group(4)
                ) / (CLOCK_FREQ_GHZ * 1e3)
                result["batchTransitionGapUs"] = float(match.group(5)) / (
                    CLOCK_FREQ_GHZ * 1e3
                )
                result["tailDependencyWaitUs"] = float(match.group(6)) / (
                    CLOCK_FREQ_GHZ * 1e3
                )
                result["dmaDrainWaitUs"] = float(match.group(7)) / (
                    CLOCK_FREQ_GHZ * 1e3
                )
                result["deviceFinalCompletionOverheadUs"] = float(
                    match.group(8)
                ) / (CLOCK_FREQ_GHZ * 1e3)
                result["body_active_us"] = float(match.group(9)) / (
                    CLOCK_FREQ_GHZ * 1e3
                )
                result["right_edge_active_us"] = float(match.group(10)) / (
                    CLOCK_FREQ_GHZ * 1e3
                )
                result["bottom_edge_active_us"] = float(match.group(11)) / (
                    CLOCK_FREQ_GHZ * 1e3
                )
                result["corner_final_active_us"] = float(match.group(12)) / (
                    CLOCK_FREQ_GHZ * 1e3
                )
                result["batchStepCount"] = float(match.group(13))
                result["batchNoOpStepCount"] = float(match.group(14))
                result["batchTransitionCount"] = float(match.group(15))
                continue
            match = subproblem_done_pat.search(line)
            if match:
                dims = tuple(int(match.group(i)) for i in range(1, 4))
                index = int(match.group(5))
                if index == 1 and dims[1] == 1:
                    result["rightEdgeDescriptorNoOp"] = result[
                        "rightEdgeDescriptorNoOp"
                    ] or ("computeCycles=0" in line)

    if result["deviceWindowBeginTick"] and result["deviceWindowEndTick"]:
        result["engineDeviceWindowUs"] = (
            result["deviceWindowEndTick"] - result["deviceWindowBeginTick"]
        ) / GEM5_TICKS_PER_US
    if (
        result["finalUsefulWorkDoneTick"]
        and result["deviceCompletionFlagWriteTick"]
    ):
        result["deviceFinalCompletionOverheadUs"] = (
            result["deviceCompletionFlagWriteTick"]
            - result["finalUsefulWorkDoneTick"]
        ) / GEM5_TICKS_PER_US
    if result["sum_subproblem_active_us"] <= 0.0:
        result["sum_subproblem_active_us"] = (
            result["body_active_us"]
            + result["right_edge_active_us"]
            + result["bottom_edge_active_us"]
            + result["corner_final_active_us"]
        )
    if clean_device_us > 0.0 and result["sum_subproblem_active_us"] > 0.0:
        result["residualDeviceOverheadUs"] = max(
            0.0, clean_device_us - result["sum_subproblem_active_us"]
        )
    return result


def serial_log_has_pure_gemm(serial_log: Path) -> bool:
    if not serial_log.exists():
        return False
    text = serial_log.read_text(encoding="utf-8", errors="ignore")
    return (
        "[Timing] workload=pure_gemm" in text
        and "[Phase 2] skipped for pure_gemm" in text
        and "[Phase 3] skipped for pure_gemm" in text
        and "MatrixFlow benchmark finished" in text
    )


def main() -> None:
    args = parse_args()
    ensure_libm5()
    OUTPUT_DIR.mkdir(exist_ok=True)

    if not args.skip_compile:
        run_cmd(COMPILE_CMD)

    if not args.skip_inject:
        run_cmd(
            f"sudo ./inject_trigger_gemm.sh --no-compile {args.matrix_size}"
        )

    run_tag = f"pure_gemm_{args.matrix_size}_cxl_{args.device_link_gbs}g"
    if args.irregular_gemm_fused_edges_completion_optimized_first_cut:
        run_tag += "__fused01"
    if args.irregular_gemm_no_wait_fused_right_first_cut:
        run_tag += "__no_wait_right"
    if args.irregular_gemm_no_wait_fused_bottom_first_cut:
        run_tag += "__no_wait_bottom"
    if args.irregular_gemm_single_fused_descriptor_corner_collapse_first_cut:
        run_tag += "__single_fused_corner"
    if args.irregular_gemm_final_completion_chain_autopsy_first_cut:
        run_tag += "__completion_autopsy"
    if args.irregular_gemm_boundary_only_hold_early_body_writeback_first_cut:
        run_tag += "__boundary_earlywb"
    if (
        args.irregular_gemm_static_output_tile_classifier_boundary_hold_first_cut
    ):
        run_tag += "__static_classifier"
    if args.irregular_gemm_boundary_writeback_coalescing_first_cut:
        run_tag += "__boundary_coalesce"
    if args.irregular_gemm_streaming_body_writeback_first_cut:
        run_tag += "__stream_body_wb"
    if args.interior_writeback_stripe_rows > 0:
        run_tag += f"__wbstripe{args.interior_writeback_stripe_rows}"
    if args.max_outstanding_stripes > 0:
        run_tag += f"__wbout{args.max_outstanding_stripes}"
    if args.boundary_right_writeback_bytes != 4:
        run_tag += f"__rightwb{args.boundary_right_writeback_bytes}"
    if args.irregular_gemm_residual_device_overhead_autopsy_first_cut:
        run_tag += "__autopsy01"
    if args.irregular_gemm_fused_right_edge_clean_timing_first_cut:
        run_tag += "__fused_right_clean"
    elif args.irregular_gemm_b_tail_scratchpad_output_hold_first_cut:
        run_tag += "__tail01"
    elif args.pure_gemm_peeled_rect_batched_single_doorbell_first_cut:
        run_tag += "__peeled_batch1"
    elif args.pure_gemm_peeled_rect_v2_right_edge_rectified:
        run_tag += "__peeledv2_rightedge"
    elif args.pure_gemm_peeled_rect_v1:
        run_tag += "__peeledv1"
    elif args.pure_gemm_pack_tail_align16:
        run_tag += "__pack16"
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
        print("[*] 复用已有 m5out/log，仅重新汇总指标。")
    else:
        gem5_env = os.environ.copy()
        gem5_env["M5_SOURCE_OBJECTS_FALLBACK"] = "1"
        if args.interior_writeback_stripe_rows > 0:
            gem5_env["MATRIXFLOW_BODY_INTERIOR_WRITEBACK_STRIPE_ROWS"] = str(
                args.interior_writeback_stripe_rows
            )
        if args.max_outstanding_stripes > 0:
            gem5_env[
                "MATRIXFLOW_BODY_INTERIOR_WRITEBACK_MAX_OUTSTANDING_STRIPES"
            ] = str(args.max_outstanding_stripes)
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
            f"{f'--interior-writeback-stripe-rows {args.interior_writeback_stripe_rows} ' if args.interior_writeback_stripe_rows > 0 else ''}"
            f"{f'--max-outstanding-stripes {args.max_outstanding_stripes} ' if args.max_outstanding_stripes > 0 else ''}"
            f"{f'--boundary-right-writeback-bytes {args.boundary_right_writeback_bytes} ' if args.boundary_right_writeback_bytes != 4 else ''}"
            f"{'--irregular-gemm-streaming-body-writeback-first-cut ' if args.irregular_gemm_streaming_body_writeback_first_cut else ''}"
            f"{'--irregular-gemm-boundary-writeback-coalescing-first-cut ' if args.irregular_gemm_boundary_writeback_coalescing_first_cut else ''}"
            f"{'--irregular-gemm-static-output-tile-classifier-boundary-hold-first-cut ' if args.irregular_gemm_static_output_tile_classifier_boundary_hold_first_cut else ''}"
            f"{'--irregular-gemm-boundary-only-hold-early-body-writeback-first-cut ' if args.irregular_gemm_boundary_only_hold_early_body_writeback_first_cut else ''}"
            f"{'--irregular-gemm-final-completion-chain-autopsy-first-cut ' if args.irregular_gemm_final_completion_chain_autopsy_first_cut else ''}"
            f"{'--irregular-gemm-single-fused-descriptor-corner-collapse-first-cut ' if args.irregular_gemm_single_fused_descriptor_corner_collapse_first_cut else ''}"
            f"{'--irregular-gemm-no-wait-fused-bottom-first-cut ' if args.irregular_gemm_no_wait_fused_bottom_first_cut else ''}"
            f"{'--irregular-gemm-no-wait-fused-right-first-cut ' if args.irregular_gemm_no_wait_fused_right_first_cut else ''}"
            f"{'--irregular-gemm-residual-device-overhead-autopsy-first-cut ' if args.irregular_gemm_residual_device_overhead_autopsy_first_cut else ''}"
            f"{'--irregular-gemm-fused-right-edge-clean-timing-first-cut ' if (args.irregular_gemm_fused_right_edge_clean_timing_first_cut and not args.irregular_gemm_final_completion_chain_autopsy_first_cut and not args.irregular_gemm_no_wait_fused_right_first_cut and not args.irregular_gemm_no_wait_fused_bottom_first_cut and not args.irregular_gemm_single_fused_descriptor_corner_collapse_first_cut) else ''}"
            f"{'--irregular-gemm-fused-edges-completion-optimized-first-cut ' if (args.irregular_gemm_fused_edges_completion_optimized_first_cut and not args.irregular_gemm_final_completion_chain_autopsy_first_cut and not args.irregular_gemm_fused_right_edge_clean_timing_first_cut and not args.irregular_gemm_residual_device_overhead_autopsy_first_cut and not args.irregular_gemm_no_wait_fused_right_first_cut and not args.irregular_gemm_no_wait_fused_bottom_first_cut and not args.irregular_gemm_single_fused_descriptor_corner_collapse_first_cut) else ''}"
            f"{'--irregular-gemm-b-tail-scratchpad-output-hold-first-cut ' if (args.irregular_gemm_b_tail_scratchpad_output_hold_first_cut and not args.irregular_gemm_final_completion_chain_autopsy_first_cut and not args.irregular_gemm_fused_edges_completion_optimized_first_cut and not args.irregular_gemm_fused_right_edge_clean_timing_first_cut and not args.irregular_gemm_no_wait_fused_right_first_cut and not args.irregular_gemm_no_wait_fused_bottom_first_cut and not args.irregular_gemm_single_fused_descriptor_corner_collapse_first_cut) else ''}"
            f"{'--pure-gemm-peeled-rect-batched-single-doorbell-first-cut ' if (args.pure_gemm_peeled_rect_batched_single_doorbell_first_cut and not args.irregular_gemm_final_completion_chain_autopsy_first_cut and not args.irregular_gemm_b_tail_scratchpad_output_hold_first_cut and not args.irregular_gemm_fused_edges_completion_optimized_first_cut and not args.irregular_gemm_fused_right_edge_clean_timing_first_cut and not args.irregular_gemm_no_wait_fused_right_first_cut and not args.irregular_gemm_no_wait_fused_bottom_first_cut and not args.irregular_gemm_single_fused_descriptor_corner_collapse_first_cut) else ''}"
            f"{'--pure-gemm-peeled-rect-v2-right-edge-rectified ' if (args.pure_gemm_peeled_rect_v2_right_edge_rectified and not args.irregular_gemm_final_completion_chain_autopsy_first_cut and not args.pure_gemm_peeled_rect_batched_single_doorbell_first_cut and not args.irregular_gemm_b_tail_scratchpad_output_hold_first_cut and not args.irregular_gemm_fused_edges_completion_optimized_first_cut and not args.irregular_gemm_fused_right_edge_clean_timing_first_cut and not args.irregular_gemm_no_wait_fused_right_first_cut and not args.irregular_gemm_no_wait_fused_bottom_first_cut and not args.irregular_gemm_single_fused_descriptor_corner_collapse_first_cut) else ''}"
            f"{'--pure-gemm-peeled-rect-v1 ' if (args.pure_gemm_peeled_rect_v1 and not args.irregular_gemm_final_completion_chain_autopsy_first_cut and not args.pure_gemm_peeled_rect_v2_right_edge_rectified and not args.pure_gemm_peeled_rect_batched_single_doorbell_first_cut and not args.irregular_gemm_b_tail_scratchpad_output_hold_first_cut and not args.irregular_gemm_fused_edges_completion_optimized_first_cut and not args.irregular_gemm_fused_right_edge_clean_timing_first_cut and not args.irregular_gemm_no_wait_fused_right_first_cut and not args.irregular_gemm_no_wait_fused_bottom_first_cut and not args.irregular_gemm_single_fused_descriptor_corner_collapse_first_cut) else ''}"
            f"{'--pure-gemm-pack-tail-align16 ' if (args.pure_gemm_pack_tail_align16 and not args.irregular_gemm_final_completion_chain_autopsy_first_cut and not args.pure_gemm_peeled_rect_v1 and not args.pure_gemm_peeled_rect_v2_right_edge_rectified and not args.pure_gemm_peeled_rect_batched_single_doorbell_first_cut and not args.irregular_gemm_b_tail_scratchpad_output_hold_first_cut and not args.irregular_gemm_fused_edges_completion_optimized_first_cut and not args.irregular_gemm_fused_right_edge_clean_timing_first_cut and not args.irregular_gemm_no_wait_fused_right_first_cut and not args.irregular_gemm_no_wait_fused_bottom_first_cut and not args.irregular_gemm_single_fused_descriptor_corner_collapse_first_cut) else ''}"
            "--allow-local-trigger "
            f"> {terminal_log} 2>&1"
        )
        rc = run_gem5_with_progress(
            cmd, serial_log, terminal_log, env=gem5_env
        )
        if rc != 0:
            run_status = f"failed({rc})"

    metrics = extract_roi_metrics(stats_file)
    metrics = patch_metrics_from_debug_log(metrics, terminal_log)
    phase = extract_phase_timings(serial_log)
    autopsy = extract_residual_autopsy(
        terminal_log, phase["clean_phase1_device_us"]
    )
    gemm_active_s = extract_gemm_active_time(str(terminal_log))
    gemm_wall_s = (
        gemm_active_s
        if gemm_active_s > 0
        else (phase["phase1_ms"] / 1.0e3 if phase["phase1_ms"] > 0 else 0.0)
    )
    flops = 2.0 * (args.matrix_size**3)
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

    row = {
        "Workload": "pure_gemm",
        "Preset": f"pure_gemm_{args.matrix_size}",
        "Run Tag": run_tag,
        "Device Link (GB/s)": args.device_link_gbs,
        "Matrix Size": args.matrix_size,
        "Tail Pack Align16": args.pure_gemm_pack_tail_align16,
        "Peeled Rect V1": args.pure_gemm_peeled_rect_v1,
        "Peeled Rect V2 Right Edge Rectified": (
            args.pure_gemm_peeled_rect_v2_right_edge_rectified
        ),
        "Peeled Rect Batched Single Doorbell First Cut": (
            args.pure_gemm_peeled_rect_batched_single_doorbell_first_cut
        ),
        "Irregular GEMM B-tail Scratchpad Output Hold First Cut": (
            args.irregular_gemm_b_tail_scratchpad_output_hold_first_cut
        ),
        "Irregular GEMM Fused Edges Completion Optimized First Cut": (
            args.irregular_gemm_fused_edges_completion_optimized_first_cut
        ),
        "Irregular GEMM Fused Right Edge Clean Timing First Cut": (
            args.irregular_gemm_fused_right_edge_clean_timing_first_cut
        ),
        "Irregular GEMM Residual Device Overhead Autopsy First Cut": (
            args.irregular_gemm_residual_device_overhead_autopsy_first_cut
        ),
        "Irregular GEMM No-Wait Fused Right First Cut": (
            args.irregular_gemm_no_wait_fused_right_first_cut
        ),
        "Irregular GEMM No-Wait Fused Bottom First Cut": (
            args.irregular_gemm_no_wait_fused_bottom_first_cut
        ),
        "Irregular GEMM Single Fused Descriptor Corner Collapse First Cut": (
            args.irregular_gemm_single_fused_descriptor_corner_collapse_first_cut
        ),
        "Irregular GEMM Final Completion Chain Autopsy First Cut": (
            args.irregular_gemm_final_completion_chain_autopsy_first_cut
        ),
        "Irregular GEMM Boundary Only Hold Early Body Writeback First Cut": (
            args.irregular_gemm_boundary_only_hold_early_body_writeback_first_cut
        ),
        "Irregular GEMM Static Output Tile Classifier Boundary Hold First Cut": (
            args.irregular_gemm_static_output_tile_classifier_boundary_hold_first_cut
        ),
        "Irregular GEMM Boundary Writeback Coalescing First Cut": (
            args.irregular_gemm_boundary_writeback_coalescing_first_cut
        ),
        "Irregular GEMM Streaming Body Writeback First Cut": (
            args.irregular_gemm_streaming_body_writeback_first_cut
        ),
        "Interior Writeback Stripe Rows": args.interior_writeback_stripe_rows,
        "Max Outstanding Stripes": args.max_outstanding_stripes,
        "Run Status": run_status,
        "Workload OK": workload_ok,
        "phase1_ms": round(phase["phase1_ms"], 6),
        "clean_phase1_device_us": round(phase["clean_phase1_device_us"], 6),
        "phase2_total_ms": round(phase["phase2_total_ms"], 6),
        "end_to_end_ms": round(phase["end_to_end_ms"], 6),
        "GEMM Active Time (s)": round(gemm_active_s, 9),
        "GEMM-Effective (GFLOPS)": round(
            gemm_effective if workload_ok else 0.0, 6
        ),
        "ComputeShare (%)": round(compute_share if workload_ok else 0.0, 6),
        "engineDeviceWindowUs": round(
            float(autopsy["engineDeviceWindowUs"]), 6
        ),
        "sum_subproblem_active_us": round(
            float(autopsy["sum_subproblem_active_us"]), 6
        ),
        "residualDeviceOverheadUs": round(
            float(autopsy["residualDeviceOverheadUs"]), 6
        ),
        "descriptorDecodeOverheadUs": round(
            float(autopsy["descriptorDecodeOverheadUs"]), 6
        ),
        "batchTransitionGapUs": round(
            float(autopsy["batchTransitionGapUs"]), 6
        ),
        "tailDependencyWaitUs": round(
            float(autopsy["tailDependencyWaitUs"]), 6
        ),
        "dmaDrainWaitUs": round(float(autopsy["dmaDrainWaitUs"]), 6),
        "deviceFinalCompletionOverheadUs": round(
            float(autopsy["deviceFinalCompletionOverheadUs"]), 6
        ),
        "body_active_us": round(float(autopsy["body_active_us"]), 6),
        "right_edge_active_us": round(
            float(autopsy["right_edge_active_us"]), 6
        ),
        "bottom_edge_active_us": round(
            float(autopsy["bottom_edge_active_us"]), 6
        ),
        "corner_final_active_us": round(
            float(autopsy["corner_final_active_us"]), 6
        ),
        "batchStepCount": int(float(autopsy["batchStepCount"])),
        "batchNoOpStepCount": int(float(autopsy["batchNoOpStepCount"])),
        "batchTransitionCount": int(float(autopsy["batchTransitionCount"])),
        "fusedRightEdgeAReuseFired": bool(
            autopsy["fusedRightEdgeAReuseFired"]
        ),
        "rightEdgeDescriptorNoOp": bool(autopsy["rightEdgeDescriptorNoOp"]),
        "tailScratchpadLoadBTail": bool(autopsy["tailScratchpadLoadBTail"]),
        "tailScratchpadHitBTail": bool(autopsy["tailScratchpadHitBTail"]),
        "dmaRead": int(metrics["dmaRead"]),
        "dmaWrite": int(metrics["dmaWrite"]),
        "computeCycles": int(metrics["computeCycles"]),
        "m5out_dir": str(m5out_dir),
        "terminal_log": str(terminal_log),
        "serial_log": str(serial_log),
    }

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    SUMMARY_TXT.write_text(str(row) + "\n", encoding="utf-8")
    print(
        f"[{'√' if workload_ok else '!'}] {run_tag}: "
        f"phase1_ms={row['phase1_ms']}, "
        f"clean_phase1_device_us={row['clean_phase1_device_us']}, "
        f"GEMM-Effective={row['GEMM-Effective (GFLOPS)']} GFLOPS, "
        f"ComputeShare={row['ComputeShare (%)']}%, "
        f"status={run_status}"
    )


if __name__ == "__main__":
    main()
