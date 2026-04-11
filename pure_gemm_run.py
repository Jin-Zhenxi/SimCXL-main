#!/usr/bin/env python3
"""Run 2048x2048 pure GEMM with run_sweep architecture presets."""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
from pathlib import Path

from run_sweep import (
    MAX_INVALID_RUN_RETRIES,
    PREFETCH_CONFIGS,
    PRESETS,
    RUN_ONLY_LABELS,
    RUN_ONLY_PRESET_NAMES,
)


GEM5_ROOT = Path("/home/jzx8091/SimCXL-main")
OUTPUT_DIR = GEM5_ROOT / "my_outputData"
TRIGGER_BIN = GEM5_ROOT / "trigger_gemm"
TRIGGER_SRC = GEM5_ROOT / "trigger_gemm.c"
GEM5_BIN = GEM5_ROOT / "build/X86/gem5.opt"
PURE_GEMM_SCRIPT = (
    "configs/example/gem5_library/x86-cxl-type3-with-classic-pure-gemm.py"
)
LIBM5_PATH = GEM5_ROOT / "util/m5/build/x86/out/libm5.a"
COMPILE_CMD = (
    f"gcc -O3 -static -mno-80387 -o {TRIGGER_BIN} {TRIGGER_SRC} "
    f"-I{GEM5_ROOT}/include -L{GEM5_ROOT}/util/m5/build/x86/out -lm -lm5"
)
INJECT_CMD_TEMPLATE = "sudo ./inject_trigger_gemm.sh --no-compile {matrix_size}"
CHUNK_SEPARATOR = "---------- Begin Simulation Statistics ----------"
CLOCK_FREQ_GHZ = 2.4
SUMMARY_CSV = OUTPUT_DIR / "pure_gemm_summary.csv"
SUMMARY_TXT = OUTPUT_DIR / "pure_gemm_summary.txt"
DEFAULT_LABELS = [
    cfg["label"] for cfg in PREFETCH_CONFIGS if cfg["label"] in RUN_ONLY_LABELS
]
DEFAULT_LABELS_ARG = ",".join(DEFAULT_LABELS)
DEFAULT_PRESET_NAME = next(
    preset["name"] for preset in PRESETS if preset["name"] in RUN_ONLY_PRESET_NAMES
)


def ensure_libm5() -> None:
    if LIBM5_PATH.exists():
        return
    raise SystemExit(
        "libm5.a 未找到。请先执行: "
        f"cd {GEM5_ROOT / 'util/m5'} && scons build/x86/out/libm5.a"
    )


def run_cmd(cmd: str, *, env: dict[str, str] | None = None) -> None:
    print(f"[*] {cmd}")
    subprocess.run(cmd, shell=True, cwd=GEM5_ROOT, env=env, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 2048x2048 pure_gemm with run_sweep architecture presets."
    )
    parser.add_argument(
        "--matrix-size",
        "--matrix_size",
        dest="matrix_size",
        type=int,
        default=2048,
        help="Matrix size passed to pure_gemm.",
    )
    parser.add_argument(
        "--labels",
        "--architectures",
        dest="labels",
        type=str,
        default=DEFAULT_LABELS_ARG,
        help="Comma-separated run_sweep config labels to run.",
    )
    parser.add_argument(
        "--preset-name",
        "--preset_name",
        dest="preset_name",
        type=str,
        default=DEFAULT_PRESET_NAME,
        help="run_sweep preset profile used to resolve preset-specific knobs.",
    )
    parser.add_argument(
        "--cpu-type",
        "--cpu_type",
        dest="cpu_type",
        choices=["TIMING", "O3"],
        default="TIMING",
        help="Detailed CPU type after boot.",
    )
    parser.add_argument(
        "--boot-cpu-type",
        "--boot_cpu_type",
        dest="boot_cpu_type",
        choices=["KVM", "ATOMIC"],
        default="KVM",
        help="Boot CPU type before switching to the detailed core.",
    )
    parser.add_argument(
        "--skip-compile",
        action="store_true",
        help="Reuse existing trigger_gemm binary.",
    )
    parser.add_argument(
        "--skip-inject",
        action="store_true",
        help="Assume the correct trigger_gemm is already injected into parsec.img.",
    )
    parser.add_argument(
        "--build-p2p",
        action="store_true",
        help="Run ./build_p2p.sh before the GEMM runs.",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not delete existing m5out/log files before rerunning.",
    )
    return parser.parse_args()


def selected_configs(arg: str) -> list[dict[str, object]]:
    labels = [item.strip() for item in arg.split(",") if item.strip()]
    available = {cfg["label"]: cfg for cfg in PREFETCH_CONFIGS}
    unknown = sorted(set(labels) - set(available))
    if unknown:
        raise SystemExit(f"未知 run_sweep 配置: {', '.join(unknown)}")
    return [available[label] for label in labels]


def resolve_preset(name: str) -> dict[str, object]:
    for preset in PRESETS:
        if preset["name"] == name:
            return preset
    raise SystemExit(f"未知 preset: {name}")


def rows_b_for_config(cfg: dict[str, object], preset_name: str) -> int:
    return int(cfg.get("rows_b_by_preset", {}).get(preset_name, cfg["rows_b"]))


def build_env(cfg: dict[str, object], preset_name: str) -> dict[str, str]:
    rows_b = rows_b_for_config(cfg, preset_name)
    env = os.environ.copy()
    env["MATRIXFLOW_NEXT_PREFETCH_MODE"] = str(cfg["mode"])
    env["MATRIXFLOW_NEXT_PREFETCH_TRIGGER"] = str(cfg["trigger"])
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
    env["MATRIXFLOW_AB_SCHEDULER_MODE"] = str(
        cfg.get("ab_scheduler_mode", "baseline")
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
    return env


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


def extract_phase_timings(serial_log: Path) -> dict[str, float]:
    keys = [
        "phase1_ms",
        "phase2_total_ms",
        "phase3_ms",
        "end_to_end_ms",
        "gemm_tile_count_per_gemm",
        "gemm_tile_count_total",
        "descriptor_launch_count",
        "doorbell_launch_count",
        "phase1_poll_count",
        "phase3_poll_count",
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
            match = workload_pattern.search(line)
            if match:
                values["workload"] = match.group(1)
            for key, pattern in patterns.items():
                match = pattern.search(line)
                if match:
                    values[key] = float(match.group(1))

    return values


def serial_log_has_pure_gemm(serial_log: Path) -> bool:
    if not serial_log.exists():
        return False
    text = serial_log.read_text(encoding="utf-8", errors="ignore")
    return (
        "[Timing] workload=pure_gemm" in text
        and "[Phase 2] skipped for pure_gemm" in text
        and "[Phase 3] skipped for pure_gemm" in text
    )


def summarize_result(
    cfg: dict[str, object],
    preset_name: str,
    matrix_size: int,
    stats_file: Path,
    serial_log: Path,
    m5out_dir: Path,
    terminal_log: Path,
) -> dict[str, object]:
    metrics = extract_roi_metrics(stats_file)
    phase = extract_phase_timings(serial_log)
    gemm_flops = 2.0 * (matrix_size**3)
    gemm_wall_s = phase["phase1_ms"] / 1.0e3
    compute_time_s = (
        metrics["computeCycles"] * 2.0 / (CLOCK_FREQ_GHZ * 1.0e9)
        if metrics["computeCycles"] > 0
        else 0.0
    )
    peak_gflops = (
        ((gemm_flops / 2.0) / 1.0e9)
        / (metrics["computeCycles"] / (CLOCK_FREQ_GHZ * 1.0e9))
        if metrics["computeCycles"] > 0
        else 0.0
    )
    effective_gflops = (
        (gemm_flops / 1.0e9) / gemm_wall_s if gemm_wall_s > 0 else 0.0
    )
    compute_share = (
        100.0 * compute_time_s / gemm_wall_s if gemm_wall_s > 0 else 0.0
    )
    dma_bw = (
        (metrics["dmaRead"] + metrics["dmaWrite"]) / 1.0e9 / gemm_wall_s
        if gemm_wall_s > 0
        else 0.0
    )

    return {
        "Config Label": str(cfg["label"]),
        "Preset Profile": preset_name,
        "Matrix Size": matrix_size,
        "Workload OK": serial_log_has_pure_gemm(serial_log),
        "Scheduler Mode": str(cfg.get("ab_scheduler_mode", "baseline")),
        "Rows A": int(cfg["rows_a"]),
        "Rows B": rows_b_for_config(cfg, preset_name),
        "VIP B Rows Capacity": int(cfg.get("vip_b_rows_capacity", 0)),
        "MHot B Rows Capacity": int(cfg.get("mhot_b_rows_capacity", 0)),
        "Coverage Shadow Rows Capacity": int(
            cfg.get("coverage_shadow_rows_capacity", 0)
        ),
        "Coverage Gather Min Budget": int(
            cfg.get("coverage_gather_min_issue_budget", 0)
        ),
        "ROI simSeconds": round(metrics["simSeconds"], 9),
        "totalComputeCycles": int(metrics["computeCycles"]),
        "DMA Read Bytes": int(metrics["dmaRead"]),
        "DMA Write Bytes": int(metrics["dmaWrite"]),
        "Phase-1 GEMM Time (ms)": round(phase["phase1_ms"], 6),
        "Phase-2 Time (ms)": round(phase["phase2_total_ms"], 6),
        "Phase-3 Time (ms)": round(phase["phase3_ms"], 6),
        "End-to-End Time (ms)": round(phase["end_to_end_ms"], 6),
        "descriptor_launch_count": int(phase["descriptor_launch_count"]),
        "doorbell_launch_count": int(phase["doorbell_launch_count"]),
        "gemm_tile_count_total": int(phase["gemm_tile_count_total"]),
        "GEMM Compute Share (%)": round(compute_share, 6),
        "Peak MAC Throughput (GFLOPS)": round(peak_gflops, 6),
        "GEMM-Effective (GFLOPS)": round(effective_gflops, 6),
        "GEMM DMA BW (GB/s)": round(dma_bw, 6),
        "m5out_dir": str(m5out_dir),
        "terminal_log": str(terminal_log),
        "serial_log": str(serial_log),
    }


def write_results(rows: list[dict[str, object]]) -> None:
    if not rows:
        return

    OUTPUT_DIR.mkdir(exist_ok=True)
    headers: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                headers.append(key)

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    lines = []
    for row in rows:
        lines.append(
            f"{row['Config Label']}: "
            f"ok={row['Workload OK']}, "
            f"phase1_ms={row['Phase-1 GEMM Time (ms)']}, "
            f"end_to_end_ms={row['End-to-End Time (ms)']}, "
            f"GEMM-Effective={row['GEMM-Effective (GFLOPS)']} GFLOPS, "
            f"ComputeShare={row['GEMM Compute Share (%)']}%"
        )
    SUMMARY_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    preset = resolve_preset(args.preset_name)
    configs = selected_configs(args.labels)

    ensure_libm5()

    print("[*] 运行 pure_gemm，复用 run_sweep 架构配置")
    print(f"[*] preset profile: {preset['name']}")
    print(f"[*] labels: {', '.join(cfg['label'] for cfg in configs)}")

    if args.build_p2p:
        run_cmd("./build_p2p.sh")

    if not args.skip_compile:
        run_cmd(COMPILE_CMD)

    if not args.skip_inject:
        run_cmd(INJECT_CMD_TEMPLATE.format(matrix_size=args.matrix_size))

    results: list[dict[str, object]] = []
    for cfg in configs:
        label = str(cfg["label"])
        env = build_env(cfg, preset["name"])
        run_tag = f"pure_gemm_{args.matrix_size}_{label}"
        m5out_dir = OUTPUT_DIR / f"m5out_{run_tag}"
        terminal_log = OUTPUT_DIR / f"terminal_log_{run_tag}.txt"
        serial_log = m5out_dir / "board.pc.com_1.device"
        stats_file = m5out_dir / "stats.txt"

        if not args.keep_existing:
            if m5out_dir.exists():
                shutil.rmtree(m5out_dir)
            if terminal_log.exists():
                terminal_log.unlink()

        cmd = (
            f"{GEM5_BIN} -p {GEM5_ROOT / 'src/python'} "
            f"-d {m5out_dir} "
            f"{PURE_GEMM_SCRIPT} "
            f"--is_asic True "
            f"--cpu_type {args.cpu_type} "
            f"--boot_cpu_type {args.boot_cpu_type} "
            "--no-network "
            f"--matrixflow_size {args.matrix_size} "
            "--allow-local-trigger "
            f"> {terminal_log} 2>&1"
        )

        for attempt in range(1, MAX_INVALID_RUN_RETRIES + 1):
            run_cmd(cmd, env=env)
            result = summarize_result(
                cfg=cfg,
                preset_name=preset["name"],
                matrix_size=args.matrix_size,
                stats_file=stats_file,
                serial_log=serial_log,
                m5out_dir=m5out_dir,
                terminal_log=terminal_log,
            )
            valid_metrics = not (
                result["ROI simSeconds"] == 0.0
                and result["DMA Read Bytes"] == 0
                and result["DMA Write Bytes"] == 0
                and result["totalComputeCycles"] == 0
            )
            if result["Workload OK"] and valid_metrics:
                results.append(result)
                print(
                    f"[√] {label} 完成: "
                    f"phase1_ms={result['Phase-1 GEMM Time (ms)']}, "
                    f"GEMM-Effective={result['GEMM-Effective (GFLOPS)']} GFLOPS, "
                    f"ComputeShare={result['GEMM Compute Share (%)']}%"
                )
                break

            print(
                f"[!] 无效 run，准备重试 {label} "
                f"({attempt}/{MAX_INVALID_RUN_RETRIES})"
            )
            if attempt < MAX_INVALID_RUN_RETRIES and not args.keep_existing:
                if m5out_dir.exists():
                    shutil.rmtree(m5out_dir)
                if terminal_log.exists():
                    terminal_log.unlink()
        else:
            raise SystemExit(f"多次重试后仍未拿到有效结果: {label}")

    write_results(results)
    print(f"[*] 结果已写入 {SUMMARY_CSV}")
    print(f"[*] 摘要已写入 {SUMMARY_TXT}")


if __name__ == "__main__":
    main()
