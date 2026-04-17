#!/usr/bin/env python3
"""Independent irregular pure-GEMM sweep wrapper.

This script intentionally does not modify existing runners, modes, result
files, or default flows. It writes only under its own result files/prefix.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib
import io
import json
import re
import subprocess
import sys
import traceback
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

GEM5_ROOT = Path("/home/jzx8091/SimCXL-main")
OUTPUT_ROOT = GEM5_ROOT / "my_outputData"
SWEEP_ROOT = OUTPUT_ROOT / "unfactor_pure_runsweep"
RUNS_ROOT = SWEEP_ROOT / "runs"
RESULTS_CSV = OUTPUT_ROOT / "unfactor_pure_runsweep_results.csv"
SUMMARY_JSON = OUTPUT_ROOT / "unfactor_pure_runsweep_summary.json"
RUNNER_PATH = GEM5_ROOT / "pure_gemm_cxl_run.py"

DEFAULT_STRIPE_ROWS = [4, 8, 16, 32]
DEFAULT_MAX_OUTSTANDING = [1, 2, 4]

BASE_MODE_FLAGS = {
    "static_classifier": [
        "--irregular-gemm-static-output-tile-classifier-boundary-hold-first-cut"
    ],
    "boundary_earlywb": [
        "--irregular-gemm-boundary-only-hold-early-body-writeback-first-cut"
    ],
    "final_completion_autopsy": [
        "--irregular-gemm-final-completion-chain-autopsy-first-cut"
    ],
    "single_fused_corner": [
        "--irregular-gemm-single-fused-descriptor-corner-collapse-first-cut"
    ],
    "no_wait_bottom": ["--irregular-gemm-no-wait-fused-bottom-first-cut"],
    "no_wait_right": ["--irregular-gemm-no-wait-fused-right-first-cut"],
    "fused_right_clean": [
        "--irregular-gemm-fused-right-edge-clean-timing-first-cut"
    ],
}

STRIPE_ROWS_FLAG_CANDIDATES = [
    "--interior-writeback-stripe-rows",
    "--body-interior-writeback-stripe-rows",
    "--writeback-stripe-rows",
]
MAX_OUTSTANDING_FLAG_CANDIDATES = [
    "--max-outstanding-stripes",
    "--body-interior-writeback-max-outstanding-stripes",
    "--writeback-max-outstanding-stripes",
]

FIELDNAMES = [
    "label",
    "matrix_size",
    "device_link_gbs",
    "base_mode",
    "stripe_rows",
    "max_outstanding_stripes",
    "clean_device_us",
    "phase1_ms",
    "gemm_effective_gflops",
    "gemm_active_time_us",
    "finalCompletionCycles",
    "writebackResponseDrainCycles",
    "finalWritebackDrainCycles",
    "phase1_poll_count",
    "stripe_params_applied",
    "status",
    "note",
    "return_code",
    "output_dir",
    "terminal_log",
]


@dataclass(frozen=True)
class RunnerCapabilities:
    stripe_rows_flag: str | None
    max_outstanding_flag: str | None
    help_error: str = ""

    @property
    def supports_stripe_sweep(self) -> bool:
        return (
            self.stripe_rows_flag is not None
            and self.max_outstanding_flag is not None
        )


def parse_int_list(value: str) -> list[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError(
            "list must contain at least one integer"
        )
    try:
        parsed = [int(item) for item in items]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("all values must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Independent sweep wrapper for irregular pure-GEMM writeback "
            "stripe parameters."
        )
    )
    parser.add_argument("--matrix-size", type=int, default=257)
    parser.add_argument("--device-link-gbs", type=int, default=32)
    parser.add_argument(
        "--stripe-rows-list",
        type=parse_int_list,
        default=DEFAULT_STRIPE_ROWS,
        help="Comma-separated stripe row sizes. Default: 4,8,16,32",
    )
    parser.add_argument(
        "--max-outstanding-list",
        type=parse_int_list,
        default=DEFAULT_MAX_OUTSTANDING,
        help="Comma-separated max outstanding stripe counts. Default: 1,2,4",
    )
    parser.add_argument(
        "--base-mode",
        choices=sorted(BASE_MODE_FLAGS),
        default="static_classifier",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--skip-compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass --skip-compile to the existing runner. Default: true.",
    )
    parser.add_argument(
        "--skip-inject",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass --skip-inject to the existing runner. Default: true.",
    )
    return parser.parse_args()


def sanitize_label_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def make_label(
    matrix_size: int,
    device_link_gbs: int,
    base_mode: str,
    stripe_rows: int,
    max_outstanding: int,
) -> str:
    mode = sanitize_label_part(base_mode)
    return (
        f"pure{matrix_size}_{mode}_link{device_link_gbs}g_"
        f"stripe{stripe_rows}_out{max_outstanding}"
    )


def read_existing_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def is_legacy_fake_row(row: dict[str, str]) -> bool:
    note = (row.get("note") or "").lower()
    status = (row.get("status") or "").lower()
    stripe_params_applied = str(row.get("stripe_params_applied") or "")
    return (
        "reused baseline run" in note
        or (
            "runner/config stripe cli not detected" in note
            and stripe_params_applied == "0"
            and status in {"success", "dry_run", "unsupported"}
        )
        or (
            "runner lacks stripe cli" in note
            and stripe_params_applied == "0"
            and status in {"success", "dry_run", "unsupported"}
        )
    )


def successful_labels(rows: list[dict[str, str]]) -> set[str]:
    return {row["label"] for row in rows if row.get("status") == "success"}


def write_results(rows: list[dict[str, Any]]) -> None:
    OUTPUT_ROOT.mkdir(exist_ok=True)
    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: row.get(field, "") for field in FIELDNAMES}
            )


def detect_runner_capabilities() -> RunnerCapabilities:
    try:
        proc = subprocess.run(
            [sys.executable, str(RUNNER_PATH), "--help"],
            cwd=GEM5_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    except OSError as exc:
        return RunnerCapabilities(None, None, help_error=str(exc))

    help_text = proc.stdout or ""
    stripe_flag = next(
        (flag for flag in STRIPE_ROWS_FLAG_CANDIDATES if flag in help_text),
        None,
    )
    outstanding_flag = next(
        (
            flag
            for flag in MAX_OUTSTANDING_FLAG_CANDIDATES
            if flag in help_text
        ),
        None,
    )
    help_error = (
        "" if proc.returncode == 0 else f"--help returned {proc.returncode}"
    )
    return RunnerCapabilities(stripe_flag, outstanding_flag, help_error)


def planned_runner_argv(
    args: argparse.Namespace,
    caps: RunnerCapabilities,
    stripe_rows: int,
    max_outstanding: int,
) -> list[str]:
    argv = [
        "pure_gemm_cxl_run.py",
        "--matrix-size",
        str(args.matrix_size),
        "--device-link-gbs",
        str(args.device_link_gbs),
        *BASE_MODE_FLAGS[args.base_mode],
    ]
    if args.skip_compile:
        argv.append("--skip-compile")
    if args.skip_inject:
        argv.append("--skip-inject")
    if caps.stripe_rows_flag is not None:
        argv.extend([caps.stripe_rows_flag, str(stripe_rows)])
    if caps.max_outstanding_flag is not None:
        argv.extend([caps.max_outstanding_flag, str(max_outstanding)])
    return argv


@contextlib.contextmanager
def patched_runner_output(runner: Any, point_root: Path):
    old_output_dir = runner.OUTPUT_DIR
    old_summary_csv = runner.SUMMARY_CSV
    old_summary_txt = runner.SUMMARY_TXT
    try:
        runner.OUTPUT_DIR = point_root
        runner.SUMMARY_CSV = point_root / "pure_gemm_cxl_summary.csv"
        runner.SUMMARY_TXT = point_root / "pure_gemm_cxl_summary.txt"
        yield
    finally:
        runner.OUTPUT_DIR = old_output_dir
        runner.SUMMARY_CSV = old_summary_csv
        runner.SUMMARY_TXT = old_summary_txt


@contextlib.contextmanager
def patched_argv(argv: list[str]):
    old_argv = sys.argv[:]
    try:
        sys.argv = argv[:]
        yield
    finally:
        sys.argv = old_argv


def parse_runner_summary(point_root: Path) -> dict[str, str]:
    summary_csv = point_root / "pure_gemm_cxl_summary.csv"
    if not summary_csv.exists():
        return {}
    with summary_csv.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return rows[0] if rows else {}


def parse_serial_for_poll_count(output_dir: Path | None) -> str:
    if output_dir is None:
        return ""
    serial_log = output_dir / "board.pc.com_1.device"
    if not serial_log.exists():
        return ""
    pat = re.compile(r"\[Timing\]\s+phase1_poll_count=(\d+)")
    with serial_log.open(encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            match = pat.search(line)
            if match:
                return match.group(1)
    return ""


def parse_terminal_completion_metrics(
    terminal_log: Path | None,
) -> dict[str, str]:
    result = {
        "finalCompletionCycles": "",
        "writebackResponseDrainCycles": "",
        "finalWritebackDrainCycles": "",
    }
    if terminal_log is None or not terminal_log.exists():
        return result
    pat = re.compile(
        r"finalCompletionChainSummary: totalCycles=(\d+) "
        r"finalWritebackDrainCycles=(\d+) "
        r"writebackResponseDrainCycles=(\d+)"
    )
    with terminal_log.open(encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            match = pat.search(line)
            if not match:
                continue
            result["finalCompletionCycles"] = match.group(1)
            result["finalWritebackDrainCycles"] = match.group(2)
            result["writebackResponseDrainCycles"] = match.group(3)
    return result


def make_result_row(
    *,
    label: str,
    args: argparse.Namespace,
    stripe_rows: int,
    max_outstanding: int,
    status: str,
    point_root: Path,
    runner_row: dict[str, str] | None = None,
    note: str = "",
    return_code: str | int = "",
    stripe_params_applied: int = 0,
    output_dir_override: str = "",
    terminal_log_override: str = "",
) -> dict[str, Any]:
    runner_row = runner_row or {}
    output_dir_str = output_dir_override or runner_row.get("m5out_dir", "")
    terminal_log_str = terminal_log_override or runner_row.get(
        "terminal_log", ""
    )
    output_dir = Path(output_dir_str) if output_dir_str else None
    terminal_log = Path(terminal_log_str) if terminal_log_str else None
    completion = parse_terminal_completion_metrics(terminal_log)

    active_s = runner_row.get("GEMM Active Time (s)", "")
    active_us = ""
    if active_s not in ("", None):
        try:
            active_us = f"{float(active_s) * 1.0e6:.6f}"
        except ValueError:
            active_us = ""

    return {
        "label": label,
        "matrix_size": args.matrix_size,
        "device_link_gbs": args.device_link_gbs,
        "base_mode": args.base_mode,
        "stripe_rows": stripe_rows,
        "max_outstanding_stripes": max_outstanding,
        "clean_device_us": runner_row.get("clean_phase1_device_us", ""),
        "phase1_ms": runner_row.get("phase1_ms", ""),
        "gemm_effective_gflops": runner_row.get("GEMM-Effective (GFLOPS)", ""),
        "gemm_active_time_us": active_us,
        "finalCompletionCycles": completion["finalCompletionCycles"],
        "writebackResponseDrainCycles": completion[
            "writebackResponseDrainCycles"
        ],
        "finalWritebackDrainCycles": completion["finalWritebackDrainCycles"],
        "phase1_poll_count": parse_serial_for_poll_count(output_dir),
        "stripe_params_applied": stripe_params_applied,
        "status": status,
        "note": note,
        "return_code": return_code,
        "output_dir": output_dir_str or str(point_root),
        "terminal_log": terminal_log_str,
    }


def run_point(
    args: argparse.Namespace,
    caps: RunnerCapabilities,
    label: str,
    stripe_rows: int,
    max_outstanding: int,
    point_root: Path,
) -> dict[str, Any]:
    argv = planned_runner_argv(args, caps, stripe_rows, max_outstanding)
    if args.dry_run:
        print(f"[dry-run] {label}")
        print(f"  output_root={point_root}")
        if not caps.supports_stripe_sweep:
            print("  note=runner/config stripe CLI not detected yet")
        print("  runner argv=" + " ".join(argv))
        return make_result_row(
            label=label,
            args=args,
            stripe_rows=stripe_rows,
            max_outstanding=max_outstanding,
            status="dry_run",
            point_root=point_root,
            note=(
                ""
                if caps.supports_stripe_sweep
                else "runner/config stripe CLI not detected yet"
            ),
        )

    point_root.mkdir(parents=True, exist_ok=True)
    runner = importlib.import_module("pure_gemm_cxl_run")
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    try:
        with patched_runner_output(runner, point_root):
            with patched_argv(argv):
                with contextlib.redirect_stdout(stdout_buf):
                    with contextlib.redirect_stderr(stderr_buf):
                        runner.main()
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        reason = str(exc)
        (point_root / "runner_stdout.txt").write_text(
            stdout_buf.getvalue(), encoding="utf-8"
        )
        (point_root / "runner_stderr.txt").write_text(
            stderr_buf.getvalue(), encoding="utf-8"
        )
        return make_result_row(
            label=label,
            args=args,
            stripe_rows=stripe_rows,
            max_outstanding=max_outstanding,
            status="fail",
            point_root=point_root,
            note=reason,
            return_code=code,
        )
    except Exception as exc:  # keep sweep alive for later points
        (point_root / "runner_stdout.txt").write_text(
            stdout_buf.getvalue(), encoding="utf-8"
        )
        (point_root / "runner_stderr.txt").write_text(
            stderr_buf.getvalue() + "\n" + traceback.format_exc(),
            encoding="utf-8",
        )
        return make_result_row(
            label=label,
            args=args,
            stripe_rows=stripe_rows,
            max_outstanding=max_outstanding,
            status="fail",
            point_root=point_root,
            note=str(exc),
            return_code=1,
        )

    (point_root / "runner_stdout.txt").write_text(
        stdout_buf.getvalue(), encoding="utf-8"
    )
    (point_root / "runner_stderr.txt").write_text(
        stderr_buf.getvalue(), encoding="utf-8"
    )
    runner_row = parse_runner_summary(point_root)
    runner_status = runner_row.get("Run Status", "")
    status = "success" if runner_status == "ok" else "fail"
    return make_result_row(
        label=label,
        args=args,
        stripe_rows=stripe_rows,
        max_outstanding=max_outstanding,
        status=status,
        point_root=point_root,
        runner_row=runner_row,
        note="" if status == "success" else runner_status,
        stripe_params_applied=1 if caps.supports_stripe_sweep else 0,
    )


def print_rankings(rows: list[dict[str, Any]]) -> None:
    successes = [row for row in rows if row.get("status") == "success"]
    if not successes:
        print("\nNo successful sweep points to rank yet.")
        print(
            "If rows are marked unsupported, the current runner/config does not "
            "expose stripe parameters yet."
        )
        return

    def as_float(row: dict[str, Any], key: str, default: float) -> float:
        try:
            return float(row.get(key, ""))
        except (TypeError, ValueError):
            return default

    print("\nTop by GEMM-Effective GFLOPS:")
    for row in sorted(
        successes,
        key=lambda item: as_float(item, "gemm_effective_gflops", -1.0),
        reverse=True,
    )[:5]:
        print(
            f"  {row['label']}: {row['gemm_effective_gflops']} GFLOPS, "
            f"clean={row['clean_device_us']} us"
        )

    print("\nTop by clean_device_us:")
    for row in sorted(
        successes,
        key=lambda item: as_float(item, "clean_device_us", float("inf")),
    )[:5]:
        print(
            f"  {row['label']}: clean={row['clean_device_us']} us, "
            f"GFLOPS={row['gemm_effective_gflops']}"
        )


def write_summary_json(
    rows: list[dict[str, Any]], caps: RunnerCapabilities
) -> None:
    success_count = sum(1 for row in rows if row.get("status") == "success")
    fallback_count = sum(
        1 for row in rows if str(row.get("stripe_params_applied", "")) == "0"
    )
    fail_count = sum(1 for row in rows if row.get("status") == "fail")
    payload = {
        "results_csv": str(RESULTS_CSV),
        "runs_root": str(RUNS_ROOT),
        "runner": str(RUNNER_PATH),
        "stripe_rows_flag": caps.stripe_rows_flag,
        "max_outstanding_flag": caps.max_outstanding_flag,
        "supports_stripe_sweep": caps.supports_stripe_sweep,
        "total_rows": len(rows),
        "success_count": success_count,
        "fallback_count": fallback_count,
        "fail_count": fail_count,
    }
    SUMMARY_JSON.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    caps = detect_runner_capabilities()
    if not caps.supports_stripe_sweep:
        missing = []
        if caps.stripe_rows_flag is None:
            missing.append("stripe rows CLI")
        if caps.max_outstanding_flag is None:
            missing.append("max outstanding stripes CLI")
        detail = ", ".join(missing) if missing else "unknown CLI wiring"
        raise SystemExit(
            "unfactor_pure_runsweep.py: true sweep unavailable because "
            f"runner/config does not expose {detail}. "
            "No sweep results were written. "
            "The previous baseline-reuse fallback has been removed because it "
            "produced misleading identical out1/out2/out4 rows."
        )

    OUTPUT_ROOT.mkdir(exist_ok=True)
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)

    existing = read_existing_rows(RESULTS_CSV)
    existing = [row for row in existing if not is_legacy_fake_row(row)]
    existing_success = successful_labels(existing)
    rows: list[dict[str, Any]] = list(existing)

    grid = list(product(args.stripe_rows_list, args.max_outstanding_list))
    if args.limit and args.limit > 0:
        grid = grid[: args.limit]

    print(f"unfactor pure sweep points: {len(grid)}")
    print(
        f"base_mode={args.base_mode} matrix_size={args.matrix_size} link={args.device_link_gbs}GB/s"
    )
    print(f"results_csv={RESULTS_CSV}")
    print(f"summary_json={SUMMARY_JSON}")

    for stripe_rows, max_outstanding in grid:
        label = make_label(
            args.matrix_size,
            args.device_link_gbs,
            args.base_mode,
            stripe_rows,
            max_outstanding,
        )
        if args.resume and label in existing_success:
            print(f"[resume] skip successful point {label}")
            continue
        point_root = RUNS_ROOT / label
        row = run_point(
            args,
            caps,
            label,
            stripe_rows,
            max_outstanding,
            point_root,
        )
        rows = [old for old in rows if old.get("label") != label]
        rows.append(row)
        write_results(rows)
        print(
            f"[{row['status']}] {label} "
            f"clean={row.get('clean_device_us', '')} "
            f"gflops={row.get('gemm_effective_gflops', '')} "
            f"note={row.get('note', '')}"
        )

    write_results(rows)
    write_summary_json(rows, caps)
    print_rankings(rows)


if __name__ == "__main__":
    main()
