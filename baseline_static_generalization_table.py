#!/usr/bin/env python3
"""Run baseline/static_classifier comparison points and aggregate one table."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/jzx8091/SimCXL-main")
OUTDIR = ROOT / "my_outputData"
PURE_SUMMARY = OUTDIR / "pure_gemm_cxl_summary.csv"
VIT_SUMMARY = OUTDIR / "cxl_vit_proxy_data.csv"
RESULTS_CSV = OUTDIR / "baseline_static_generalization_results.csv"
RESULTS_MD = OUTDIR / "baseline_static_generalization_results.md"

PURE_SIZES = [256, 2048, 257, 272, 513, 258]
VIT_PRESETS = ["ViT-Base-like", "ViT-Large-like", "ViT-Huge-like"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run baseline/static_classifier comparison for pure GEMM and ViT proxy."
    )
    parser.add_argument(
        "--device-link-gbs",
        "--device_link_gbs",
        dest="device_link_gbs",
        type=int,
        default=32,
        choices=[16, 32, 64, 128, 256],
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print commands without executing.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip rows already present in the combined CSV.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N points.",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Pass through --keep-existing to underlying runners.",
    )
    return parser.parse_args()


def load_existing_results() -> dict[tuple[str, str, str], dict[str, str]]:
    if not RESULTS_CSV.exists():
        return {}
    with RESULTS_CSV.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return {
        (row["workload"], row["case_name"], row["mode"]): row for row in rows
    }


def read_single_row(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return rows[-1] if rows else {}


def build_points(
    device_link_gbs: int, keep_existing: bool
) -> list[dict[str, object]]:
    points: list[dict[str, object]] = []
    pure_compile_done = False
    for size in PURE_SIZES:
        for mode in ("baseline", "static_classifier"):
            cmd = [
                sys.executable,
                "pure_gemm_cxl_run.py",
                "--matrix-size",
                str(size),
                "--device-link-gbs",
                str(device_link_gbs),
            ]
            if pure_compile_done:
                cmd.append("--skip-compile")
            if mode == "static_classifier":
                cmd.append(
                    "--irregular-gemm-static-output-tile-classifier-boundary-hold-first-cut"
                )
                cmd.append("--skip-inject")
            if keep_existing:
                cmd.append("--keep-existing")
            points.append(
                {
                    "workload": "pure_gemm",
                    "case_name": f"pure{size}",
                    "mode": mode,
                    "cmd": cmd,
                    "summary": PURE_SUMMARY,
                }
            )
        pure_compile_done = True

    for preset in VIT_PRESETS:
        compile_for_preset = True
        for mode in ("baseline", "static_classifier"):
            cmd = [
                sys.executable,
                "vit_proxy_cxl_run.py",
                "--device-link-gbs",
                str(device_link_gbs),
                "--preset-names",
                preset,
            ]
            if compile_for_preset:
                pass
            else:
                cmd.extend(["--skip-compile", "--skip-inject"])
            if mode == "static_classifier":
                cmd.append(
                    "--irregular-gemm-static-output-tile-classifier-boundary-hold-first-cut"
                )
            if keep_existing:
                cmd.append("--keep-existing")
            points.append(
                {
                    "workload": "vit_proxy",
                    "case_name": preset,
                    "mode": mode,
                    "cmd": cmd,
                    "summary": VIT_SUMMARY,
                }
            )
            compile_for_preset = False
    return points


def normalize_row(
    point: dict[str, object], rc: int | None, runner_row: dict[str, str]
) -> dict[str, str]:
    row = {
        "workload": str(point["workload"]),
        "case_name": str(point["case_name"]),
        "mode": str(point["mode"]),
        "run_status": runner_row.get(
            "Run Status", runner_row.get("Run Status".lower(), "")
        ),
        "workload_ok": runner_row.get("Workload OK", ""),
        "device_link_gbs": runner_row.get("Device Link (GB/s)", ""),
        "matrix_size": "",
        "seq_len": "",
        "phase1_ms": runner_row.get("phase1_ms", ""),
        "phase2_total_ms": runner_row.get("phase2_total_ms", ""),
        "phase2_non_gemm_ms": runner_row.get("phase2_non_gemm_ms", ""),
        "phase3_ms": runner_row.get("phase3_ms", ""),
        "end_to_end_ms": runner_row.get("end_to_end_ms", ""),
        "clean_phase1_device_us": runner_row.get("clean_phase1_device_us", ""),
        "gemm_active_time_s": runner_row.get("GEMM Active Time (s)", ""),
        "gemm_effective_gflops": runner_row.get("GEMM-Effective (GFLOPS)", ""),
        "phase2_proxy_throughput_gflops": runner_row.get(
            "Phase2 Proxy Throughput (GFLOPS)", ""
        ),
        "non_gemm_effective_gflops": runner_row.get(
            "Non-GEMM Effective (GFLOPS)", ""
        ),
        "system_effective_gflops": runner_row.get(
            "System-Effective (GFLOPS)", ""
        ),
        "compute_share_pct": runner_row.get("ComputeShare (%)", ""),
        "dma_read": runner_row.get(
            "dmaRead", runner_row.get("totalDmaRead", "")
        ),
        "dma_write": runner_row.get(
            "dmaWrite", runner_row.get("totalDmaWrite", "")
        ),
        "compute_cycles": runner_row.get(
            "computeCycles", runner_row.get("totalComputeCycles", "")
        ),
        "m5out_dir": runner_row.get("m5out_dir", ""),
        "terminal_log": runner_row.get("terminal_log", ""),
        "serial_log": runner_row.get("serial_log", ""),
        "return_code": "" if rc is None else str(rc),
    }
    if point["workload"] == "pure_gemm":
        row["matrix_size"] = runner_row.get("Matrix Size", "")
        row["run_status"] = runner_row.get("Run Status", row["run_status"])
        row["workload_ok"] = runner_row.get("Workload OK", row["workload_ok"])
        row["device_link_gbs"] = runner_row.get(
            "Device Link (GB/s)", row["device_link_gbs"]
        )
    else:
        row["seq_len"] = runner_row.get("SeqLen", "")
    if not row["run_status"]:
        row["run_status"] = (
            f"failed({rc})" if rc not in (None, 0) else "unknown"
        )
    return row


def write_outputs(rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    headers = list(rows[0].keys())
    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "| Workload | Case | Mode | Status | GEMM GFLOPS | clean_phase1_device_us | phase1_ms | phase2_total_ms | Non-GEMM GFLOPS | Phase2 Proxy GFLOPS | end_to_end_ms |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {workload} | {case_name} | {mode} | {run_status} | {gemm_effective_gflops} | {clean_phase1_device_us} | {phase1_ms} | {phase2_total_ms} | {non_gemm_effective_gflops} | {phase2_proxy_throughput_gflops} | {end_to_end_ms} |".format(
                **row
            )
        )
    RESULTS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    existing = load_existing_results() if args.resume else {}
    points = build_points(args.device_link_gbs, args.keep_existing)
    if args.limit is not None:
        points = points[: args.limit]

    rows: list[dict[str, str]] = list(existing.values())
    done = set(existing.keys())

    print(f"points={len(points)} results_csv={RESULTS_CSV}")
    for idx, point in enumerate(points, start=1):
        key = (
            str(point["workload"]),
            str(point["case_name"]),
            str(point["mode"]),
        )
        if key in done:
            print(f"[-] skip resume {key}")
            continue

        cmd = [str(part) for part in point["cmd"]]
        pretty = " ".join(cmd)
        print(
            f"\n[{idx}/{len(points)}] {point['workload']} {point['case_name']} {point['mode']}"
        )
        print(pretty)
        if args.dry_run:
            rows.append(
                {
                    "workload": key[0],
                    "case_name": key[1],
                    "mode": key[2],
                    "run_status": "dry_run",
                    "workload_ok": "",
                    "device_link_gbs": str(args.device_link_gbs),
                    "matrix_size": key[1].replace("pure", "")
                    if key[0] == "pure_gemm"
                    else "",
                    "seq_len": "",
                    "phase1_ms": "",
                    "phase2_total_ms": "",
                    "phase2_non_gemm_ms": "",
                    "phase3_ms": "",
                    "end_to_end_ms": "",
                    "clean_phase1_device_us": "",
                    "gemm_active_time_s": "",
                    "gemm_effective_gflops": "",
                    "phase2_proxy_throughput_gflops": "",
                    "non_gemm_effective_gflops": "",
                    "system_effective_gflops": "",
                    "compute_share_pct": "",
                    "dma_read": "",
                    "dma_write": "",
                    "compute_cycles": "",
                    "m5out_dir": "",
                    "terminal_log": "",
                    "serial_log": "",
                    "return_code": "",
                }
            )
            done.add(key)
            continue

        rc = subprocess.run(cmd, cwd=ROOT).returncode
        runner_row = read_single_row(Path(point["summary"]))
        row = normalize_row(point, rc, runner_row)
        rows.append(row)
        done.add(key)
        write_outputs(rows)

    write_outputs(rows)
    print(f"\ncombined csv: {RESULTS_CSV}")
    print(f"markdown table: {RESULTS_MD}")


if __name__ == "__main__":
    main()
