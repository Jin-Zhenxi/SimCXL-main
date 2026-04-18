#!/usr/bin/env python3
"""Run HBM+PCIe baseline generalization points and aggregate one table."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/jzx8091/SimCXL-main")
OUTDIR = ROOT / "my_outputData"
PURE_SUMMARY = OUTDIR / "pure_gemm_hbm_bandwidth_sweep.csv"
VIT_SUMMARY = OUTDIR / "hbm_vit_proxy_data.csv"
RESULTS_CSV = OUTDIR / "hbm_baseline_generalization_results.csv"
RESULTS_MD = OUTDIR / "hbm_baseline_generalization_results.md"

PURE_SIZES = [256, 2048, 257, 272, 513, 258]
VIT_PRESETS = ["ViT-Base-like", "ViT-Large-like", "ViT-Huge-like"]
DEFAULT_DEVICE_LINK_GBS = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run HBM+PCIe baseline generalization table."
    )
    parser.add_argument(
        "--device-link-gbs",
        "--device_link_gbs",
        dest="device_link_gbs",
        type=int,
        default=DEFAULT_DEVICE_LINK_GBS,
        choices=[16, 32, 64, 128, 256],
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--keep-existing", action="store_true")
    return parser.parse_args()


def load_existing_results() -> dict[tuple[str, str], dict[str, str]]:
    if not RESULTS_CSV.exists():
        return {}
    with RESULTS_CSV.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return {(row["workload"], row["case_name"]): row for row in rows}


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
    compile_done = False
    for size in PURE_SIZES:
        cmd = [
            sys.executable,
            "pure_gemm_hbm_run.py",
            "--matrix-size",
            str(size),
            "--bandwidths",
            str(device_link_gbs),
        ]
        if compile_done:
            cmd.append("--skip-compile")
        if keep_existing:
            cmd.append("--keep-existing")
        points.append(
            {
                "workload": "pure_gemm_hbm",
                "case_name": f"pure{size}",
                "cmd": cmd,
                "summary": PURE_SUMMARY,
            }
        )
        compile_done = True

    vit_cmd = [
        sys.executable,
        "vit_proxy_hbm_run.py",
        "--device-link-gbs",
        str(device_link_gbs),
    ]
    if keep_existing:
        vit_cmd.append("--keep-existing")
    points.append(
        {
            "workload": "vit_proxy_hbm_batch",
            "case_name": "vit_all",
            "cmd": vit_cmd,
            "summary": VIT_SUMMARY,
        }
    )
    return points


def normalize_pure_row(
    point: dict[str, object], rc: int, row: dict[str, str]
) -> dict[str, str]:
    expected_case = str(point["case_name"])
    expected_size = expected_case.removeprefix("pure")
    row_matrix_size = row.get("Matrix Size", "")
    row_status = row.get("Run Status", f"failed({rc})" if rc else "unknown")
    row_workload_ok = row.get("Workload OK", "")
    validated_status = row_status

    if rc != 0:
        validated_status = f"failed({rc})"
    elif row_matrix_size != expected_size:
        validated_status = "invalid_case_mismatch"
    elif row_workload_ok not in {"True", "true", "1"}:
        validated_status = "invalid_workload"

    return {
        "workload": str(point["workload"]),
        "case_name": expected_case,
        "run_status": validated_status,
        "workload_ok": row_workload_ok,
        "device_link_gbs": row.get("Device Link (GB/s)", ""),
        "matrix_size": row_matrix_size,
        "seq_len": "",
        "phase1_ms": row.get("Phase-1 GEMM Time (ms)", ""),
        "phase2_total_ms": "",
        "phase2_non_gemm_ms": "",
        "end_to_end_ms": row.get("End-to-End Time (ms)", ""),
        "gemm_active_time_ms": row.get("GEMM Active Time (ms)", ""),
        "gemm_effective_gflops": row.get("GEMM-Effective (GFLOPS)", ""),
        "timing_source": row.get("Timing Source", ""),
        "metrics_source": row.get("Metrics Source", ""),
        "non_gemm_effective_gflops": "",
        "system_effective_gflops": "",
        "compute_share_pct": row.get("GEMM Compute Share (%)", ""),
        "dma_read": row.get("DMA Read Bytes", ""),
        "dma_write": row.get("DMA Write Bytes", ""),
        "compute_cycles": row.get("totalComputeCycles", ""),
        "m5out_dir": row.get("m5out_dir", ""),
        "terminal_log": row.get("terminal_log", ""),
        "serial_log": row.get("serial_log", ""),
        "return_code": str(rc),
    }


def normalize_vit_row(row: dict[str, str]) -> dict[str, str]:
    return {
        "workload": "vit_proxy_hbm",
        "case_name": row.get("Preset", ""),
        "run_status": "ok"
        if row.get("Workload OK") in {"True", "true", "1"}
        else "invalid",
        "workload_ok": row.get("Workload OK", ""),
        "device_link_gbs": row.get("Device Link (GB/s)", ""),
        "matrix_size": "",
        "seq_len": row.get("SeqLen", ""),
        "phase1_ms": row.get("phase1_ms", ""),
        "phase2_total_ms": row.get("phase2_total_ms", ""),
        "phase2_non_gemm_ms": row.get("phase2_non_gemm_ms", ""),
        "end_to_end_ms": row.get("end_to_end_ms", ""),
        "gemm_active_time_ms": "",
        "gemm_effective_gflops": row.get("Effective GEMM Peak (GFLOPS)", ""),
        "timing_source": "",
        "metrics_source": "",
        "non_gemm_effective_gflops": "",
        "system_effective_gflops": row.get(
            "Effective System Throughput (GFLOPS)", ""
        ),
        "compute_share_pct": "",
        "dma_read": row.get("totalDmaRead", ""),
        "dma_write": row.get("totalDmaWrite", ""),
        "compute_cycles": row.get("totalComputeCycles", ""),
        "m5out_dir": row.get("m5out_dir", ""),
        "terminal_log": row.get("terminal_log", ""),
        "serial_log": row.get("serial_log", ""),
        "return_code": "0",
    }


def write_outputs(rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    headers = list(rows[0].keys())
    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "| Workload | Case | Status | GEMM GFLOPS | timing_source | metrics_source | phase1_ms | phase2_total_ms | system_effective_gflops | end_to_end_ms |",
        "|---|---|---|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {workload} | {case_name} | {run_status} | {gemm_effective_gflops} | {timing_source} | {metrics_source} | {phase1_ms} | {phase2_total_ms} | {system_effective_gflops} | {end_to_end_ms} |".format(
                **row
            )
        )
    RESULTS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    existing = load_existing_results() if args.resume else {}
    done = set(existing.keys())
    rows = list(existing.values())
    points = build_points(args.device_link_gbs, args.keep_existing)
    if args.limit is not None:
        points = points[: args.limit]

    print(f"points={len(points)} results_csv={RESULTS_CSV}")
    for idx, point in enumerate(points, start=1):
        key = (str(point["workload"]), str(point["case_name"]))
        if key in done:
            print(f"[-] skip resume {key}")
            continue

        cmd = [str(part) for part in point["cmd"]]
        print(
            f"\n[{idx}/{len(points)}] {point['workload']} {point['case_name']}"
        )
        print(" ".join(cmd))

        if args.dry_run:
            continue

        rc = subprocess.run(cmd, cwd=ROOT).returncode
        if point["workload"] == "vit_proxy_hbm_batch":
            summary_rows: list[dict[str, str]] = []
            if Path(point["summary"]).exists():
                with Path(point["summary"]).open(
                    newline="", encoding="utf-8"
                ) as fh:
                    summary_rows = list(csv.DictReader(fh))
            for row in summary_rows:
                norm = normalize_vit_row(row)
                vit_key = (norm["workload"], norm["case_name"])
                rows = [
                    r
                    for r in rows
                    if (r["workload"], r["case_name"]) != vit_key
                ]
                rows.append(norm)
                done.add(vit_key)
        else:
            runner_row = read_single_row(Path(point["summary"]))
            norm = normalize_pure_row(point, rc, runner_row)
            rows = [r for r in rows if (r["workload"], r["case_name"]) != key]
            rows.append(norm)
            done.add(key)
        write_outputs(rows)

    write_outputs(rows)
    print(f"\ncombined csv: {RESULTS_CSV}")
    print(f"markdown table: {RESULTS_MD}")


if __name__ == "__main__":
    main()
