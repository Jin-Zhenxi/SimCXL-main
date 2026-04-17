#!/usr/bin/env python3
"""Build the focused CXL 32G comparison table:
pure_gemm_2048 vs ViT Base/Large/Huge.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

GEM5_ROOT = Path("/home/jzx8091/SimCXL-main")
OUTPUT_DIR = GEM5_ROOT / "my_outputData"
PURE_CSV = OUTPUT_DIR / "pure_gemm_cxl_summary.csv"
VIT_CSV = OUTPUT_DIR / "cxl_vit_proxy_data.csv"
OUT_CSV = OUTPUT_DIR / "cxl_focus_compare.csv"
OUT_TXT = OUTPUT_DIR / "cxl_focus_compare.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine CXL pure_gemm_2048 and ViT proxy rows into one table."
    )
    parser.add_argument(
        "--device-link-gbs",
        "--device_link_gbs",
        dest="device_link_gbs",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--pure-matrix-size",
        "--pure_matrix_size",
        dest="pure_matrix_size",
        type=int,
        default=2048,
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def to_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "0") or 0)
    except ValueError:
        return 0.0


def to_int(row: dict[str, str], key: str) -> int:
    try:
        return int(float(row.get(key, "0") or 0))
    except ValueError:
        return 0


def main() -> None:
    args = parse_args()

    pure_rows = read_csv_rows(PURE_CSV)
    vit_rows = read_csv_rows(VIT_CSV)

    target_pure = None
    for row in pure_rows:
        if (
            to_int(row, "Device Link (GB/s)") == args.device_link_gbs
            and to_int(row, "Matrix Size") == args.pure_matrix_size
        ):
            target_pure = row
            break

    rows: list[dict[str, object]] = []

    if target_pure is not None:
        pure_gemm = to_float(target_pure, "GEMM-Effective (GFLOPS)")
        rows.append(
            {
                "Case": f"CXL {args.device_link_gbs}G pure_gemm_{args.pure_matrix_size}",
                "Workload": "pure_gemm",
                "GEMM Size": args.pure_matrix_size,
                "GEMM-Effective (GFLOPS)": round(pure_gemm, 6),
                "GEMM vs pure2048 (%)": 100.0,
                "phase2_total_ms": round(
                    to_float(target_pure, "phase2_total_ms"), 6
                ),
                "end_to_end_ms": round(
                    to_float(target_pure, "end_to_end_ms"), 6
                ),
                "Run Status": target_pure.get("Run Status", "unknown"),
                "Workload OK": target_pure.get("Workload OK", "False"),
            }
        )
    else:
        pure_gemm = 0.0

    ordered_presets = ["ViT-Base-like", "ViT-Large-like", "ViT-Huge-like"]
    vit_map = {
        row.get("Preset", ""): row
        for row in vit_rows
        if to_int(row, "Device Link (GB/s)") == args.device_link_gbs
    }

    for preset in ordered_presets:
        row = vit_map.get(preset)
        if row is None:
            continue
        gemm = to_float(row, "GEMM-Effective (GFLOPS)")
        rows.append(
            {
                "Case": f"CXL {args.device_link_gbs}G {preset}",
                "Workload": "vit_proxy",
                "GEMM Size": to_int(row, "GEMM Proxy Size"),
                "GEMM-Effective (GFLOPS)": round(gemm, 6),
                "GEMM vs pure2048 (%)": round(
                    (100.0 * gemm / pure_gemm) if pure_gemm > 0 else 0.0, 6
                ),
                "phase2_total_ms": round(to_float(row, "phase2_total_ms"), 6),
                "end_to_end_ms": round(to_float(row, "end_to_end_ms"), 6),
                "Run Status": row.get("Run Status", "unknown"),
                "Workload OK": row.get("Workload OK", "False"),
            }
        )

    if not rows:
        raise SystemExit(
            "未找到可汇总的数据。请先运行 pure_gemm_cxl_run.py 和 vit_proxy_cxl_run.py。"
        )

    headers = list(rows[0].keys())
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    with OUT_TXT.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(str(row) + "\n")

    print(f"[done] comparison table written to {OUT_CSV}")
    for row in rows:
        print(
            f"{row['Case']}: "
            f"GEMM={row['GEMM-Effective (GFLOPS)']} GFLOPS, "
            f"phase2={row['phase2_total_ms']} ms, "
            f"end_to_end={row['end_to_end_ms']} ms"
        )


if __name__ == "__main__":
    main()
