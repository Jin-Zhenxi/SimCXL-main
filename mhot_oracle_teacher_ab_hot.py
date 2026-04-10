#!/usr/bin/env python3
"""
M-hot Oracle teacher sweep wrapper.

This script does not change runtime policy.
It prepares a dedicated Oracle-teacher CSV for M-hot analysis while keeping:
- mainline direct path semantics unchanged
- VIP V3 unchanged
- M-hot as a teacher-side hot-object study only

If the upstream A/B teacher sweep is missing or incomplete, this wrapper
invokes `oracle_ab_teacher_sweep.py` to regenerate the source data.
"""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path

import oracle_ab_teacher_sweep as ab


OUTPUT_DIR = Path("my_outputData")
SRC_CSV = OUTPUT_DIR / "ab_oracle_teacher_sweep.csv"
SRC_RAW = OUTPUT_DIR / "ab_oracle_teacher_sweep_raw.txt"
DST_CSV = OUTPUT_DIR / "mhot_oracle_teacher_ab_hot.csv"
DST_RAW = OUTPUT_DIR / "mhot_oracle_teacher_ab_hot_raw.txt"

EXPECTED_CASES = {
    (preset["name"], label)
    for preset in ab.ANALYSIS_PRESETS
    for label in ab.MODE_LABELS
}


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def completed_keys(rows: list[dict[str, str]]) -> set[tuple[str, str]]:
    return {
        (row["Preset"], row["Prefetch Label"])
        for row in rows
        if row.get("Preset") and row.get("Prefetch Label")
    }


def source_ready() -> bool:
    rows = load_rows(SRC_CSV)
    return completed_keys(rows) >= EXPECTED_CASES


def ensure_source_teacher_data() -> None:
    ab.ensure_gem5_binary()
    if source_ready():
        print(f"[*] 复用已有 A/B teacher 数据: {SRC_CSV}")
        return
    print("[*] 上游 A/B teacher 数据缺失或不完整，正在补跑 oracle_ab_teacher_sweep.py ...")
    subprocess.run("python3 oracle_ab_teacher_sweep.py", shell=True, check=True)


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    headers: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                headers.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    ensure_source_teacher_data()

    rows = load_rows(SRC_CSV)
    rows.sort(key=lambda r: (r["Preset"], r["Prefetch Label"]))
    write_rows(DST_CSV, rows)

    with DST_RAW.open("w") as f:
        for row in rows:
            f.write(str(row) + "\n")

    print("🚀 M-hot Oracle teacher input prepared")
    print(f"   source: {SRC_CSV}")
    print(f"   output: {DST_CSV}")
    print(f"   cases: {len(rows)}")
    print(
        "   note: this is teacher-side data only; runtime M-hot admission is intentionally not enabled here"
    )


if __name__ == "__main__":
    main()
