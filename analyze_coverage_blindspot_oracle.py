#!/usr/bin/env python3
"""
Analyze multi-workload Oracle teacher data to isolate B-side coverage blind-spot
patterns and separate them from:
1. direct-mainline next_output objects,
2. VIP-like rescue victims,
3. gap-aware M-hot slip buckets.

This is analysis-only. It does not change runtime policy.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


INPUT_CSV = Path("my_outputData/ab_oracle_teacher_sweep.csv")
PATTERN_CSV = Path("my_outputData/coverage_blindspot_oracle_pattern_rules.csv")
WORKLOAD_CSV = Path("my_outputData/coverage_blindspot_oracle_workload_summary.csv")
A_PROVISIONAL_CSV = Path("my_outputData/coverage_blindspot_oracle_a_provisional.csv")
BOUNDARY_CSV = Path("my_outputData/coverage_blindspot_oracle_boundary.csv")
SUMMARY_MD = Path("my_outputData/coverage_blindspot_oracle_summary.md")

CLAIM_MODE = "ab_hierarchical_claim_hole_filling"
PRESETS = [
    "Kernel-S128",
    "Kernel-S192",
    "Kernel-S256",
    "Kernel-S320",
    "ViT-Base-like",
    "ViT-Large-like",
]
B_SOURCES = ["next_output", "claim", "carry_over", "current_window", "normal"]
B_DISTS = ["immediate", "near", "far"]
A_DISTS = ["immediate", "near", "far"]
A_REUSE = ["one", "multi"]
A_RECUR = ["first", "repeat"]


def load_rows() -> list[dict[str, str]]:
    if not INPUT_CSV.exists():
        raise SystemExit(f"未找到 Oracle teacher 数据: {INPUT_CSV}")
    with INPUT_CSV.open(newline="") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, 0)
    return float(value) if value not in ("", None) else 0.0


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def classify_pattern(source: str, dist: str, fallback_ratio: float,
                     workload_coverage: int, avg_fb_reuse: float) -> tuple[str, str]:
    if source == "next_output" and dist in {"immediate", "near"} and fallback_ratio == 0.0:
        return (
            "mainline_owned",
            "Already handled by direct mainline; not a coverage-blindspot carrier.",
        )

    if source == "current_window" and dist == "immediate" and fallback_ratio >= 0.95:
        return (
            "vip_boundary",
            "Current-window immediate victim; better owned by VIP rescue semantics.",
        )

    if source == "normal" and dist == "near" and fallback_ratio >= 0.95:
        return (
            "gap_hot_secondary",
            "Near-distance Normal spill bucket. Important, but already better aligned with gap-aware M-hot than blind-spot promotion.",
        )

    if (
        source == "normal"
        and dist == "far"
        and fallback_ratio >= 0.95
        and workload_coverage >= 4
        and avg_fb_reuse >= 80.0
    ):
        return (
            "coverage_blindspot_primary",
            "Primary cross-workload coverage blind-spot: Normal + far, broad workload coverage, and meaningful reuse weight.",
        )

    if (
        source == "normal"
        and dist == "near"
        and fallback_ratio >= 0.95
        and workload_coverage >= 4
        and avg_fb_reuse >= 80.0
    ):
        return (
            "coverage_blindspot_secondary",
            "Secondary blind-spot family: Normal + near. High-value but overlaps with existing gap-aware handling.",
        )

    if source == "current_window" and dist == "far" and fallback_ratio >= 0.95:
        return (
            "slip_bucket",
            "Current-window far slip bucket. Mainline-adjacent miss path, but not the dominant Normal blind-spot family.",
        )

    return (
        "other",
        "Does not define the cross-workload coverage-blindspot picture.",
    )


def build_pattern_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], dict[str, object]] = defaultdict(
        lambda: {
            "local_rows": 0.0,
            "fallback_rows": 0.0,
            "fallback_reuse": 0.0,
            "workloads": set(),
        }
    )

    for row in rows:
        if row["Prefetch Label"] != CLAIM_MODE:
            continue
        for source in B_SOURCES:
            for dist in B_DISTS:
                local = as_float(row, f"oracle_b_local_{source}_{dist}")
                fallback = as_float(row, f"oracle_b_fallback_{source}_{dist}")
                if local + fallback == 0:
                    continue
                g = grouped[(source, dist)]
                g["local_rows"] += local
                g["fallback_rows"] += fallback
                g["fallback_reuse"] += as_float(
                    row, f"oracle_b_fallback_reuse_weight_{source}_{dist}"
                )
                g["workloads"].add(row["Preset"])

    out: list[dict[str, object]] = []
    for (source, dist), g in grouped.items():
        total = float(g["local_rows"]) + float(g["fallback_rows"])
        if total == 0:
            continue
        fallback_ratio = float(g["fallback_rows"]) / total
        avg_fb_reuse = (
            float(g["fallback_reuse"]) / float(g["fallback_rows"])
            if float(g["fallback_rows"]) > 0
            else 0.0
        )
        pattern_class, note = classify_pattern(
            source, dist, fallback_ratio, len(g["workloads"]), avg_fb_reuse
        )
        out.append(
            {
                "Source Class": source,
                "Distance Bucket": dist,
                "Pattern Class": pattern_class,
                "Workload Coverage": len(g["workloads"]),
                "Local Rows": int(g["local_rows"]),
                "Fallback Rows": int(g["fallback_rows"]),
                "Fallback Ratio": round(fallback_ratio, 6),
                "Fallback Reuse Weight": round(float(g["fallback_reuse"]), 3),
                "Avg Fallback Reuse Weight": round(avg_fb_reuse, 6),
                "Recommended Note": note,
            }
        )

    order = {
        "coverage_blindspot_primary": 0,
        "coverage_blindspot_secondary": 1,
        "gap_hot_secondary": 2,
        "vip_boundary": 3,
        "slip_bucket": 4,
        "mainline_owned": 5,
        "other": 6,
    }
    out.sort(key=lambda r: (order[r["Pattern Class"]], -int(r["Fallback Rows"])))
    return out


def build_workload_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    by_key = {(r["Preset"], r["Prefetch Label"]): r for r in rows}
    for preset in PRESETS:
        row = by_key.get((preset, CLAIM_MODE))
        if not row:
            continue
        normal_near = int(as_float(row, "oracle_b_fallback_normal_near"))
        normal_far = int(as_float(row, "oracle_b_fallback_normal_far"))
        cw_im = int(as_float(row, "oracle_b_fallback_current_window_immediate"))
        cw_far = int(as_float(row, "oracle_b_fallback_current_window_far"))
        next_im = int(as_float(row, "oracle_b_fallback_next_output_immediate"))
        next_near = int(as_float(row, "oracle_b_fallback_next_output_near"))
        total = int(as_float(row, "fallback_b_rows_fetched"))
        blindspot = normal_far
        gap_like = normal_near + cw_far
        vip_like = cw_im
        out.append(
            {
                "Preset": preset,
                "Total Fallback": total,
                "Normal Near": normal_near,
                "Normal Far": normal_far,
                "CurrentWindow Immediate": cw_im,
                "CurrentWindow Far": cw_far,
                "next_output Immediate": next_im,
                "next_output Near": next_near,
                "Blindspot Share (Normal Far)": round(
                    blindspot / total if total else 0.0, 6
                ),
                "Gap-like Share (Normal Near + CW Far)": round(
                    gap_like / total if total else 0.0, 6
                ),
                "VIP-like Share (CW Immediate)": round(
                    vip_like / total if total else 0.0, 6
                ),
                "Mainline next_output Share": round(
                    (next_im + next_near) / total if total else 0.0, 6
                ),
            }
        )
    return out


def build_a_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for bucket in A_DISTS:
        remote = sum(
            as_float(row, f"oracle_a_remote_distance_{bucket}")
            for row in rows
            if row["Prefetch Label"] == CLAIM_MODE
        )
        out.append(
            {
                "A Feature": "distance",
                "Bucket": bucket,
                "Remote Count": int(remote),
                "Status": (
                    "provisional_hot"
                    if bucket in {"immediate", "near"}
                    else "weak"
                ),
            }
        )
    for bucket in A_REUSE:
        remote = sum(
            as_float(row, f"oracle_a_remote_reuse_{bucket}")
            for row in rows
            if row["Prefetch Label"] == CLAIM_MODE
        )
        out.append(
            {
                "A Feature": "reuse",
                "Bucket": bucket,
                "Remote Count": int(remote),
                "Status": "provisional_hot" if bucket == "multi" else "weak",
            }
        )
    for bucket in A_RECUR:
        remote = sum(
            as_float(row, f"oracle_a_remote_recurrence_{bucket}")
            for row in rows
            if row["Prefetch Label"] == CLAIM_MODE
        )
        out.append(
            {
                "A Feature": "recurrence",
                "Bucket": bucket,
                "Remote Count": int(remote),
                "Status": "provisional_hot" if bucket == "repeat" else "weak",
            }
        )
    out.sort(key=lambda r: (-int(r["Remote Count"]), r["A Feature"], r["Bucket"]))
    return out


def build_boundary_rows(pattern_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    class_totals = defaultdict(int)
    for row in pattern_rows:
        class_totals[row["Pattern Class"]] += int(row["Fallback Rows"])

    return [
        {
            "Category": "mainline_owned",
            "Owner": "Direct mainline",
            "Fallback Rows": class_totals["mainline_owned"],
            "Representative": "next_output/immediate; next_output/near",
        },
        {
            "Category": "vip_boundary",
            "Owner": "VIP V3",
            "Fallback Rows": class_totals["vip_boundary"],
            "Representative": "current_window/immediate",
        },
        {
            "Category": "gap_hot_secondary",
            "Owner": "gap-aware B-side M-hot",
            "Fallback Rows": class_totals["gap_hot_secondary"] + class_totals["slip_bucket"],
            "Representative": "normal/near; current_window/far",
        },
        {
            "Category": "coverage_blindspot_family",
            "Owner": "future coverage-aware B-side M-hot slice",
            "Fallback Rows": class_totals["coverage_blindspot_primary"] + class_totals["coverage_blindspot_secondary"],
            "Representative": "normal/far; normal/near (secondary)",
        },
    ]


def build_summary(pattern_rows: list[dict[str, object]],
                  workload_rows: list[dict[str, object]],
                  a_rows: list[dict[str, object]],
                  boundary_rows: list[dict[str, object]]) -> str:
    def find(source: str, dist: str) -> dict[str, object]:
        for row in pattern_rows:
            if row["Source Class"] == source and row["Distance Bucket"] == dist:
                return row
        raise KeyError((source, dist))

    normal_far = find("normal", "far")
    normal_near = find("normal", "near")
    cw_im = find("current_window", "immediate")
    cw_far = find("current_window", "far")
    next_im = find("next_output", "immediate")
    next_near = find("next_output", "near")

    a_repeat = next(
        row for row in a_rows
        if row["A Feature"] == "recurrence" and row["Bucket"] == "repeat"
    )
    a_im = next(
        row for row in a_rows
        if row["A Feature"] == "distance" and row["Bucket"] == "immediate"
    )
    a_multi = next(
        row for row in a_rows
        if row["A Feature"] == "reuse" and row["Bucket"] == "multi"
    )

    lines = [
        "# Coverage Blind-Spot Oracle Summary",
        "",
        "## Verdict",
        "",
        "- 多模型 teacher 数据支持：`Normal` 才是 coverage blind-spot 的主体，不是 `next_output`。",
        f"- `normal/far` 是最稳的 blind-spot 主家族：fallback_rows={normal_far['Fallback Rows']}, workload_coverage={normal_far['Workload Coverage']}, avg_fallback_reuse={normal_far['Avg Fallback Reuse Weight']}.",
        f"- `normal/near` 是第二家族：fallback_rows={normal_near['Fallback Rows']}, workload_coverage={normal_near['Workload Coverage']}。它有价值，但更接近 gap-aware M-hot，而不是 blind-spot primary。",
        f"- `current_window/immediate` 继续更像 VIP 边界：fallback_rows={cw_im['Fallback Rows']}.",
        f"- `current_window/far` 是 slip bucket，不是 blind-spot 主家族：fallback_rows={cw_far['Fallback Rows']}.",
        f"- `next_output/immediate|near` fallback 都是 0（{next_im['Fallback Rows']} / {next_near['Fallback Rows']}），说明它们的价值已经主要在 direct mainline 兑现。",
        "",
        "## Why Row-Level Repeat Was Too Strict",
        "",
        "- 这类 blind-spot 的稳定性更像 `pattern/class recurrence`，而不是 `same rowAddr recurrence`。",
        "- 更可靠的普适规则应该围绕：`source=Normal`、`distance bucket`、`future reuse richness`、`workload coverage`，而不是等同一 row 第二次出现。",
        "",
        "## Recommended B-side Blind-Spot Rule",
        "",
        "- Primary: `Normal + far + high reuse richness`",
        "- Secondary: `Normal + near + high reuse richness`",
        "- Keep isolated from VIP V3 and from next_output mainline hot objects.",
        "- Runtime should promote by pattern/class recurrence, not exact row recurrence.",
        "",
        "## Boundary",
        "",
    ]
    for row in boundary_rows:
        lines.append(
            f"- `{row['Category']}` -> {row['Owner']} ({row['Representative']}, fallback_rows={row['Fallback Rows']})"
        )

    lines.extend(
        [
            "",
            "## Per-Workload Readout",
            "",
        ]
    )
    for row in workload_rows:
        lines.append(
            f"- `{row['Preset']}`: total_fallback={row['Total Fallback']}, "
            f"normal_far={row['Normal Far']}, normal_near={row['Normal Near']}, "
            f"cw_immediate={row['CurrentWindow Immediate']}, cw_far={row['CurrentWindow Far']}"
        )

    lines.extend(
        [
            "",
            "## A-side Provisional Status",
            "",
            f"- `repeat`: remote_count={a_repeat['Remote Count']}",
            f"- `immediate distance`: remote_count={a_im['Remote Count']}",
            f"- `multi reuse`: remote_count={a_multi['Remote Count']}",
            "- A-hot 仍建议保持 teacher-side provisional，不在这一轮上线 runtime。",
            "",
            "## Practical Next Step",
            "",
            "- 不要再把 blind-spot 规则写成“同一 row 第二次出现才收”。",
            "- 如果后面继续做 coverage-aware M-hot，优先围绕 `normal/far` 主家族设计 pattern-based admission。",
            "- `normal/near` 继续由 gap-aware M-hot 承接；`current_window/immediate` 继续留给 VIP V3。",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    rows = load_rows()
    pattern_rows = build_pattern_rows(rows)
    workload_rows = build_workload_rows(rows)
    a_rows = build_a_rows(rows)
    boundary_rows = build_boundary_rows(pattern_rows)

    write_csv(PATTERN_CSV, pattern_rows)
    write_csv(WORKLOAD_CSV, workload_rows)
    write_csv(A_PROVISIONAL_CSV, a_rows)
    write_csv(BOUNDARY_CSV, boundary_rows)
    SUMMARY_MD.write_text(
        build_summary(pattern_rows, workload_rows, a_rows, boundary_rows)
    )

    print("Coverage blind-spot Oracle analysis complete")
    print(f"  pattern rules: {PATTERN_CSV}")
    print(f"  workload summary: {WORKLOAD_CSV}")
    print(f"  A provisional: {A_PROVISIONAL_CSV}")
    print(f"  boundary: {BOUNDARY_CSV}")
    print(f"  summary: {SUMMARY_MD}")


if __name__ == "__main__":
    main()
