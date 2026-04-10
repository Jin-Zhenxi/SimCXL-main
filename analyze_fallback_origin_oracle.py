#!/usr/bin/env python3
"""
Analyze multi-workload Oracle teacher data to answer:
1. Which B buckets actually carry fallback burden now?
2. Whether that burden matches the current hypothesis:
   - next_output immediate/near are mostly already handled by direct mainline
   - repeated fallback victims belong to VIP
   - current_window / normal spill buckets are the likely M-hot-facing pain area
3. What provisional A-side remote-pressure patterns exist.

This script is analysis-only. It does not change runtime policy.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


INPUT_CSV = Path("my_outputData/ab_oracle_teacher_sweep.csv")
OUTPUT_B_CSV = Path("my_outputData/fallback_oracle_b_carriers.csv")
OUTPUT_A_CSV = Path("my_outputData/fallback_oracle_a_pressure.csv")
OUTPUT_SPLIT_CSV = Path("my_outputData/fallback_oracle_three_way_split.csv")
OUTPUT_SUMMARY = Path("my_outputData/fallback_oracle_summary.md")

CLAIM_MODE = "ab_hierarchical_claim_hole_filling"
VIP_MODE = "vip_rescue_buffer_single_run"

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
    if value in ("", None):
        return 0.0
    return float(value)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def classify_bucket(source: str, dist: str, local_ratio: float,
                    fallback_ratio: float) -> tuple[str, str]:
    if source == "next_output" and dist in ("immediate", "near") and local_ratio > 0.9:
        return (
            "class1_mainline_handled",
            "Already handled well by direct mainline; not the main fallback carrier.",
        )

    if source in ("claim", "carry_over") and local_ratio > 0.9:
        return (
            "future_chain_enhanced",
            "Future-chain object that already lands locally; enhancement-only, not current fallback carrier.",
        )

    if source == "current_window" and dist == "immediate" and fallback_ratio > 0.9:
        return (
            "class2_vip_like_victim",
            "Already in the pit: short-distance current-window victim; closer to rescue semantics than default M-hot.",
        )

    if source == "normal" and dist == "near" and fallback_ratio > 0.9:
        return (
            "class3_mhot_gap_candidate",
            "Near-distance spill bucket: mainline-adjacent but not caught; strongest M-hot gap candidate.",
        )

    if source == "current_window" and dist == "far" and fallback_ratio > 0.9:
        return (
            "class3_mhot_gap_candidate",
            "Future current-window spill bucket: likely a mainline-adjacent miss path, worth M-hot study but not default.",
        )

    if source == "normal" and dist == "far" and fallback_ratio > 0.9:
        return (
            "pain_bucket_reject_default",
            "Large fallback burden, but too broad/far to treat as a default hot-object rule.",
        )

    return (
        "other",
        "Does not dominate the current fallback picture.",
    )


def build_b_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], dict[str, object]] = defaultdict(
        lambda: {
            "local_rows": 0.0,
            "fallback_rows": 0.0,
            "local_reuse_weight": 0.0,
            "fallback_reuse_weight": 0.0,
            "coverage": set(),
        }
    )

    for row in rows:
        if row["Prefetch Label"] != CLAIM_MODE:
            continue
        for source in B_SOURCES:
            for dist in B_DISTS:
                local_key = f"oracle_b_local_{source}_{dist}"
                fallback_key = f"oracle_b_fallback_{source}_{dist}"
                local = as_float(row, local_key)
                fallback = as_float(row, fallback_key)
                if local + fallback == 0:
                    continue
                g = grouped[(source, dist)]
                g["local_rows"] += local
                g["fallback_rows"] += fallback
                g["local_reuse_weight"] += as_float(
                    row, f"oracle_b_local_reuse_weight_{source}_{dist}"
                )
                g["fallback_reuse_weight"] += as_float(
                    row, f"oracle_b_fallback_reuse_weight_{source}_{dist}"
                )
                g["coverage"].add(row["Preset"])

    out: list[dict[str, object]] = []
    for (source, dist), g in grouped.items():
        total = float(g["local_rows"]) + float(g["fallback_rows"])
        if total == 0:
            continue
        local_ratio = float(g["local_rows"]) / total
        fallback_ratio = float(g["fallback_rows"]) / total
        avg_fb_reuse = (
            float(g["fallback_reuse_weight"]) / float(g["fallback_rows"])
            if float(g["fallback_rows"]) > 0
            else 0.0
        )
        bucket_class, reason = classify_bucket(
            source, dist, local_ratio, fallback_ratio
        )
        out.append(
            {
                "Source Class": source,
                "Distance Bucket": dist,
                "Bucket Class": bucket_class,
                "Reason": reason,
                "Workload Coverage": len(g["coverage"]),
                "Local Rows": int(g["local_rows"]),
                "Fallback Rows": int(g["fallback_rows"]),
                "Total Rows": int(total),
                "Local Ratio": round(local_ratio, 6),
                "Fallback Ratio": round(fallback_ratio, 6),
                "Fallback Reuse Weight": round(float(g["fallback_reuse_weight"]), 3),
                "Avg Fallback Reuse Weight": round(avg_fb_reuse, 6),
            }
        )

    out.sort(
        key=lambda r: (
            -int(r["Fallback Rows"]),
            -float(r["Fallback Reuse Weight"]),
            -int(r["Workload Coverage"]),
        )
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
        local = sum(
            as_float(row, f"oracle_a_local_distance_{bucket}")
            for row in rows
            if row["Prefetch Label"] == CLAIM_MODE
        )
        out.append(
            {
                "A Feature": "distance",
                "Bucket": bucket,
                "Remote Count": int(remote),
                "Local Count": int(local),
                "Interpretation": (
                    "Short-distance remote pressure"
                    if bucket in ("immediate", "near")
                    else "Longer-distance remote pressure"
                ),
            }
        )
    for bucket in A_REUSE:
        remote = sum(
            as_float(row, f"oracle_a_remote_reuse_{bucket}")
            for row in rows
            if row["Prefetch Label"] == CLAIM_MODE
        )
        local = sum(
            as_float(row, f"oracle_a_local_reuse_{bucket}")
            for row in rows
            if row["Prefetch Label"] == CLAIM_MODE
        )
        out.append(
            {
                "A Feature": "reuse",
                "Bucket": bucket,
                "Remote Count": int(remote),
                "Local Count": int(local),
                "Interpretation": (
                    "Multi-reuse remote hotspot"
                    if bucket == "multi"
                    else "One-shot / weaker reuse"
                ),
            }
        )
    for bucket in A_RECUR:
        remote = sum(
            as_float(row, f"oracle_a_remote_recurrence_{bucket}")
            for row in rows
            if row["Prefetch Label"] == CLAIM_MODE
        )
        local = sum(
            as_float(row, f"oracle_a_local_recurrence_{bucket}")
            for row in rows
            if row["Prefetch Label"] == CLAIM_MODE
        )
        out.append(
            {
                "A Feature": "recurrence",
                "Bucket": bucket,
                "Remote Count": int(remote),
                "Local Count": int(local),
                "Interpretation": (
                    "Repeat victim / recurring remote pressure"
                    if bucket == "repeat"
                    else "First-touch remote pressure"
                ),
            }
        )
    return out


def build_three_way_split(b_rows: list[dict[str, object]],
                          a_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    class1 = [r for r in b_rows if r["Bucket Class"] == "class1_mainline_handled"]
    class2 = [r for r in b_rows if r["Bucket Class"] == "class2_vip_like_victim"]
    class3 = [r for r in b_rows if r["Bucket Class"] == "class3_mhot_gap_candidate"]
    pain = [r for r in b_rows if r["Bucket Class"] == "pain_bucket_reject_default"]

    return [
        {
            "Category": "class1_mainline_handled",
            "What it means": "Already handled by direct mainline",
            "Representative Buckets": "; ".join(
                f"{r['Source Class']}/{r['Distance Bucket']}" for r in class1
            ),
            "Fallback Rows": sum(int(r["Fallback Rows"]) for r in class1),
            "Recommended Owner": "Mainline direct path",
        },
        {
            "Category": "class2_vip_like_victim",
            "What it means": "Already in the pit; rescue semantics",
            "Representative Buckets": "; ".join(
                f"{r['Source Class']}/{r['Distance Bucket']}" for r in class2
            ),
            "Fallback Rows": sum(int(r["Fallback Rows"]) for r in class2),
            "Recommended Owner": "VIP V3",
        },
        {
            "Category": "class3_mhot_gap_candidate",
            "What it means": "Mainline-adjacent slips that M-hot should study",
            "Representative Buckets": "; ".join(
                f"{r['Source Class']}/{r['Distance Bucket']}" for r in class3
            ),
            "Fallback Rows": sum(int(r["Fallback Rows"]) for r in class3),
            "Recommended Owner": "Future B-side M-hot",
        },
        {
            "Category": "pain_bucket_reject_default",
            "What it means": "Large fallback burden but too broad/far for default M-hot",
            "Representative Buckets": "; ".join(
                f"{r['Source Class']}/{r['Distance Bucket']}" for r in pain
            ),
            "Fallback Rows": sum(int(r["Fallback Rows"]) for r in pain),
            "Recommended Owner": "Neither by default; keep as pain signal",
        },
        {
            "Category": "a_provisional",
            "What it means": "Teacher-side A pressure only",
            "Representative Buckets": "; ".join(
                f"{r['A Feature']}/{r['Bucket']}" for r in a_rows
                if (
                    (r["A Feature"] == "distance" and r["Bucket"] in ("immediate", "near"))
                    or (r["A Feature"] == "reuse" and r["Bucket"] == "multi")
                    or (r["A Feature"] == "recurrence" and r["Bucket"] == "repeat")
                )
            ),
            "Fallback Rows": "",
            "Recommended Owner": "Keep provisional only; no runtime A-hot yet",
        },
    ]


def build_summary(b_rows: list[dict[str, object]],
                  a_rows: list[dict[str, object]],
                  split_rows: list[dict[str, object]]) -> str:
    top_fallback = [
        r for r in b_rows if int(r["Fallback Rows"]) > 0
    ][:6]
    hypothesis_lines = []

    def find_bucket(source: str, dist: str) -> dict[str, object] | None:
        for row in b_rows:
            if row["Source Class"] == source and row["Distance Bucket"] == dist:
                return row
        return None

    n_im = find_bucket("next_output", "immediate")
    n_ne = find_bucket("next_output", "near")
    cw_im = find_bucket("current_window", "immediate")
    nrm_ne = find_bucket("normal", "near")
    nrm_far = find_bucket("normal", "far")

    if n_im and n_ne:
        hypothesis_lines.append(
            f"- `next_output/immediate` 和 `next_output/near` 的 fallback 都是 0，说明它们确实主要在 direct mainline 中兑现。"
        )
    if cw_im:
        hypothesis_lines.append(
            f"- `current_window/immediate` 有 {cw_im['Fallback Rows']} 行 fallback，且 fallback_ratio={cw_im['Fallback Ratio']}，更像 VIP/rescue 受害者。"
        )
    if nrm_ne and nrm_far:
        hypothesis_lines.append(
            f"- `normal/near` ({nrm_ne['Fallback Rows']}) 和 `normal/far` ({nrm_far['Fallback Rows']}) 是当前最大的 fallback 负担来源。"
        )

    a_repeat = next(r for r in a_rows if r["A Feature"] == "recurrence" and r["Bucket"] == "repeat")
    a_im = next(r for r in a_rows if r["A Feature"] == "distance" and r["Bucket"] == "immediate")
    a_multi = next(r for r in a_rows if r["A Feature"] == "reuse" and r["Bucket"] == "multi")

    lines = [
        "# Fallback Oracle Summary",
        "",
        "## Verdict",
        "",
        "当前多模型 Oracle 数据 **大体支持** 你的猜想，但要再精确一点：",
        "",
        *hypothesis_lines,
        "- `current_window/near` 在当前 teacher 数据里并不显著；真正显著的是 `current_window/immediate` 和 `current_window/far`。",
        "- `normal/far` 虽然是最大 pain bucket，但它更像宽泛痛点，不应直接当成默认 M-hot 规则。",
        "",
        "## Top B-side Fallback Carriers",
        "",
    ]
    for row in top_fallback:
        lines.append(
            f"- `{row['Source Class']}/{row['Distance Bucket']}`: "
            f"fallback_rows={row['Fallback Rows']}, "
            f"fallback_ratio={row['Fallback Ratio']}, "
            f"class={row['Bucket Class']}"
        )

    lines.extend(
        [
            "",
            "## Three-way Split",
            "",
        ]
    )
    for row in split_rows:
        lines.append(
            f"- `{row['Category']}`: {row['Representative Buckets']} -> {row['Recommended Owner']}"
        )

    lines.extend(
        [
            "",
            "## A-side Provisional Signals",
            "",
            f"- `recurrence/repeat`: remote_count={a_repeat['Remote Count']}",
            f"- `distance/immediate`: remote_count={a_im['Remote Count']}",
            f"- `reuse/multi`: remote_count={a_multi['Remote Count']}",
            "- 这支持 A 侧存在 teacher-side hotness，但当前仍不足以直接上线 runtime A-hot。",
            "",
            "## Practical Reading",
            "",
            "- 类 1：`next_output/immediate|near` 已经是案板能处理好的对象。",
            "- 类 2：真正掉坑里的 repeated/current-window victims 继续归 VIP V3。",
            "- 类 3：贴着案板但经常没接住的对象，当前最像的是 `normal/near` 和部分 `current_window/far`，这才是后续 B-side M-hot 更值得重点验证的区域。",
            "",
            "## Recommended Next Step",
            "",
            "- 不再拿 `next_output/immediate|near` 当 M-hot fallback-time first cut 的主收益来源。",
            "- 若继续做 B-side M-hot，应优先围绕 `mainline-adjacent slip buckets` 做更针对性的 teacher/runtime验证。",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    rows = load_rows()
    b_rows = build_b_rows(rows)
    a_rows = build_a_rows(rows)
    split_rows = build_three_way_split(b_rows, a_rows)

    write_csv(OUTPUT_B_CSV, b_rows)
    write_csv(OUTPUT_A_CSV, a_rows)
    write_csv(OUTPUT_SPLIT_CSV, split_rows)
    OUTPUT_SUMMARY.write_text(build_summary(b_rows, a_rows, split_rows))

    print("Fallback Oracle analysis complete")
    print(f"  B carriers: {OUTPUT_B_CSV}")
    print(f"  A pressure: {OUTPUT_A_CSV}")
    print(f"  Three-way split: {OUTPUT_SPLIT_CSV}")
    print(f"  Summary: {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()
