#!/usr/bin/env python3
"""
Analyze M-hot Oracle teacher data.

This analysis keeps the intended ordering explicit:
1. Try direct mainline first
2. Only when an object would otherwise go remote/fallback, consult nearby pools by tag
   - VIP for rescue-victim labels
   - M-hot for hot-object labels
3. If no nearby hit exists, perform the remote fetch

The goal is not to deploy M-hot yet.
The goal is to identify which A/B objects should eventually be tagged as
M-hot candidates, while keeping VIP and M-hot clearly separated.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


INPUT_CSV = Path("my_outputData/mhot_oracle_teacher_ab_hot.csv")
WORKLOAD_SUMMARY_CSV = Path("my_outputData/mhot_oracle_workload_summary.csv")
B_HOT_RULES_CSV = Path("my_outputData/mhot_oracle_b_hot_rules.csv")
A_HOT_RULES_CSV = Path("my_outputData/mhot_oracle_a_hot_rules.csv")
BOUNDARY_STATS_CSV = Path("my_outputData/mhot_oracle_boundary_stats.csv")
PATH_RULES_CSV = Path("my_outputData/mhot_oracle_path_rules.csv")
SUMMARY_MD = Path("my_outputData/mhot_oracle_summary.md")

BASELINE_MODE = "baseline_serial"
CLAIM_MODE = "ab_hierarchical_claim_hole_filling"
VIP_V3_MODE = "vip_rescue_buffer_single_run"

B_SOURCES = ["next_output", "claim", "carry_over", "current_window", "normal"]
B_DISTANCES = ["immediate", "near", "far"]
A_DISTANCE_BUCKETS = ["immediate", "near", "far"]
A_REUSE_BUCKETS = ["one", "multi"]
A_RECURRENCE_BUCKETS = ["first", "repeat"]

SOURCE_BIAS = {
    "next_output": 1.0,
    "claim": 0.8,
    "carry_over": 0.75,
    "current_window": 0.3,
    "normal": 0.1,
}
DIST_BIAS = {"immediate": 1.0, "near": 0.75, "far": 0.35}


def load_rows() -> list[dict[str, str]]:
    if not INPUT_CSV.exists():
        raise SystemExit(f"未找到 M-hot Oracle teacher 输入: {INPUT_CSV}")
    with INPUT_CSV.open(newline="") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, 0)
    return float(value) if value not in ("", None) else 0.0


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def workload_summary(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    by_key = {(r["Preset"], r["Prefetch Label"]): r for r in rows}
    out = []
    for preset in sorted({r["Preset"] for r in rows}):
        base = by_key.get((preset, BASELINE_MODE))
        claim = by_key.get((preset, CLAIM_MODE))
        vip = by_key.get((preset, VIP_V3_MODE))
        if not base or not claim:
            continue
        item = {
            "Preset": preset,
            "Base GFLOPS": round(as_float(base, "GEMM-Effective (GFLOPS)"), 6),
            "Claim GFLOPS": round(as_float(claim, "GEMM-Effective (GFLOPS)"), 6),
            "Claim Delta GFLOPS": round(
                as_float(claim, "GEMM-Effective (GFLOPS)")
                - as_float(base, "GEMM-Effective (GFLOPS)"),
                6,
            ),
            "Claim Fallback": int(as_float(claim, "fallback_b_rows_fetched")),
            "Claim PrefetchedB": int(as_float(claim, "prefetched_b_rows_consumed")),
            "Claim next_output_ready": int(
                as_float(claim, "next_output_rows_ready_at_boundary")
            ),
            "Claim future_claims": int(as_float(claim, "future_claim_set_count")),
        }
        if vip:
            item.update(
                {
                    "VIP V3 GFLOPS": round(
                        as_float(vip, "GEMM-Effective (GFLOPS)"), 6
                    ),
                    "VIP V3 Delta GFLOPS": round(
                        as_float(vip, "GEMM-Effective (GFLOPS)")
                        - as_float(claim, "GEMM-Effective (GFLOPS)"),
                        6,
                    ),
                    "VIP V3 Fallback": int(as_float(vip, "fallback_b_rows_fetched")),
                    "VIP V3 Rescue Hits": int(as_float(vip, "vip_rescue_hit_count")),
                }
            )
        out.append(item)
    return out


def classify_b_bucket(source: str, dist: str) -> tuple[str, str]:
    if source == "next_output" and dist in {"immediate", "near"}:
        return "default", "Stable mainline B-hot default"
    if source == "next_output" and dist == "far":
        return "enhanced", "Large-workload future-span enhancement"
    if source in {"claim", "carry_over"}:
        return "enhanced", "Future-chain enhancement, not default"
    if source == "current_window" and dist == "immediate":
        return "pain", "Pain bucket / emergency signal, not default M-hot"
    return "reject", "Not a default M-hot object"


def build_b_hot_rules(rows: list[dict[str, str]]) -> tuple[list[dict[str, object]], dict[str, float]]:
    grouped = defaultdict(
        lambda: {
            "local_rows": 0.0,
            "fallback_rows": 0.0,
            "local_reuse": 0.0,
            "fallback_reuse": 0.0,
            "coverage": set(),
        }
    )

    for row in rows:
        if row["Prefetch Label"] != CLAIM_MODE:
            continue
        for source in B_SOURCES:
            for dist in B_DISTANCES:
                key = (source, dist)
                g = grouped[key]
                g["local_rows"] += as_float(row, f"oracle_b_local_{source}_{dist}")
                g["fallback_rows"] += as_float(row, f"oracle_b_fallback_{source}_{dist}")
                g["local_reuse"] += as_float(
                    row, f"oracle_b_local_reuse_weight_{source}_{dist}"
                )
                g["fallback_reuse"] += as_float(
                    row, f"oracle_b_fallback_reuse_weight_{source}_{dist}"
                )
                if (
                    as_float(row, f"oracle_b_local_{source}_{dist}")
                    + as_float(row, f"oracle_b_fallback_{source}_{dist}")
                    > 0
                ):
                    g["coverage"].add(row["Preset"])

    rule_rows: list[dict[str, object]] = []
    summary = defaultdict(float)
    total_observed = 0.0
    mhot_like = 0.0
    vip_like = 0.0
    overlap = 0.0

    for source in B_SOURCES:
        for dist in B_DISTANCES:
            g = grouped[(source, dist)]
            total = g["local_rows"] + g["fallback_rows"]
            if total == 0:
                continue
            total_observed += total
            local_ratio = g["local_rows"] / total
            fallback_ratio = g["fallback_rows"] / total
            avg_reuse = (g["local_reuse"] + g["fallback_reuse"]) / total
            tier, reason = classify_b_bucket(source, dist)
            score = (
                SOURCE_BIAS[source] * 4.0
                + DIST_BIAS[dist] * 3.0
                + local_ratio * 2.0
                + len(g["coverage"]) / 6.0
                + min(avg_reuse / 256.0, 1.5)
            )

            if tier in {"default", "enhanced"}:
                mhot_like += total
                summary["b_hot_candidate_count"] += total
                summary[f"b_hot_{source}_count"] += total
                summary[f"b_hot_distance_{dist}_count"] += total
                summary[f"b_hot_{source}_{dist}_count"] += total
            if source in {"normal", "current_window"} and fallback_ratio >= 0.5:
                vip_like += total
                if tier in {"default", "enhanced"}:
                    overlap += total

            rule_rows.append(
                {
                    "Source Class": source,
                    "Distance Bucket": dist,
                    "Recommended Tier": tier,
                    "Reason": reason,
                    "Workload Coverage": len(g["coverage"]),
                    "Total Rows": round(total, 3),
                    "Local Rows": round(g["local_rows"], 3),
                    "Fallback Rows": round(g["fallback_rows"], 3),
                    "Local Ratio": round(local_ratio, 6),
                    "Fallback Ratio": round(fallback_ratio, 6),
                    "Avg Reuse Weight": round(avg_reuse, 6),
                    "Weighted M-hot Teacher Score": round(score, 6),
                }
            )

    summary["vip_like_victim_count"] = vip_like
    summary["mhot_like_hot_count"] = mhot_like
    summary["overlap_vip_mhot_candidate_count"] = overlap
    summary["neither_vip_nor_mhot_count"] = max(
        total_observed - vip_like - mhot_like + overlap, 0.0
    )
    rule_rows.sort(
        key=lambda r: (
            {"default": 0, "enhanced": 1, "pain": 2, "reject": 3}[r["Recommended Tier"]],
            -r["Weighted M-hot Teacher Score"],
            -r["Total Rows"],
        )
    )
    return rule_rows, summary


def build_a_hot_rules(rows: list[dict[str, str]]) -> tuple[list[dict[str, object]], dict[str, float]]:
    agg = defaultdict(float)
    for row in rows:
        if row["Prefetch Label"] != CLAIM_MODE:
            continue
        for bucket in A_DISTANCE_BUCKETS:
            agg[f"distance_{bucket}"] += as_float(row, f"oracle_a_local_distance_{bucket}")
            agg[f"distance_{bucket}"] += as_float(row, f"oracle_a_remote_distance_{bucket}")
            agg[f"remote_distance_{bucket}"] += as_float(
                row, f"oracle_a_remote_distance_{bucket}"
            )
        for bucket in A_REUSE_BUCKETS:
            agg[f"reuse_{bucket}"] += as_float(row, f"oracle_a_local_reuse_{bucket}")
            agg[f"reuse_{bucket}"] += as_float(row, f"oracle_a_remote_reuse_{bucket}")
            agg[f"remote_reuse_{bucket}"] += as_float(
                row, f"oracle_a_remote_reuse_{bucket}"
            )
        for bucket in A_RECURRENCE_BUCKETS:
            agg[f"recurrence_{bucket}"] += as_float(
                row, f"oracle_a_local_recurrence_{bucket}"
            )
            agg[f"recurrence_{bucket}"] += as_float(
                row, f"oracle_a_remote_recurrence_{bucket}"
            )
            agg[f"remote_recurrence_{bucket}"] += as_float(
                row, f"oracle_a_remote_recurrence_{bucket}"
            )

    short_distance = agg["distance_immediate"] + agg["distance_near"]
    multi_reuse = agg["reuse_multi"]
    repeat = agg["recurrence_repeat"]
    remote_repeat = agg["remote_recurrence_repeat"]
    candidate = min(short_distance, multi_reuse, repeat)

    rows_out = [
        {
            "Rule Name": "a_hot_teacher_provisional",
            "Status": "provisional",
            "Decision": "teacher_only_candidate",
            "Signals": "short_distance(immediate|near) AND multi_reuse AND recurrence/repeat",
            "Rationale": "A-side teacher evidence clusters around repeat + short-distance + multi-reuse, but runtime A-hot is not validated yet",
            "Candidate Count": round(candidate, 3),
            "Short Distance Count": round(short_distance, 3),
            "Multi Reuse Count": round(multi_reuse, 3),
            "Repeat Count": round(repeat, 3),
            "Recurring Remote Fetch Count": round(remote_repeat, 3),
        },
        {
            "Rule Name": "a_hot_runtime_hold",
            "Status": "hold",
            "Decision": "do_not_enable_yet",
            "Signals": "teacher support exists but runtime A-local hit benefit remains unvalidated",
            "Rationale": "Keep interface/labels only; do not deploy symmetric A-M-hot admission yet",
            "Candidate Count": 0,
            "Short Distance Count": round(short_distance, 3),
            "Multi Reuse Count": round(multi_reuse, 3),
            "Repeat Count": round(repeat, 3),
            "Recurring Remote Fetch Count": round(remote_repeat, 3),
        },
    ]

    summary = {
        "a_hot_candidate_count": candidate,
        "a_hot_repeat_count": repeat,
        "a_hot_short_distance_count": short_distance,
        "a_hot_multi_reuse_count": multi_reuse,
        "a_hot_future_j_tile_reuse_multi_count": agg["reuse_multi"],
        "a_hot_future_j_tile_reuse_one_count": agg["reuse_one"],
        "a_hot_recurring_remote_fetch_count": remote_repeat,
    }
    return rows_out, summary


def build_boundary_rows(b_summary: dict[str, float], a_summary: dict[str, float]) -> list[dict[str, object]]:
    return [
        {
            "vip_like_victim_count": round(b_summary.get("vip_like_victim_count", 0.0), 3),
            "mhot_like_hot_count": round(
                b_summary.get("mhot_like_hot_count", 0.0) + a_summary.get("a_hot_candidate_count", 0.0),
                3,
            ),
            "overlap_vip_mhot_candidate_count": round(
                b_summary.get("overlap_vip_mhot_candidate_count", 0.0), 3
            ),
            "neither_vip_nor_mhot_count": round(
                b_summary.get("neither_vip_nor_mhot_count", 0.0), 3
            ),
            "b_hot_candidate_count": round(b_summary.get("b_hot_candidate_count", 0.0), 3),
            "b_hot_next_output_immediate_count": round(
                b_summary.get("b_hot_next_output_immediate_count", 0.0), 3
            ),
            "b_hot_next_output_near_count": round(
                b_summary.get("b_hot_next_output_near_count", 0.0), 3
            ),
            "b_hot_next_output_far_count": round(
                b_summary.get("b_hot_next_output_far_count", 0.0),
                3,
            ),
            "b_hot_claim_count": round(b_summary.get("b_hot_claim_count", 0.0), 3),
            "b_hot_carry_over_count": round(b_summary.get("b_hot_carry_over_count", 0.0), 3),
            "b_hot_normal_count": round(b_summary.get("b_hot_normal_count", 0.0), 3),
            "a_hot_candidate_count": round(a_summary.get("a_hot_candidate_count", 0.0), 3),
            "a_hot_repeat_count": round(a_summary.get("a_hot_repeat_count", 0.0), 3),
            "a_hot_short_distance_count": round(
                a_summary.get("a_hot_short_distance_count", 0.0), 3
            ),
            "a_hot_multi_reuse_count": round(
                a_summary.get("a_hot_multi_reuse_count", 0.0), 3
            ),
            "a_hot_future_j_tile_reuse_one_count": round(
                a_summary.get("a_hot_future_j_tile_reuse_one_count", 0.0), 3
            ),
            "a_hot_future_j_tile_reuse_multi_count": round(
                a_summary.get("a_hot_future_j_tile_reuse_multi_count", 0.0), 3
            ),
            "a_hot_recurring_remote_fetch_count": round(
                a_summary.get("a_hot_recurring_remote_fetch_count", 0.0), 3
            ),
        }
    ]


def build_path_rules() -> list[dict[str, str]]:
    return [
        {
            "Step": "1",
            "Condition": "Always",
            "Action": "Try direct mainline / current-ready path first",
            "Notes": "Do not pre-check VIP or M-hot before mainline",
        },
        {
            "Step": "2",
            "Condition": "Object would otherwise go remote/fallback AND tag == vip_rescue",
            "Action": "Lookup VIP V3 rescue pool",
            "Notes": "VIP remains a small B-side repeated-victim rescue pool",
        },
        {
            "Step": "3",
            "Condition": "Object would otherwise go remote/fallback AND tag == mhot_hot",
            "Action": "Lookup M-hot hot pool",
            "Notes": "M-hot is a larger hot-object pool, not a mainline-front gate",
        },
        {
            "Step": "4",
            "Condition": "No nearby-pool hit",
            "Action": "Perform the remote fetch/fallback",
            "Notes": "Nearby pools are label-driven fallback-time shortcuts, not a unified cache hierarchy",
        },
    ]


def build_summary_md(
    workload_rows: list[dict[str, object]],
    b_hot_rules: list[dict[str, object]],
    a_hot_rules: list[dict[str, object]],
    boundary_rows: list[dict[str, object]],
    path_rules: list[dict[str, str]],
) -> None:
    top_default = [r for r in b_hot_rules if r["Recommended Tier"] == "default"][:4]
    top_enh = [r for r in b_hot_rules if r["Recommended Tier"] == "enhanced"][:4]
    boundary = boundary_rows[0]
    lines = [
        "# M-hot Oracle Teacher Summary",
        "",
        "## Ordering Rule",
        "",
        "- First try the direct mainline path.",
        "- Only when the object would otherwise go remote/fallback, consult nearby pools by label.",
        "- VIP remains the rescue-victim path.",
        "- M-hot is the hot-object path.",
        "- This is not a unified cache hierarchy.",
        "",
        "## Architecture Split",
        "",
        "- Mainline direct path: highest priority path for already-proven direct supply chains.",
        "- VIP V3: keep unchanged as B-side fallback-rescue.",
        "- M-hot: larger hot SRAM for future A-hot + B-hot, but this round only studies teacher-side tags.",
        "",
        "## Workloads",
        "",
        ", ".join(row["Preset"] for row in workload_rows),
        "",
        "## B-hot Default Rules",
        "",
    ]
    for row in top_default:
        lines.append(
            f"- {row['Source Class']} / {row['Distance Bucket']}: "
            f"tier={row['Recommended Tier']}, "
            f"coverage={row['Workload Coverage']}, "
            f"local_ratio={row['Local Ratio']:.3f}, "
            f"score={row['Weighted M-hot Teacher Score']:.3f}"
        )

    lines += [
        "",
        "## B-hot Enhanced Rules",
        "",
    ]
    for row in top_enh:
        lines.append(
            f"- {row['Source Class']} / {row['Distance Bucket']}: "
            f"tier={row['Recommended Tier']}, "
            f"coverage={row['Workload Coverage']}, "
            f"local_ratio={row['Local Ratio']:.3f}, "
            f"score={row['Weighted M-hot Teacher Score']:.3f}"
        )

    lines += [
        "",
        "## A-hot Provisional Takeaway",
        "",
        f"- Candidate count proxy: {boundary['a_hot_candidate_count']}",
        f"- Strongest stable teacher-side signals: repeat={boundary['a_hot_repeat_count']}, short_distance={boundary['a_hot_short_distance_count']}, multi_reuse={boundary['a_hot_multi_reuse_count']}",
        "- A still does not have runtime-validated local-hit benefit, so keep it provisional only.",
        "",
        "## VIP vs M-hot Boundary",
        "",
        f"- VIP-like victim count: {boundary['vip_like_victim_count']}",
        f"- M-hot-like hot count: {boundary['mhot_like_hot_count']}",
        f"- Overlap count: {boundary['overlap_vip_mhot_candidate_count']}",
        f"- Neither bucket: {boundary['neither_vip_nor_mhot_count']}",
        "",
        "## Practical Recommendation",
        "",
        "- Keep VIP V3 unchanged as the B-side rescue path.",
        "- If/when M-hot is implemented, default B-hot admission should start with next_output / immediate and next_output / near.",
        "- next_output / far, claim, and carry_over should remain enhanced-only candidates.",
        "- Do not enable runtime A-hot yet; keep A tags/metadata only.",
        "- Assume M-hot is a larger pool (>=64KB) and not a VIP-sized SRAM.",
        "",
        "## Path Rule",
        "",
    ]
    for row in path_rules:
        lines.append(f"- Step {row['Step']}: {row['Action']} ({row['Condition']})")

    SUMMARY_MD.write_text("\n".join(lines))


def main() -> None:
    rows = load_rows()
    workload_rows = workload_summary(rows)
    b_hot_rules, b_summary = build_b_hot_rules(rows)
    a_hot_rules, a_summary = build_a_hot_rules(rows)
    boundary_rows = build_boundary_rows(b_summary, a_summary)
    path_rules = build_path_rules()

    write_csv(WORKLOAD_SUMMARY_CSV, workload_rows)
    write_csv(B_HOT_RULES_CSV, b_hot_rules)
    write_csv(A_HOT_RULES_CSV, a_hot_rules)
    write_csv(BOUNDARY_STATS_CSV, boundary_rows)
    write_csv(PATH_RULES_CSV, path_rules)
    build_summary_md(workload_rows, b_hot_rules, a_hot_rules, boundary_rows, path_rules)

    print(f"OK workload summary: {WORKLOAD_SUMMARY_CSV}")
    print(f"OK B-hot rules: {B_HOT_RULES_CSV}")
    print(f"OK A-hot rules: {A_HOT_RULES_CSV}")
    print(f"OK boundary stats: {BOUNDARY_STATS_CSV}")
    print(f"OK path rules: {PATH_RULES_CSV}")
    print(f"OK summary: {SUMMARY_MD}")


if __name__ == "__main__":
    main()
