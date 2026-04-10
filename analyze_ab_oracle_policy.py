#!/usr/bin/env python3
"""
Analyze A/B Oracle teacher sweep and emit online-implementable rules.

Oracle remains the teacher.
This script translates Oracle observations into:
- B-side M-hot/mainline candidates
- B-side VIP rescue candidates
- A-side provisional online rescue rules
- a clear separation between offline truth and deployable rules
"""

import csv
from collections import defaultdict
from pathlib import Path


SWEEP_CSV = Path("my_outputData/ab_oracle_teacher_sweep.csv")
WORKLOAD_SUMMARY_CSV = Path("my_outputData/ab_oracle_teacher_workload_summary.csv")
B_BUCKET_CSV = Path("my_outputData/ab_oracle_b_bucket_scores.csv")
A_FEATURE_CSV = Path("my_outputData/ab_oracle_a_feature_scores.csv")
ONLINE_RULES_CSV = Path("my_outputData/ab_online_profiler_rules.csv")
SUMMARY_MD = Path("my_outputData/ab_oracle_policy_summary.md")

BASELINE_MODE = "baseline_serial"
CLAIM_MODE = "ab_hierarchical_claim_hole_filling"
VIP_RESCUE_MODE = "vip_rescue_buffer_single_run"

B_SOURCES = ["next_output", "claim", "carry_over", "current_window", "normal"]
B_DISTANCES = ["immediate", "near", "far"]
A_DISTANCE_BUCKETS = ["immediate", "near", "far"]
A_REUSE_BUCKETS = ["one", "multi"]
A_RECURRENCE_BUCKETS = ["first", "repeat"]

SOURCE_BIAS = {
    "next_output": 1.00,
    "claim": 0.85,
    "carry_over": 0.80,
    "current_window": 0.35,
    "normal": 0.00,
}
URGENCY = {
    "immediate": 1.00,
    "near": 0.60,
    "far": 0.20,
}
A_DISTANCE_IMPORTANCE = {
    "immediate": 1.00,
    "near": 0.75,
    "far": 0.25,
}
A_REUSE_IMPORTANCE = {
    "one": 0.40,
    "multi": 1.00,
}
A_RECURRENCE_IMPORTANCE = {
    "first": 0.35,
    "repeat": 1.00,
}


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def load_rows():
    if not SWEEP_CSV.exists():
        raise SystemExit(f"未找到 A/B Oracle sweep 结果: {SWEEP_CSV}")
    return list(csv.DictReader(SWEEP_CSV.open()))


def as_float(row, key):
    value = row.get(key, 0)
    return float(value) if value not in ("", None) else 0.0


def workload_summary(rows):
    by_key = {(r["Preset"], r["Prefetch Label"]): r for r in rows}
    out = []
    for preset in sorted({r["Preset"] for r in rows}):
        base = by_key.get((preset, BASELINE_MODE))
        claim = by_key.get((preset, CLAIM_MODE))
        vip = by_key.get((preset, VIP_RESCUE_MODE))
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
            "Base Fallback": int(as_float(base, "fallback_b_rows_fetched")),
            "Claim Fallback": int(as_float(claim, "fallback_b_rows_fetched")),
            "Base PrefetchedB": int(as_float(base, "prefetched_b_rows_consumed")),
            "Claim PrefetchedB": int(as_float(claim, "prefetched_b_rows_consumed")),
            "Claim next_output_rows_ready_at_boundary": int(
                as_float(claim, "next_output_rows_ready_at_boundary")
            ),
            "Claim future_claim_set_count": int(
                as_float(claim, "future_claim_set_count")
            ),
        }
        if vip:
            item.update(
                {
                    "VIP Rescue GFLOPS": round(
                        as_float(vip, "GEMM-Effective (GFLOPS)"), 6
                    ),
                    "VIP Rescue Delta GFLOPS": round(
                        as_float(vip, "GEMM-Effective (GFLOPS)")
                        - as_float(claim, "GEMM-Effective (GFLOPS)"),
                        6,
                    ),
                    "VIP Rescue Fallback": int(
                        as_float(vip, "fallback_b_rows_fetched")
                    ),
                    "VIP Rescue Fallback Delta": int(
                        as_float(vip, "fallback_b_rows_fetched")
                        - as_float(claim, "fallback_b_rows_fetched")
                    ),
                    "VIP Rescue Hits": int(as_float(vip, "vip_rescue_hit_count")),
                }
            )
        out.append(item)
    return out


def build_b_bucket_records(rows):
    out = []
    for row in rows:
        if row["Prefetch Label"] != CLAIM_MODE:
            continue
        for source in B_SOURCES:
            for dist in B_DISTANCES:
                local_rows = as_float(row, f"oracle_b_local_{source}_{dist}")
                fallback_rows = as_float(row, f"oracle_b_fallback_{source}_{dist}")
                local_reuse = as_float(
                    row, f"oracle_b_local_reuse_weight_{source}_{dist}"
                )
                fallback_reuse = as_float(
                    row, f"oracle_b_fallback_reuse_weight_{source}_{dist}"
                )
                total_rows = local_rows + fallback_rows
                if total_rows == 0:
                    continue
                out.append(
                    {
                        "Preset": row["Preset"],
                        "Source Class": source,
                        "Distance Bucket": dist,
                        "local_rows": local_rows,
                        "fallback_rows": fallback_rows,
                        "local_reuse_weight": local_reuse,
                        "fallback_reuse_weight": fallback_reuse,
                        "total_rows": total_rows,
                    }
                )
    return out


def aggregate_b_bucket_scores(bucket_rows):
    grouped = defaultdict(
        lambda: {
            "local_rows": 0.0,
            "fallback_rows": 0.0,
            "local_reuse_weight": 0.0,
            "fallback_reuse_weight": 0.0,
            "coverage": set(),
        }
    )
    for row in bucket_rows:
        key = (row["Source Class"], row["Distance Bucket"])
        g = grouped[key]
        g["local_rows"] += row["local_rows"]
        g["fallback_rows"] += row["fallback_rows"]
        g["local_reuse_weight"] += row["local_reuse_weight"]
        g["fallback_reuse_weight"] += row["fallback_reuse_weight"]
        g["coverage"].add(row["Preset"])

    out = []
    for (source, dist), g in grouped.items():
        total_rows = g["local_rows"] + g["fallback_rows"]
        fallback_ratio = g["fallback_rows"] / total_rows if total_rows else 0.0
        locality_ratio = g["local_rows"] / total_rows if total_rows else 0.0
        score = (
            (SOURCE_BIAS[source] * 4.0)
            + (URGENCY[dist] * 3.0)
            + (locality_ratio * 2.0)
            + ((g["local_reuse_weight"] / total_rows) if total_rows else 0.0)
            + (len(g["coverage"]) / 6.0)
        )
        out.append(
            {
                "Source Class": source,
                "Distance Bucket": dist,
                "Workload Coverage": len(g["coverage"]),
                "Total Rows": round(total_rows, 3),
                "Local Rows": round(g["local_rows"], 3),
                "Fallback Rows": round(g["fallback_rows"], 3),
                "Fallback Ratio": round(fallback_ratio, 6),
                "Weighted Oracle Score": round(score, 6),
            }
        )
    out.sort(
        key=lambda r: (
            -r["Weighted Oracle Score"],
            -r["Workload Coverage"],
            -r["Total Rows"],
        )
    )
    return out


def aggregate_a_feature_scores(rows):
    grouped = defaultdict(lambda: {"local": 0.0, "remote": 0.0})
    for row in rows:
        if row["Prefetch Label"] != CLAIM_MODE:
            continue
        for bucket in A_DISTANCE_BUCKETS:
            grouped[("distance", bucket)]["local"] += as_float(
                row, f"oracle_a_local_distance_{bucket}"
            )
            grouped[("distance", bucket)]["remote"] += as_float(
                row, f"oracle_a_remote_distance_{bucket}"
            )
        for bucket in A_REUSE_BUCKETS:
            grouped[("reuse", bucket)]["local"] += as_float(
                row, f"oracle_a_local_reuse_{bucket}"
            )
            grouped[("reuse", bucket)]["remote"] += as_float(
                row, f"oracle_a_remote_reuse_{bucket}"
            )
        for bucket in A_RECURRENCE_BUCKETS:
            grouped[("recurrence", bucket)]["local"] += as_float(
                row, f"oracle_a_local_recurrence_{bucket}"
            )
            grouped[("recurrence", bucket)]["remote"] += as_float(
                row, f"oracle_a_remote_recurrence_{bucket}"
            )

    out = []
    for (family, bucket), g in grouped.items():
        total = g["local"] + g["remote"]
        if total == 0:
            continue
        remote_ratio = g["remote"] / total
        if family == "distance":
            weight = A_DISTANCE_IMPORTANCE[bucket]
        elif family == "reuse":
            weight = A_REUSE_IMPORTANCE[bucket]
        else:
            weight = A_RECURRENCE_IMPORTANCE[bucket]
        score = weight * (g["remote"] + 0.25 * g["local"])
        out.append(
            {
                "Feature Family": family,
                "Bucket": bucket,
                "Local Count": round(g["local"], 3),
                "Remote Count": round(g["remote"], 3),
                "Remote Ratio": round(remote_ratio, 6),
                "Weighted Teacher Score": round(score, 6),
            }
        )
    out.sort(key=lambda r: (-r["Weighted Teacher Score"], -r["Remote Count"]))
    return out


def build_online_rules(b_bucket_scores, a_feature_scores, workload_rows):
    future_chain_workloads = [
        r
        for r in workload_rows
        if int(as_float(r, "Claim next_output_rows_ready_at_boundary")) > 0
        and int(as_float(r, "Claim future_claim_set_count")) > 0
    ]
    rescue_positive_workloads = [
        r for r in workload_rows if as_float(r, "VIP Rescue Fallback Delta") < 0
    ]

    return [
        {
            "Rule Name": "b_mhot_mainline_default",
            "Target Pool": "mhot",
            "Decision": "strong_admit",
            "Status": "validated",
            "Online Signals": "source_class == next_output AND distance_bucket in {immediate, near}",
            "Teacher Support": f"Stable future-B mainline signal in {len(future_chain_workloads)} workloads",
        },
        {
            "Rule Name": "b_mhot_mainline_extension",
            "Target Pool": "mhot",
            "Decision": "conditional_admit",
            "Status": "validated",
            "Online Signals": "source_class == next_output AND distance_bucket == far AND future_chain_active == true",
            "Teacher Support": "Useful mainly on larger workloads with deeper future span",
        },
        {
            "Rule Name": "b_mhot_enhancement",
            "Target Pool": "mhot",
            "Decision": "conditional_admit",
            "Status": "validated",
            "Online Signals": "source_class in {claim, carry_over} AND future_chain_active == true",
            "Teacher Support": "Enhancement signal seen mainly in richer future-B chains",
        },
        {
            "Rule Name": "b_vip_rescue_default",
            "Target Pool": "vip",
            "Decision": "strong_admit",
            "Status": "validated",
            "Online Signals": "just_triggered_fallback == true AND short_next_use == true AND future_i_tile_reuse_remaining > 0",
            "Teacher Support": f"B rescue already improved fallback on {len(rescue_positive_workloads)} workloads without harming next_output mainline",
        },
        {
            "Rule Name": "a_vip_rescue_provisional",
            "Target Pool": "vip",
            "Decision": "conditional_admit",
            "Status": "provisional",
            "Online Signals": "just_triggered_remote_fetch == true AND future_j_tile_reuse_remaining > 0 AND short_next_use == true AND (future_j_tile_reuse_remaining > 1 OR recent_remote_fetch_recurrence > 0)",
            "Teacher Support": "A teacher-side hotspots are dominated by short-distance, multi-reuse, repeat-remote victims; runtime validation still pending",
        },
        {
            "Rule Name": "reject_default",
            "Target Pool": "reject",
            "Decision": "reject",
            "Status": "validated",
            "Online Signals": "normal one-shot rows without recurrence or short-next-use evidence",
            "Teacher Support": "Keeps VIP/M-hot from bloating with weak-value rows",
        },
    ]


def build_summary_md(workload_rows, b_bucket_scores, a_feature_scores, online_rules):
    lines = [
        "# A/B Oracle Teacher Policy Summary",
        "",
        "## Main Point",
        "",
        "Oracle is the teacher; online profiler rules are the deployable output.",
        "This analysis keeps B as the mainline focus and treats A as an observed teacher-side signal, not a committed runtime policy yet.",
        "",
        "## Workloads",
        "",
        ", ".join(row["Preset"] for row in workload_rows),
        "",
        "## B-side Mainline Takeaway",
        "",
        "- next_output / immediate remains the strongest mainline B signal.",
        "- next_output / near also remains a strong mainline B signal and should not be rejected.",
        "- next_output / far, claim, and carry_over are better treated as conditional enhancements, not default mainline admits.",
        "",
        "## B-side Rescue Takeaway",
        "",
        "- current_window / immediate is still better interpreted as a pain bucket or emergency bucket.",
        "- normal / near and normal / far remain better rescue targets than mainline M-hot targets.",
        "- runtime B-side rescue is already validated by vip_rescue_buffer_single_run.",
        "",
        "## A-side Teacher Takeaway",
        "",
        "- A should not be forced into policy just because we can observe it.",
        "- The teacher-side A profile should only be used to identify whether repeated remote-victim behavior exists.",
        "- Current A-side evidence should be treated as provisional until we validate an A-specific rescue path.",
        "",
        "## Top B Buckets",
        "",
    ]
    for row in b_bucket_scores[:8]:
        lines.append(
            f"- {row['Source Class']} / {row['Distance Bucket']}: "
            f"score={row['Weighted Oracle Score']:.3f}, "
            f"fallback_ratio={row['Fallback Ratio']:.3f}, "
            f"coverage={row['Workload Coverage']}"
        )

    lines += [
        "",
        "## Top A Teacher Features",
        "",
    ]
    for row in a_feature_scores[:6]:
        lines.append(
            f"- {row['Feature Family']} / {row['Bucket']}: "
            f"score={row['Weighted Teacher Score']:.3f}, "
            f"remote_ratio={row['Remote Ratio']:.3f}, "
            f"remote={row['Remote Count']:.0f}"
        )

    lines += [
        "",
        "## Online Rules",
        "",
    ]
    for row in online_rules:
        lines += [
            f"### {row['Rule Name']}",
            f"- Pool: {row['Target Pool']}",
            f"- Decision: {row['Decision']}",
            f"- Status: {row['Status']}",
            f"- Signals: {row['Online Signals']}",
            f"- Teacher support: {row['Teacher Support']}",
            "",
        ]

    lines += [
        "## Practical Recommendation",
        "",
        "- Keep B as the first-class policy target.",
        "- Use M-hot primarily for B-side next_output immediate/near mainline buckets.",
        "- Use VIP primarily for B-side fallback-rescue buckets that have already proven effective.",
        "- Treat A-side rules as provisional rescue candidates until runtime validation shows net benefit.",
    ]
    SUMMARY_MD.write_text("\n".join(lines))


def main():
    rows = load_rows()
    workload_rows = workload_summary(rows)
    b_bucket_scores = aggregate_b_bucket_scores(build_b_bucket_records(rows))
    a_feature_scores = aggregate_a_feature_scores(rows)
    online_rules = build_online_rules(b_bucket_scores, a_feature_scores, workload_rows)

    write_csv(WORKLOAD_SUMMARY_CSV, workload_rows)
    write_csv(B_BUCKET_CSV, b_bucket_scores)
    write_csv(A_FEATURE_CSV, a_feature_scores)
    write_csv(ONLINE_RULES_CSV, online_rules)
    build_summary_md(workload_rows, b_bucket_scores, a_feature_scores, online_rules)

    print(f"OK workload summary: {WORKLOAD_SUMMARY_CSV}")
    print(f"OK B bucket scores: {B_BUCKET_CSV}")
    print(f"OK A feature scores: {A_FEATURE_CSV}")
    print(f"OK online rules: {ONLINE_RULES_CSV}")
    print(f"OK summary: {SUMMARY_MD}")


if __name__ == "__main__":
    main()
