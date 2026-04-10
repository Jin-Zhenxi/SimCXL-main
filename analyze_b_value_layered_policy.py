#!/usr/bin/env python3
"""
Layered B-policy analysis.

Purpose:
- Correct the raw Oracle bucket view into an architecture-oriented policy view.
- Separate:
  1. M-hot mainline candidates
  2. VIP rescue candidates
  3. Reject / non-default candidates
- Emit online-implementable rules rather than Oracle-only rules.
"""

import csv
from pathlib import Path


BUCKET_SCORES_CSV = Path("my_outputData/b_value_oracle_bucket_scores.csv")
WORKLOAD_SUMMARY_CSV = Path("my_outputData/b_value_oracle_workload_summary.csv")
OUT_LAYERED_BUCKETS_CSV = Path("my_outputData/b_value_layered_bucket_policy.csv")
OUT_ONLINE_RULES_CSV = Path("my_outputData/b_value_online_profiler_rules.csv")
OUT_SUMMARY_MD = Path("my_outputData/b_value_layered_policy_summary.md")


def load_csv(path):
    if not path.exists():
        raise SystemExit(f"Missing input file: {path}")
    with path.open() as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def as_float(row, key):
    val = row.get(key, 0)
    return float(val) if val not in ("", None) else 0.0


def classify_bucket(row):
    source = row["Source Class"]
    dist = row["Distance Bucket"]
    coverage = int(row["Workload Coverage"])
    fallback_ratio = as_float(row, "Fallback Ratio")
    score = as_float(row, "Weighted Oracle Score")

    pool = "reject"
    tier = "reject"
    reason = "Not a default near-end target"

    if source == "next_output" and dist in {"immediate", "near"} and coverage >= 3:
        pool = "mhot"
        tier = "strong_mainline"
        reason = "Most stable future-B mainline bucket across workloads"
    elif source == "next_output" and dist == "far" and coverage >= 2:
        pool = "mhot"
        tier = "conditional_mainline"
        reason = "Large-workload future-span extension signal"
    elif source in {"claim", "carry_over"} and coverage >= 1:
        pool = "mhot"
        tier = "conditional_enhancement"
        reason = "Enhancement signal that appears mainly in richer future-B chains"
    elif source == "current_window" and dist == "immediate" and fallback_ratio >= 0.9:
        pool = "vip"
        tier = "pain_bucket_emergency"
        reason = "Pain bucket: suitable for small emergency reserve or rescue heuristics, not mainline caching"
    elif source == "normal" and dist in {"near", "far"} and fallback_ratio >= 0.9:
        pool = "vip"
        tier = "rescue_candidate"
        reason = "Residual fallback victim bucket; better suited for rescue buffer than mainline pool"
    elif source == "current_window" and dist == "far" and fallback_ratio >= 0.9:
        pool = "vip"
        tier = "rescue_candidate"
        reason = "Pain bucket with weak future locality; only rescue if recurrence is observed"

    return {
        "Source Class": source,
        "Distance Bucket": dist,
        "Workload Coverage": coverage,
        "Weighted Oracle Score": round(score, 6),
        "Fallback Ratio": round(fallback_ratio, 6),
        "Recommended Pool": pool,
        "Tier": tier,
        "Reason": reason,
    }


def build_online_rules(bucket_rows, workload_rows):
    future_chain_workloads = [
        r for r in workload_rows
        if int(as_float(r, "Claim next_output_rows_ready_at_boundary")) > 0
        and int(as_float(r, "Claim future_claim_set_count")) > 0
    ]
    future_chain_support = len(future_chain_workloads)

    rules = [
        {
            "Rule Name": "mhot_mainline_default",
            "Target Pool": "mhot",
            "Decision": "strong_admit",
            "Online Signals": "source_class == next_output AND distance_bucket in {immediate, near}",
            "Oracle Support": f"Stable in {future_chain_support} workloads; next_output immediate/near is the most consistent mainline future-B signal",
            "Why Implementable": "source_class and distance bucket are already derived inside MatrixFlow scheduler",
        },
        {
            "Rule Name": "mhot_mainline_extension",
            "Target Pool": "mhot",
            "Decision": "conditional_admit",
            "Online Signals": "source_class == next_output AND distance_bucket == far AND future_chain_active == true",
            "Oracle Support": "Useful mainly on larger workloads with deeper future span",
            "Why Implementable": "future_chain_active can be approximated by next_output_rows_ready_at_boundary > 0 and future_claim_set_count > 0",
        },
        {
            "Rule Name": "mhot_future_enhancement",
            "Target Pool": "mhot",
            "Decision": "conditional_admit",
            "Online Signals": "source_class in {claim, carry_over} AND future_chain_active == true",
            "Oracle Support": "Appears as a large-workload enhancement signal, not a universal default rule",
            "Why Implementable": "claim and carry-over state already exist in runtime metadata",
        },
        {
            "Rule Name": "vip_rescue_default",
            "Target Pool": "vip",
            "Decision": "strong_admit",
            "Online Signals": "just_triggered_fallback == true AND short_next_use == true AND future_tile_reuse_remaining > 0",
            "Oracle Support": "Current_window immediate and normal near/far appear as pain buckets; rescue data already shows fallback reduction without harming mainline",
            "Why Implementable": "all three signals can be estimated online without Oracle labels",
        },
        {
            "Rule Name": "vip_rescue_repeat_victim",
            "Target Pool": "vip",
            "Decision": "conditional_admit",
            "Online Signals": "just_triggered_fallback == true AND (future_tile_reuse_remaining > 1 OR recent_fallback_recurrence > 0)",
            "Oracle Support": "Useful for repeated fallback victims even when distance bucket is not explicitly labeled",
            "Why Implementable": "fallback recurrence and future tile reuse can be maintained by small counters",
        },
        {
            "Rule Name": "reject_normal_default",
            "Target Pool": "reject",
            "Decision": "reject",
            "Online Signals": "source_class == normal AND no rescue recurrence AND no short next-use evidence",
            "Oracle Support": "normal buckets mostly expose pain, not default near-end value",
            "Why Implementable": "keeps pools from bloating with one-shot or weak-value rows",
        },
    ]
    return rules


def build_summary_md(layered_rows, online_rules, workload_rows):
    mhot_rows = [r for r in layered_rows if r["Recommended Pool"] == "mhot"]
    vip_rows = [r for r in layered_rows if r["Recommended Pool"] == "vip"]
    reject_rows = [r for r in layered_rows if r["Recommended Pool"] == "reject"]

    lines = [
        "# Layered B-Policy Summary",
        "",
        "## Main Point",
        "",
        "Oracle should be treated as a teacher, not as the deployed policy itself.",
        "The deployed policy should be split into:",
        "- M-hot for broad, stable future-B mainline buckets",
        "- VIP for small rescue-only fallback victims",
        "",
        "## Workloads Used",
        "",
        ", ".join(r["Preset"] for r in workload_rows),
        "",
        "## Corrected Mainline Reading",
        "",
        "- next_output / immediate is a strong mainline bucket.",
        "- next_output / near is also a strong mainline bucket and should not be rejected.",
        "- claim / carry_over look more like conditional large-workload enhancements than universal defaults.",
        "- current_window / immediate is better interpreted as a pain bucket / emergency signal than as a mainline cache target.",
        "",
        "## Recommended M-hot Buckets",
        "",
    ]
    for row in mhot_rows:
        lines.append(
            f"- {row['Source Class']} / {row['Distance Bucket']}: {row['Tier']} ({row['Reason']})"
        )

    lines += [
        "",
        "## Recommended VIP Rescue Buckets",
        "",
    ]
    for row in vip_rows:
        lines.append(
            f"- {row['Source Class']} / {row['Distance Bucket']}: {row['Tier']} ({row['Reason']})"
        )

    lines += [
        "",
        "## Buckets To Reject By Default",
        "",
    ]
    for row in reject_rows:
        lines.append(
            f"- {row['Source Class']} / {row['Distance Bucket']}"
        )

    lines += [
        "",
        "## Online-Implementable Rules",
        "",
    ]
    for row in online_rules:
        lines += [
            f"### {row['Rule Name']}",
            f"- Pool: {row['Target Pool']}",
            f"- Decision: {row['Decision']}",
            f"- Signals: {row['Online Signals']}",
            f"- Oracle support: {row['Oracle Support']}",
            f"- Why implementable: {row['Why Implementable']}",
            "",
        ]

    lines += [
        "## Practical Takeaway",
        "",
        "- M-hot should carry the stable future-B mainline, especially next_output immediate/near.",
        "- VIP should stay small and specialized, focusing on fallback victims and rescue opportunities.",
        "- Oracle remains useful for validation and rule extraction, but online admission must rely on runtime-observable signals.",
    ]

    OUT_SUMMARY_MD.write_text("\n".join(lines))


def main():
    bucket_rows = load_csv(BUCKET_SCORES_CSV)
    workload_rows = load_csv(WORKLOAD_SUMMARY_CSV)

    layered_rows = [classify_bucket(row) for row in bucket_rows]
    layered_rows.sort(
        key=lambda r: (
            {"mhot": 0, "vip": 1, "reject": 2}[r["Recommended Pool"]],
            {"strong_mainline": 0, "conditional_mainline": 1,
             "conditional_enhancement": 2, "pain_bucket_emergency": 3,
             "rescue_candidate": 4, "reject": 5}[r["Tier"]],
            -r["Weighted Oracle Score"],
        )
    )
    online_rules = build_online_rules(layered_rows, workload_rows)

    write_csv(OUT_LAYERED_BUCKETS_CSV, layered_rows)
    write_csv(OUT_ONLINE_RULES_CSV, online_rules)
    build_summary_md(layered_rows, online_rules, workload_rows)

    print(f"OK layered buckets: {OUT_LAYERED_BUCKETS_CSV}")
    print(f"OK online rules: {OUT_ONLINE_RULES_CSV}")
    print(f"OK summary: {OUT_SUMMARY_MD}")


if __name__ == "__main__":
    main()
