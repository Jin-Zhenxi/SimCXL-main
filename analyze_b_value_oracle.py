#!/usr/bin/env python3
"""
Offline Oracle analysis + B-value profiler extraction.

Reads oracle_b_value_sweep.csv and produces:
- workload-level deltas
- oracle bucket ranking (source x next-use distance)
- a first-pass implementable profiler rule table
"""

import csv
from collections import defaultdict
from pathlib import Path

SWEEP_CSV = Path("my_outputData/b_value_oracle_sweep.csv")
WORKLOAD_SUMMARY_CSV = Path("my_outputData/b_value_oracle_workload_summary.csv")
BUCKET_SCORES_CSV = Path("my_outputData/b_value_oracle_bucket_scores.csv")
PROFILER_RULES_CSV = Path("my_outputData/b_value_oracle_profiler_rules.csv")
SUMMARY_MD = Path("my_outputData/b_value_oracle_summary.md")

TARGET_MODE = "ab_hierarchical_claim_hole_filling"
BASELINE_MODE = "baseline_serial"

ORACLE_SOURCES = [
    "next_output",
    "claim",
    "carry_over",
    "current_window",
    "normal",
]
ORACLE_DISTANCES = ["immediate", "near", "far"]

SOURCE_BIAS = {
    "next_output": 1.00,
    "claim": 0.90,
    "carry_over": 0.80,
    "current_window": 0.40,
    "normal": 0.00,
}
URGENCY = {
    "immediate": 1.00,
    "near": 0.60,
    "far": 0.20,
}


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_rows():
    if not SWEEP_CSV.exists():
        raise SystemExit(f"未找到 sweep 结果: {SWEEP_CSV}")
    return list(csv.DictReader(SWEEP_CSV.open()))


def as_float(row, key):
    value = row.get(key, 0)
    return float(value) if value not in ("", None) else 0.0


def workload_summary(rows):
    by_key = {(r["Preset"], r["Prefetch Label"]): r for r in rows}
    out = []
    for preset in sorted({r["Preset"] for r in rows}):
        base = by_key.get((preset, BASELINE_MODE))
        claim = by_key.get((preset, TARGET_MODE))
        if not base or not claim:
            continue
        out.append(
            {
                "Preset": preset,
                "Base GFLOPS": round(as_float(base, "GEMM-Effective (GFLOPS)"), 6),
                "Claim GFLOPS": round(
                    as_float(claim, "GEMM-Effective (GFLOPS)"), 6
                ),
                "Delta GFLOPS": round(
                    as_float(claim, "GEMM-Effective (GFLOPS)")
                    - as_float(base, "GEMM-Effective (GFLOPS)"),
                    6,
                ),
                "Base Share (%)": round(as_float(base, "GEMM Compute Share (%)"), 4),
                "Claim Share (%)": round(
                    as_float(claim, "GEMM Compute Share (%)"), 4
                ),
                "Delta Share (pp)": round(
                    as_float(claim, "GEMM Compute Share (%)")
                    - as_float(base, "GEMM Compute Share (%)"),
                    4,
                ),
                "Base Fallback": int(as_float(base, "fallback_b_rows_fetched")),
                "Claim Fallback": int(as_float(claim, "fallback_b_rows_fetched")),
                "Delta Fallback": int(
                    as_float(claim, "fallback_b_rows_fetched")
                    - as_float(base, "fallback_b_rows_fetched")
                ),
                "Base PrefetchedB": int(
                    as_float(base, "prefetched_b_rows_consumed")
                ),
                "Claim PrefetchedB": int(
                    as_float(claim, "prefetched_b_rows_consumed")
                ),
                "Delta PrefetchedB": int(
                    as_float(claim, "prefetched_b_rows_consumed")
                    - as_float(base, "prefetched_b_rows_consumed")
                ),
                "Claim next_output_rows_ready_at_boundary": int(
                    as_float(claim, "next_output_rows_ready_at_boundary")
                ),
                "Claim future_claim_set_count": int(
                    as_float(claim, "future_claim_set_count")
                ),
            }
        )
    return out


def build_bucket_records(rows):
    target_rows = [r for r in rows if r["Prefetch Label"] == TARGET_MODE]
    raw_records = []
    max_avg_reuse = 1.0
    for row in target_rows:
        for source in ORACLE_SOURCES:
            for dist in ORACLE_DISTANCES:
                local_rows = as_float(
                    row, f"oracleLocalRows_{source}_{dist}"
                )
                fallback_rows = as_float(
                    row, f"oracleFallbackRows_{source}_{dist}"
                )
                local_weight = as_float(
                    row, f"oracleLocalReuseWeight_{source}_{dist}"
                )
                fallback_weight = as_float(
                    row, f"oracleFallbackReuseWeight_{source}_{dist}"
                )
                total_rows = local_rows + fallback_rows
                total_weight = local_weight + fallback_weight
                if total_rows <= 0:
                    continue
                avg_reuse = total_weight / total_rows
                max_avg_reuse = max(max_avg_reuse, avg_reuse)
                raw_records.append(
                    {
                        "Preset": row["Preset"],
                        "Source Class": source,
                        "Distance Bucket": dist,
                        "Local Rows": local_rows,
                        "Fallback Rows": fallback_rows,
                        "Total Rows": total_rows,
                        "Local Reuse Weight": local_weight,
                        "Fallback Reuse Weight": fallback_weight,
                        "Total Reuse Weight": total_weight,
                        "Avg Future Use Proxy": avg_reuse,
                    }
                )

    bucket_rows = []
    for rec in raw_records:
        fallback_risk = (
            rec["Fallback Rows"] / rec["Total Rows"] if rec["Total Rows"] > 0 else 0.0
        )
        reuse_norm = rec["Avg Future Use Proxy"] / max_avg_reuse
        urgency = URGENCY[rec["Distance Bucket"]]
        source_bias = SOURCE_BIAS[rec["Source Class"]]
        criticality = min(
            1.0, 0.55 * urgency + 0.45 * source_bias
        )
        oracle_score = (
            4.0 * fallback_risk
            + 3.0 * urgency
            + 2.0 * reuse_norm
            + 2.0 * criticality
            + 2.0 * source_bias
        )
        row = dict(rec)
        row["Fallback Risk"] = round(fallback_risk, 6)
        row["Urgency"] = round(urgency, 6)
        row["Source Bias"] = round(source_bias, 6)
        row["Criticality"] = round(criticality, 6)
        row["Reuse Norm"] = round(reuse_norm, 6)
        row["Oracle Score"] = round(oracle_score, 6)
        bucket_rows.append(row)
    return bucket_rows


def aggregate_bucket_scores(bucket_rows):
    grouped = defaultdict(list)
    for row in bucket_rows:
        grouped[(row["Source Class"], row["Distance Bucket"])].append(row)

    out = []
    for (source, dist), items in grouped.items():
        total_rows = sum(r["Total Rows"] for r in items)
        total_fallback = sum(r["Fallback Rows"] for r in items)
        total_local = sum(r["Local Rows"] for r in items)
        total_weight = sum(r["Total Reuse Weight"] for r in items)
        weighted_score = (
            sum(r["Oracle Score"] * r["Total Rows"] for r in items) / total_rows
            if total_rows > 0
            else 0.0
        )
        fallback_ratio = total_fallback / total_rows if total_rows > 0 else 0.0
        avg_reuse = total_weight / total_rows if total_rows > 0 else 0.0
        workload_coverage = sum(1 for r in items if r["Total Rows"] > 0)
        out.append(
            {
                "Source Class": source,
                "Distance Bucket": dist,
                "Workload Coverage": workload_coverage,
                "Total Rows": round(total_rows, 2),
                "Total Local Rows": round(total_local, 2),
                "Total Fallback Rows": round(total_fallback, 2),
                "Fallback Ratio": round(fallback_ratio, 6),
                "Avg Future Use Proxy": round(avg_reuse, 6),
                "Weighted Oracle Score": round(weighted_score, 6),
            }
        )
    out.sort(
        key=lambda r: (
            r["Weighted Oracle Score"],
            r["Fallback Ratio"],
            r["Avg Future Use Proxy"],
        ),
        reverse=True,
    )
    return out


def derive_profiler_rules(bucket_scores):
    if not bucket_scores:
        return []
    scores = [row["Weighted Oracle Score"] for row in bucket_scores]
    strong_cut = sorted(scores, reverse=True)[max(0, len(scores) // 3 - 1)]
    weak_cut = sorted(scores, reverse=True)[max(0, (2 * len(scores)) // 3 - 1)]

    rules = []
    for row in bucket_scores:
        source = row["Source Class"]
        dist = row["Distance Bucket"]
        score = row["Weighted Oracle Score"]
        fallback_ratio = row["Fallback Ratio"]
        coverage = row["Workload Coverage"]

        decision = "reject"
        rationale = "远期/普通 B，或 fallback 风险与关键性都不高"
        if source == "next_output" and dist in {"immediate", "near"} and coverage >= 3:
            decision = "strong_admit"
            rationale = "future-B 主链，next-use 近，且跨 workload 稳定出现"
        elif (
            source in {"claim", "carry_over"}
            and dist in {"immediate", "near"}
            and score >= strong_cut
            and coverage >= 2
        ):
            decision = "weak_admit"
            rationale = "future-B 增强信号，next-use 近，但当前普适性弱于 next-output 主链"
        elif (
            source == "current_window"
            and dist == "immediate"
            and score >= weak_cut
            and fallback_ratio > 0.0
        ) or (
            source in {"next_output", "claim", "carry_over"}
            and dist == "far"
            and score >= weak_cut
        ):
            decision = "weak_admit"
            rationale = "有短期关键性或未来价值，但稳定性/紧迫度低于强准入"

        rules.append(
            {
                "Source Class": source,
                "Distance Bucket": dist,
                "Weighted Oracle Score": row["Weighted Oracle Score"],
                "Fallback Ratio": fallback_ratio,
                "Workload Coverage": coverage,
                "Suggested Decision": decision,
                "Rationale": rationale,
            }
        )
    return rules


def build_summary_md(workload_rows, bucket_scores, rules):
    strong = [r for r in rules if r["Suggested Decision"] == "strong_admit"]
    weak = [r for r in rules if r["Suggested Decision"] == "weak_admit"]
    reject = [r for r in rules if r["Suggested Decision"] == "reject"]

    lines = [
        "# Oracle / B-value Profiler Summary",
        "",
        "## Workloads",
        "",
        ", ".join(row["Preset"] for row in workload_rows),
        "",
        "## Oracle High-Value B Definition",
        "",
        "High-value B = next use is near, future reuse proxy is high, and losing locality is likely to turn into fallback/normal fetch.",
        "",
        "## Workload-Level Deltas (baseline -> claim)",
        "",
    ]
    for row in workload_rows:
        lines.append(
            f"- {row['Preset']}: GFLOPS {row['Base GFLOPS']:.3f} -> {row['Claim GFLOPS']:.3f}, "
            f"fallback {row['Base Fallback']} -> {row['Claim Fallback']}, "
            f"prefetchedB {row['Base PrefetchedB']} -> {row['Claim PrefetchedB']}"
        )

    lines += [
        "",
        "## Top Oracle Buckets",
        "",
    ]
    for row in bucket_scores[:8]:
        lines.append(
            f"- {row['Source Class']} / {row['Distance Bucket']}: "
            f"score={row['Weighted Oracle Score']:.3f}, "
            f"fallback_ratio={row['Fallback Ratio']:.3f}, "
            f"coverage={row['Workload Coverage']}"
        )

    lines += [
        "",
        "## Profiler Rules",
        "",
        "### Strong Admit",
    ]
    for row in strong:
        lines.append(
            f"- {row['Source Class']} / {row['Distance Bucket']}: {row['Rationale']}"
        )
    lines += [
        "",
        "### Weak Admit",
    ]
    for row in weak:
        lines.append(
            f"- {row['Source Class']} / {row['Distance Bucket']}: {row['Rationale']}"
        )
    lines += [
        "",
        "### Reject",
    ]
    for row in reject:
        lines.append(
            f"- {row['Source Class']} / {row['Distance Bucket']}: {row['Rationale']}"
        )

    lines += [
        "",
        "## Architecture Recommendation",
        "",
        "- next_output / immediate and next_output / near are the most stable mainline future-B signals.",
        "- next-use distance is the most stable helper feature: immediate/near are clearly stronger than far.",
        "- current_window / immediate is better treated as a pain bucket or emergency reserve, not a default mainline admit.",
        "- normal-B should not be a default VIP / M-hot admit target.",
    ]
    SUMMARY_MD.write_text("\n".join(lines))


def main():
    rows = load_rows()
    workload_rows = workload_summary(rows)
    bucket_rows = build_bucket_records(rows)
    bucket_scores = aggregate_bucket_scores(bucket_rows)
    profiler_rules = derive_profiler_rules(bucket_scores)

    write_csv(WORKLOAD_SUMMARY_CSV, workload_rows)
    write_csv(BUCKET_SCORES_CSV, bucket_scores)
    write_csv(PROFILER_RULES_CSV, profiler_rules)
    build_summary_md(workload_rows, bucket_scores, profiler_rules)

    print(f"✅ workload 摘要: {WORKLOAD_SUMMARY_CSV}")
    print(f"✅ oracle bucket 排名: {BUCKET_SCORES_CSV}")
    print(f"✅ profiler 候选规则: {PROFILER_RULES_CSV}")
    print(f"✅ 文本摘要: {SUMMARY_MD}")


if __name__ == "__main__":
    main()
