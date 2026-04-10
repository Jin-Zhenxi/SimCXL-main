# B-hot Rules

## Purpose

This note freezes the current B-side hotness interpretation from the Oracle
results.

The key lesson is:

`B-hot` must be split into two different views:

1. `Mainline B-hot`
2. `Fallback-oriented B-hot`

Those are different optimization targets.

## Evidence Base

Teacher-side hotness:

- [mhot_oracle_b_hot_rules.csv](/home/jzx8091/SimCXL-main/my_outputData/mhot_oracle_b_hot_rules.csv)
- [mhot_oracle_summary.md](/home/jzx8091/SimCXL-main/my_outputData/mhot_oracle_summary.md)

Fallback burden:

- [fallback_oracle_b_carriers.csv](/home/jzx8091/SimCXL-main/my_outputData/fallback_oracle_b_carriers.csv)
- [fallback_oracle_summary.md](/home/jzx8091/SimCXL-main/my_outputData/fallback_oracle_summary.md)
- [fallback_oracle_three_way_split.csv](/home/jzx8091/SimCXL-main/my_outputData/fallback_oracle_three_way_split.csv)

## B-hot Rule Set 1: Mainline B-hot

This rule set answers:

- which B objects are the most stable long-lived hot objects
- which objects are most natural for a future `M-hot` mainline pool

### Default Mainline B-hot

These are the current default B-hot rules:

- `next_output / immediate`
- `next_output / near`

Why:

- they appear across most workloads
- they are already strongly local-served
- they represent the most stable future-B hotness pattern

Current evidence:

- `next_output / immediate`
  - coverage = 5
  - local_ratio = 1.0
- `next_output / near`
  - coverage = 5
  - local_ratio = 1.0

### Enhanced Mainline B-hot

These remain enhancement-only candidates:

- `next_output / far`
- `claim`
- `carry_over`

They should not be considered default runtime rules yet.

Why:

- they are more workload-specific
- they appear later / deeper in future-span chains
- they are useful for large workloads, but not stable enough to be default

## B-hot Rule Set 2: Fallback-oriented B-hot

This rule set answers:

- which B objects are actually tied to the current residual fallback burden
- which objects are most likely to matter if the goal is to reduce fallback

### Best Current M-hot Gap Candidates

These are the strongest current candidates:

- `normal / near`
- `current_window / far`

Why:

- they are not already fully handled by direct mainline
- they are not pure rescue-victim buckets either
- they look like `mainline-adjacent slips`

Current evidence:

- `normal / near`
  - fallback_rows = 768
  - fallback_ratio = 1.0
  - coverage = 6
- `current_window / far`
  - fallback_rows = 551
  - fallback_ratio = 1.0
  - coverage = 4

These are the strongest current `future B-side M-hot` candidates.

### Buckets That Should Not Be Default M-hot

#### Already handled by mainline

- `next_output / immediate`
- `next_output / near`

These are important hot objects, but they are not the current fallback carrier.
Their value is already being realized by the direct mainline chain.

#### Better owned by VIP

- `current_window / immediate`

This is a short-distance pit bucket and is better interpreted as a rescue victim
than a default M-hot object.

#### Pain bucket, not default hot rule

- `normal / far`

This is currently the largest fallback burden:

- fallback_rows = 2976

But it is too broad to adopt as a default M-hot runtime rule. It should remain
a pain signal rather than a default admission class.

## Current B-side Ownership Split

### Mainline direct path

- `next_output / immediate`
- `next_output / near`

### VIP V3

- repeated fallback victims
- `current_window / immediate`

### Future B-side M-hot

- `normal / near`
- `current_window / far`

### Not default-admit

- `normal / far`

## Current Implementation Guidance

When we resume runtime work, the B-side sequence should be:

1. keep the direct mainline chain intact
2. keep `VIP V3` unchanged for rescue victims
3. if we want to push `M-hot` for fallback reduction, target the
   `mainline-adjacent slip buckets`
4. do not reuse `next_output / immediate|near` as the main fallback-time M-hot
   target

This is the current frozen B-hot interpretation.
