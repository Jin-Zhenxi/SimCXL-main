# A-hot Provisional Rules

## Purpose

This note freezes the current A-side conclusion:

- `A-hot` evidence exists
- but only at the teacher side
- runtime `A-hot` admission is not ready yet

## Evidence Base

- [mhot_oracle_a_hot_rules.csv](/home/jzx8091/SimCXL-main/my_outputData/mhot_oracle_a_hot_rules.csv)
- [fallback_oracle_a_pressure.csv](/home/jzx8091/SimCXL-main/my_outputData/fallback_oracle_a_pressure.csv)
- [mhot_oracle_summary.md](/home/jzx8091/SimCXL-main/my_outputData/mhot_oracle_summary.md)

## Current Stable A-side Signals

The strongest teacher-side A signals are:

- `recurrence / repeat`
- `distance / immediate`
- `distance / near`
- `reuse / multi`

Current aggregated remote-side evidence:

- `recurrence/repeat = 4752`
- `distance/immediate = 3021`
- `distance/near = 1731`
- `reuse/multi = 1731`

This means:

- there are recurring A-side remote-pressure objects
- they often appear again at short distance
- a meaningful subset has multi-reuse

## Current Provisional A-hot Rule

The current A-hot teacher-only rule is:

- `repeat + short distance + multi reuse`

Or expressed slightly more explicitly:

- repeated A object
- next use is `immediate` or `near`
- future `j-tile` reuse is more than one-shot

## Why A-hot Is Still Provisional

We do **not** yet have runtime-validated evidence that enabling A-side hot
admission produces stable local-hit benefit.

Current status:

- teacher evidence: yes
- runtime local-hit proof: no

Therefore:

- keep A labels / metadata ideas
- keep A-side analysis alive
- do not enable symmetric runtime `A-hot` admission yet

## What This Means for Current Runtime Work

The runtime path should stay:

- `VIP V3`: B-side rescue only
- `M-hot`: B-side only for now

A-side should remain:

- analysis-only
- interface-ready if needed later
- not enabled as an admission policy

## Upgrade Condition

We should only promote A-hot from provisional to runtime when we have all three:

1. stable multi-workload teacher-side signal
2. a clean runtime rule based on observables
3. measured local-hit or end-to-end benefit that does not cannibalize the B path

Until then, this document should be treated as the frozen A-hot position.
