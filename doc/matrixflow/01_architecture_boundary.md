# MatrixFlow Architecture Boundary

## Purpose

This note freezes the current architecture split so later runtime experiments do
not drift across roles.

The current system is **not** a single cache hierarchy. It is a tagged,
role-separated nearby-data structure:

1. `Mainline direct path`
2. `VIP V3 rescue`
3. `M-hot`

## Ordering Rule

The required ordering is:

1. Try the direct mainline path first.
2. Only if the object would otherwise go `remote/fallback`, consult nearby pools
   by object label.
3. If the label is `vip_rescue`, consult `VIP V3`.
4. If the label is `mhot_hot`, consult `M-hot`.
5. If neither nearby pool hits, perform the remote fetch.

This means:

- Not `mainline > M-hot > VIP > fallback`
- But:
  - `mainline` first
  - fallback-time label dispatch second

## Role Split

### Mainline Direct Path

This remains the highest-priority path.

For B-side it already includes the established future-B chain:

- `next_output`
- direct `current-ready`
- boundary handoff
- hierarchical protected-B / claim-based hole filling where enabled

These objects must not be diverted to a nearby pool before the direct path has a
chance to work.

### VIP V3

`VIP V3` stays frozen as:

- `B-side rescue pool`
- small
- hard
- for repeated / recurrence-aware fallback victims

Its current job is:

- rescue rows that already fell into fallback
- help repeated victims
- stay separate from mainline hot-object management

`VIP V3` is **not**:

- a mainline cache
- a generic hot pool
- an A/B shared hotness layer

### M-hot

`M-hot` is the larger hot SRAM concept.

Architecturally it is intended to become:

- `A-hot + B-hot`

But the current phase only commits to:

- define `B-hot` rules clearly
- keep `A-hot` teacher-side and provisional

The intended role of `M-hot` is:

- retain hot objects longer than VIP
- service hot objects, not rescue victims
- remain separate from the rescue role

## Current Boundary Between VIP and M-hot

Using the current Oracle outputs:

- `VIP-like victim count = 4679`
- `M-hot-like hot count = 4953`
- `overlap = 0`

Source:

- [mhot_oracle_boundary_stats.csv](/home/jzx8091/SimCXL-main/my_outputData/mhot_oracle_boundary_stats.csv)

This is strong evidence that the current design should stay split:

- `VIP` continues to own rescue victims
- `M-hot` will own hot objects

## Current Design Rule

Until new evidence says otherwise:

- do not turn `M-hot` into a mainline front-door
- do not let `VIP` absorb mainline hot objects
- do not enable `A-hot` runtime admission yet

The next runtime work should be judged against this boundary, not against a
unified-cache mental model.
