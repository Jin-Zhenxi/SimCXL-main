from m5.objects.ClockedObject import ClockedObject
from m5.params import *
from m5.proxy import *


class MatrixFlowEngine(ClockedObject):
    type = "MatrixFlowEngine"
    cxx_header = "mem/matrixflow_engine.hh"
    cxx_class = "gem5::MatrixFlowEngine"

    system = Param.System(Parent.any, "System this engine belongs to")

    dma_port = RequestPort("DMA request port for MatrixFlowEngine")

    mac_array_size = Param.Unsigned(16, "MAC array dimension (N for N x N)")
    compute_latency_per_op = Param.Cycles(
        1, "Compute latency in cycles per operation chunk"
    )
    min_read_request_bytes = Param.Unsigned(
        64, "Minimum DMA read request granularity for MatrixFlow edge rows"
    )
    next_prefetch_mode = Param.String(
        "b_only",
        "Next-tile prefetch mode: none, b_only, a_b",
    )
    next_prefetch_trigger = Param.String(
        "b_ready",
        "Prefetch trigger point: compute_launch, b_ready, or a_ready",
    )
    next_prefetch_rows_a = Param.Unsigned(
        0,
        "Max A rows to prefetch for the next tile (0 means full tile)",
    )
    next_prefetch_rows_b = Param.Unsigned(
        0,
        "Max B rows to prefetch for the next tile (0 means full tile)",
    )
    carry_over_max_rows = Param.Unsigned(
        0,
        "Max rows inherited across boundary for carry-over (0 means all available)",
    )
    carry_over_inherit_inflight = Param.Bool(
        True,
        "Whether inflight next-output rows remain owned by carry-over after boundary",
    )
    hole_fill_lead_rows = Param.Unsigned(
        0,
        "Max additional rows beyond the carry-over covered prefix that normal fetch may fill while carry-over is still active (0 means unlimited)",
    )
    writec_overlap_b_issue_budget_rows = Param.Unsigned(
        0,
        "Max new B rows that WriteC overlap may issue while keeping continuation alive (0 means disabled by default)",
    )
    body_interior_writeback_stripe_rows = Param.Unsigned(
        0,
        "Stripe row window for early body-interior writeback scheduling (0 means legacy unlimited window)",
    )
    body_interior_writeback_max_outstanding_stripes = Param.Unsigned(
        0,
        "Max outstanding body-interior writeback stripes allowed in the scheduler window (0 means legacy unlimited window)",
    )
    boundary_right_writeback_bytes = Param.Unsigned(
        4,
        "Right-boundary final writeback request width in bytes (default preserves legacy 4B single-element writeback)",
    )
    vip_b_rows_capacity = Param.Unsigned(
        0,
        "Capacity of the guaranteed-local VIP pool measured in row slots (0 means disabled)",
    )
    mhot_b_rows_capacity = Param.Unsigned(
        0,
        "Capacity of the Matrix-side B-mainline hot pool measured in row slots (0 means disabled)",
    )
    coverage_shadow_rows_capacity = Param.Unsigned(
        0,
        "Capacity of the dedicated coverage-shadow SRAM slice measured in row slots (0 means disabled)",
    )
    coverage_gather_min_issue_budget = Param.Unsigned(
        0,
        "Minimum gather issue budget per tile (0 means no minimum guarantee)",
    )
    ab_scheduler_mode = Param.String(
        "baseline",
        "A/B fetch scheduler mode: baseline, lightweight, fullscore, protected_b, hierarchical_protected_b, hierarchical_claim_hole_filling, hierarchical_claim_vip_pool, vip_oracle_guided_single_run, vip_oracle_guided_mainline_only, vip_rescue_buffer_single_run, b_vip_rescue_anti_dead_block_admission, b_vip_rescue_recurrence_aware_v2, b_vip_rescue_recurrence_aware_v3, b_mhot_mainline_default, b_mhot_mainline_enhanced, b_mhot_runtime_first_cut, b_mhot_gap_aware_next_cut, b_mhot_coverage_blindspot_candidate_first_cut, b_mhot_coverage_blindspot_candidate_v2, b_coverage_shadow_controller_first_cut, b_coverage_shadow_controller_v2, b_coverage_2d_gather_first_cut, b_coverage_2d_gather_v2, b_coverage_2d_gather_no_starvation_v3, b_coverage_2d_gather_min_guarantee_first_cut, b_coverage_2d_gather_pingpong_first_cut, ab_smart_pattern_prefetch_first_cut, vip_ab_rescue_buffer_single_run, or dual_rx_ab_hierarchical_vip",
    )
    ab_a_min_credit_rows = Param.Unsigned(
        16,
        "Minimum number of A rows that lightweight/fullscore schedulers try to seed before B can dominate",
    )
    ab_bias_b = Param.Int(
        1,
        "Fixed B-path bias used by the A/B criticality scheduler",
    )
    ab_weight_urgency = Param.Int(
        4,
        "Weight of urgency in the fullscore A/B scheduler",
    )
    ab_weight_deficit = Param.Int(
        3,
        "Weight of deficit in the fullscore A/B scheduler",
    )
    ab_weight_reuse = Param.Int(
        1,
        "Weight of reuse in the fullscore A/B scheduler",
    )
    ab_weight_fallback_risk = Param.Int(
        2,
        "Weight of fallback risk in the fullscore A/B scheduler",
    )
    ab_min_launch_rows_a = Param.Unsigned(
        0,
        "Minimum ready A rows needed to launch compute in parallel A/B modes (0 means mac-array-sized window)",
    )
    ab_min_launch_rows_b = Param.Unsigned(
        0,
        "Minimum ready B rows needed to launch compute in parallel A/B modes (0 means mac-array-sized window)",
    )
