#!/usr/bin/env python3
"""
ViT-inspired layer proxy sweep for the CXL Type-3 + device-side DDR5 HDM branch.
This is not a full-model ViT runtime.
"""
import csv
import math
import os
import re
import shutil
import subprocess
from collections import Counter

# ================= 配置区 =================
GEM5_ROOT = "/home/jzx8091/SimCXL-main"
PRESETS = [
    {
        "name": "ViT-Base-like",
        "seq_len": 197,
        "hidden_dim": 768,
        "mlp_dim": 3072,
        "num_heads": 12,
    },
    {
        "name": "ViT-Large-like",
        "seq_len": 257,
        "hidden_dim": 1024,
        "mlp_dim": 4096,
        "num_heads": 16,
    },
]
OUTPUT_DIR = "my_outputData"
C_FILE = "trigger_gemm.c"
CSV_FILE = os.path.join(OUTPUT_DIR, "roofline_data.csv")
RAW_TXT_FILE = os.path.join(OUTPUT_DIR, "summary_raw.txt")
DEFAULT_PHASE2_MODE = "staged_block"
DEFAULT_STAGED_BLOCK_BYTES = 1024
DEFAULT_MIN_READ_REQUEST_BYTES = 256
MAX_INVALID_RUN_RETRIES = 3
FORCE_RERUN = True
RUN_ONLY_PRESET_NAMES = {"ViT-Large-like"}
RUN_ONLY_LABELS = {
    "ab_hierarchical_claim_hole_filling",
    "b_vip_rescue_recurrence_aware_v3",
    "b_mhot_gap_aware_next_cut",
    "b_coverage_2d_gather_v2",
    "b_coverage_2d_gather_no_starvation_v3",
    "b_coverage_2d_gather_min_guarantee_first_cut",
    "b_coverage_2d_gather_pingpong_first_cut",
}
PREFETCH_CONFIGS = [
    {
        "label": "baseline_serial",
        "mode": "b_only",
        "trigger": "compute_launch",
        "rows_a": 0,
        "rows_b": 32,
        "rows_b_by_preset": {
            "ViT-Base-like": 48,
            "ViT-Large-like": 128,
        },
        "carry_over_max_rows": 0,
        "carry_over_inherit_inflight": 0,
        "hole_fill_lead_rows": 32,
        "writec_overlap_b_issue_budget_rows": 0,
        "ab_scheduler_mode": "baseline",
        "ab_a_min_credit_rows": 16,
        "ab_bias_b": 1,
        "ab_weight_urgency": 4,
        "ab_weight_deficit": 3,
        "ab_weight_reuse": 1,
        "ab_weight_fallback_risk": 2,
        "ab_min_launch_rows_a": 0,
        "ab_min_launch_rows_b": 0,
    },
    {
        "label": "ab_lightweight",
        "mode": "b_only",
        "trigger": "compute_launch",
        "rows_a": 0,
        "rows_b": 32,
        "rows_b_by_preset": {
            "ViT-Base-like": 48,
            "ViT-Large-like": 128,
        },
        "carry_over_max_rows": 0,
        "carry_over_inherit_inflight": 0,
        "hole_fill_lead_rows": 32,
        "writec_overlap_b_issue_budget_rows": 0,
        "ab_scheduler_mode": "lightweight",
        "ab_a_min_credit_rows": 16,
        "ab_bias_b": 1,
        "ab_weight_urgency": 4,
        "ab_weight_deficit": 3,
        "ab_weight_reuse": 1,
        "ab_weight_fallback_risk": 2,
        "ab_min_launch_rows_a": 0,
        "ab_min_launch_rows_b": 0,
    },
    {
        "label": "ab_fullscore",
        "mode": "b_only",
        "trigger": "compute_launch",
        "rows_a": 0,
        "rows_b": 32,
        "rows_b_by_preset": {
            "ViT-Base-like": 48,
            "ViT-Large-like": 128,
        },
        "carry_over_max_rows": 0,
        "carry_over_inherit_inflight": 0,
        "hole_fill_lead_rows": 32,
        "writec_overlap_b_issue_budget_rows": 0,
        "ab_scheduler_mode": "fullscore",
        "ab_a_min_credit_rows": 16,
        "ab_bias_b": 1,
        "ab_weight_urgency": 4,
        "ab_weight_deficit": 3,
        "ab_weight_reuse": 1,
        "ab_weight_fallback_risk": 2,
        "ab_min_launch_rows_a": 0,
        "ab_min_launch_rows_b": 0,
    },
    {
        "label": "ab_protected_b",
        "mode": "b_only",
        "trigger": "compute_launch",
        "rows_a": 0,
        "rows_b": 32,
        "rows_b_by_preset": {
            "ViT-Base-like": 48,
            "ViT-Large-like": 128,
        },
        "carry_over_max_rows": 0,
        "carry_over_inherit_inflight": 0,
        "hole_fill_lead_rows": 32,
        "writec_overlap_b_issue_budget_rows": 0,
        "ab_scheduler_mode": "protected_b",
        "ab_a_min_credit_rows": 16,
        "ab_bias_b": 1,
        "ab_weight_urgency": 4,
        "ab_weight_deficit": 3,
        "ab_weight_reuse": 1,
        "ab_weight_fallback_risk": 2,
        "ab_min_launch_rows_a": 0,
        "ab_min_launch_rows_b": 0,
        "ab_current_protected_b_quota_rows": 4,
    },
    {
        "label": "ab_hierarchical_protected_b",
        "mode": "b_only",
        "trigger": "compute_launch",
        "rows_a": 0,
        "rows_b": 32,
        "rows_b_by_preset": {
            "ViT-Base-like": 48,
            "ViT-Large-like": 128,
        },
        "carry_over_max_rows": 0,
        "carry_over_inherit_inflight": 0,
        "hole_fill_lead_rows": 32,
        "writec_overlap_b_issue_budget_rows": 0,
        "ab_scheduler_mode": "hierarchical_protected_b",
        "ab_a_min_credit_rows": 16,
        "ab_bias_b": 1,
        "ab_weight_urgency": 4,
        "ab_weight_deficit": 3,
        "ab_weight_reuse": 1,
        "ab_weight_fallback_risk": 2,
        "ab_min_launch_rows_a": 0,
        "ab_min_launch_rows_b": 0,
        "ab_current_protected_b_quota_rows": 4,
    },
    {
        "label": "ab_hierarchical_claim_hole_filling",
        "mode": "b_only",
        "trigger": "compute_launch",
        "rows_a": 0,
        "rows_b": 32,
        "rows_b_by_preset": {
            "ViT-Base-like": 48,
            "ViT-Large-like": 128,
        },
        "carry_over_max_rows": 0,
        "carry_over_inherit_inflight": 0,
        "hole_fill_lead_rows": 32,
        "writec_overlap_b_issue_budget_rows": 0,
        "ab_scheduler_mode": "hierarchical_claim_hole_filling",
        "ab_a_min_credit_rows": 16,
        "ab_bias_b": 1,
        "ab_weight_urgency": 4,
        "ab_weight_deficit": 3,
        "ab_weight_reuse": 1,
        "ab_weight_fallback_risk": 2,
        "ab_min_launch_rows_a": 0,
        "ab_min_launch_rows_b": 0,
        "ab_current_protected_b_quota_rows": 4,
    },
    {
        "label": "ab_hierarchical_claim_vip_pool",
        "mode": "b_only",
        "trigger": "compute_launch",
        "rows_a": 0,
        "rows_b": 32,
        "rows_b_by_preset": {
            "ViT-Base-like": 48,
            "ViT-Large-like": 128,
        },
        "carry_over_max_rows": 0,
        "carry_over_inherit_inflight": 0,
        "hole_fill_lead_rows": 32,
        "writec_overlap_b_issue_budget_rows": 0,
        "vip_b_rows_capacity": 64,
        "ab_scheduler_mode": "hierarchical_claim_vip_pool",
        "ab_a_min_credit_rows": 16,
        "ab_bias_b": 1,
        "ab_weight_urgency": 4,
        "ab_weight_deficit": 3,
        "ab_weight_reuse": 1,
        "ab_weight_fallback_risk": 2,
        "ab_min_launch_rows_a": 0,
        "ab_min_launch_rows_b": 0,
        "ab_current_protected_b_quota_rows": 4,
    },
    {
        "label": "dual_rx_ab_hierarchical_vip",
        "mode": "b_only",
        "trigger": "compute_launch",
        "rows_a": 0,
        "rows_b": 32,
        "rows_b_by_preset": {
            "ViT-Base-like": 48,
            "ViT-Large-like": 128,
        },
        "carry_over_max_rows": 0,
        "carry_over_inherit_inflight": 0,
        "hole_fill_lead_rows": 32,
        "writec_overlap_b_issue_budget_rows": 0,
        "ab_scheduler_mode": "dual_rx_ab_hierarchical_vip",
        "ab_a_min_credit_rows": 16,
        "ab_bias_b": 1,
        "ab_weight_urgency": 4,
        "ab_weight_deficit": 3,
        "ab_weight_reuse": 1,
        "ab_weight_fallback_risk": 2,
        "ab_min_launch_rows_a": 0,
        "ab_min_launch_rows_b": 0,
        "ab_current_protected_b_quota_rows": 4,
        "vip_b_rows_capacity": 0,
    },
    {
        "label": "vip_oracle_guided_single_run",
        "mode": "b_only",
        "trigger": "compute_launch",
        "rows_a": 0,
        "rows_b": 32,
        "rows_b_by_preset": {
            "ViT-Base-like": 48,
            "ViT-Large-like": 128,
        },
        "carry_over_max_rows": 0,
        "carry_over_inherit_inflight": 0,
        "hole_fill_lead_rows": 32,
        "writec_overlap_b_issue_budget_rows": 0,
        "vip_b_rows_capacity": 64,
        "ab_scheduler_mode": "vip_oracle_guided_single_run",
        "ab_a_min_credit_rows": 16,
        "ab_bias_b": 1,
        "ab_weight_urgency": 4,
        "ab_weight_deficit": 3,
        "ab_weight_reuse": 1,
        "ab_weight_fallback_risk": 2,
        "ab_min_launch_rows_a": 0,
        "ab_min_launch_rows_b": 0,
        "ab_current_protected_b_quota_rows": 4,
    },
    {
        "label": "vip_oracle_guided_mainline_only_64",
        "mode": "b_only",
        "trigger": "compute_launch",
        "rows_a": 0,
        "rows_b": 32,
        "rows_b_by_preset": {
            "ViT-Base-like": 48,
            "ViT-Large-like": 128,
        },
        "carry_over_max_rows": 0,
        "carry_over_inherit_inflight": 0,
        "hole_fill_lead_rows": 32,
        "writec_overlap_b_issue_budget_rows": 0,
        "vip_b_rows_capacity": 64,
        "ab_scheduler_mode": "vip_oracle_guided_mainline_only",
        "ab_a_min_credit_rows": 16,
        "ab_bias_b": 1,
        "ab_weight_urgency": 4,
        "ab_weight_deficit": 3,
        "ab_weight_reuse": 1,
        "ab_weight_fallback_risk": 2,
        "ab_min_launch_rows_a": 0,
        "ab_min_launch_rows_b": 0,
        "ab_current_protected_b_quota_rows": 4,
    },
    {
        "label": "vip_oracle_guided_mainline_only_128",
        "mode": "b_only",
        "trigger": "compute_launch",
        "rows_a": 0,
        "rows_b": 32,
        "rows_b_by_preset": {
            "ViT-Base-like": 48,
            "ViT-Large-like": 128,
        },
        "carry_over_max_rows": 0,
        "carry_over_inherit_inflight": 0,
        "hole_fill_lead_rows": 32,
        "writec_overlap_b_issue_budget_rows": 0,
        "vip_b_rows_capacity": 128,
        "ab_scheduler_mode": "vip_oracle_guided_mainline_only",
        "ab_a_min_credit_rows": 16,
        "ab_bias_b": 1,
        "ab_weight_urgency": 4,
        "ab_weight_deficit": 3,
        "ab_weight_reuse": 1,
        "ab_weight_fallback_risk": 2,
        "ab_min_launch_rows_a": 0,
        "ab_min_launch_rows_b": 0,
        "ab_current_protected_b_quota_rows": 4,
    },
    {
        "label": "vip_oracle_guided_mainline_only_256",
        "mode": "b_only",
        "trigger": "compute_launch",
        "rows_a": 0,
        "rows_b": 32,
        "rows_b_by_preset": {
            "ViT-Base-like": 48,
            "ViT-Large-like": 128,
        },
        "carry_over_max_rows": 0,
        "carry_over_inherit_inflight": 0,
        "hole_fill_lead_rows": 32,
        "writec_overlap_b_issue_budget_rows": 0,
        "vip_b_rows_capacity": 256,
        "ab_scheduler_mode": "vip_oracle_guided_mainline_only",
        "ab_a_min_credit_rows": 16,
        "ab_bias_b": 1,
        "ab_weight_urgency": 4,
        "ab_weight_deficit": 3,
        "ab_weight_reuse": 1,
        "ab_weight_fallback_risk": 2,
        "ab_min_launch_rows_a": 0,
        "ab_min_launch_rows_b": 0,
        "ab_current_protected_b_quota_rows": 4,
    },
    {
        "label": "vip_rescue_buffer_single_run",
        "mode": "b_only",
        "trigger": "compute_launch",
        "rows_a": 0,
        "rows_b": 32,
        "rows_b_by_preset": {
            "ViT-Base-like": 48,
            "ViT-Large-like": 128,
        },
        "carry_over_max_rows": 0,
        "carry_over_inherit_inflight": 0,
        "hole_fill_lead_rows": 32,
        "writec_overlap_b_issue_budget_rows": 0,
        "vip_b_rows_capacity": 64,
        "ab_scheduler_mode": "vip_rescue_buffer_single_run",
        "ab_a_min_credit_rows": 16,
        "ab_bias_b": 1,
        "ab_weight_urgency": 4,
        "ab_weight_deficit": 3,
        "ab_weight_reuse": 1,
        "ab_weight_fallback_risk": 2,
        "ab_min_launch_rows_a": 0,
        "ab_min_launch_rows_b": 0,
        "ab_current_protected_b_quota_rows": 4,
    },
    {
        "label": "b_vip_rescue_anti_dead_block_admission",
        "mode": "b_only",
        "trigger": "compute_launch",
        "rows_a": 0,
        "rows_b": 32,
        "rows_b_by_preset": {
            "ViT-Base-like": 48,
            "ViT-Large-like": 128,
        },
        "carry_over_max_rows": 0,
        "carry_over_inherit_inflight": 0,
        "hole_fill_lead_rows": 32,
        "writec_overlap_b_issue_budget_rows": 0,
        "vip_b_rows_capacity": 64,
        "ab_scheduler_mode": "b_vip_rescue_anti_dead_block_admission",
        "ab_a_min_credit_rows": 16,
        "ab_bias_b": 1,
        "ab_weight_urgency": 4,
        "ab_weight_deficit": 3,
        "ab_weight_reuse": 1,
        "ab_weight_fallback_risk": 2,
        "ab_min_launch_rows_a": 0,
        "ab_min_launch_rows_b": 0,
        "ab_current_protected_b_quota_rows": 4,
    },
    {
        "label": "b_vip_rescue_recurrence_aware_v2",
        "mode": "b_only",
        "trigger": "compute_launch",
        "rows_a": 0,
        "rows_b": 32,
        "rows_b_by_preset": {
            "ViT-Base-like": 48,
            "ViT-Large-like": 128,
        },
        "carry_over_max_rows": 0,
        "carry_over_inherit_inflight": 0,
        "hole_fill_lead_rows": 32,
        "writec_overlap_b_issue_budget_rows": 0,
        "vip_b_rows_capacity": 64,
        "ab_scheduler_mode": "b_vip_rescue_recurrence_aware_v2",
        "ab_a_min_credit_rows": 16,
        "ab_bias_b": 1,
        "ab_weight_urgency": 4,
        "ab_weight_deficit": 3,
        "ab_weight_reuse": 1,
        "ab_weight_fallback_risk": 2,
        "ab_min_launch_rows_a": 0,
        "ab_min_launch_rows_b": 0,
        "ab_current_protected_b_quota_rows": 4,
    },
    {
        "label": "b_vip_rescue_recurrence_aware_v3",
        "mode": "b_only",
        "trigger": "compute_launch",
        "rows_a": 0,
        "rows_b": 32,
        "rows_b_by_preset": {
            "ViT-Base-like": 48,
            "ViT-Large-like": 128,
        },
        "carry_over_max_rows": 0,
        "carry_over_inherit_inflight": 0,
        "hole_fill_lead_rows": 32,
        "writec_overlap_b_issue_budget_rows": 0,
        "vip_b_rows_capacity": 64,
        "ab_scheduler_mode": "b_vip_rescue_recurrence_aware_v3",
        "ab_a_min_credit_rows": 16,
        "ab_bias_b": 1,
        "ab_weight_urgency": 4,
        "ab_weight_deficit": 3,
        "ab_weight_reuse": 1,
        "ab_weight_fallback_risk": 2,
        "ab_min_launch_rows_a": 0,
        "ab_min_launch_rows_b": 0,
        "ab_current_protected_b_quota_rows": 4,
    },
    {
        "label": "b_mhot_mainline_default",
        "mode": "b_only",
        "trigger": "compute_launch",
        "rows_a": 0,
        "rows_b": 32,
        "rows_b_by_preset": {
            "ViT-Base-like": 48,
            "ViT-Large-like": 128,
        },
        "carry_over_max_rows": 0,
        "carry_over_inherit_inflight": 0,
        "hole_fill_lead_rows": 32,
        "writec_overlap_b_issue_budget_rows": 0,
        "vip_b_rows_capacity": 64,
        "mhot_b_rows_capacity": 128,
        "ab_scheduler_mode": "b_mhot_mainline_default",
        "ab_a_min_credit_rows": 16,
        "ab_bias_b": 1,
        "ab_weight_urgency": 4,
        "ab_weight_deficit": 3,
        "ab_weight_reuse": 1,
        "ab_weight_fallback_risk": 2,
        "ab_min_launch_rows_a": 0,
        "ab_min_launch_rows_b": 0,
        "ab_current_protected_b_quota_rows": 4,
    },
    {
        "label": "b_mhot_mainline_enhanced",
        "mode": "b_only",
        "trigger": "compute_launch",
        "rows_a": 0,
        "rows_b": 32,
        "rows_b_by_preset": {
            "ViT-Base-like": 48,
            "ViT-Large-like": 128,
        },
        "carry_over_max_rows": 0,
        "carry_over_inherit_inflight": 0,
        "hole_fill_lead_rows": 32,
        "writec_overlap_b_issue_budget_rows": 0,
        "vip_b_rows_capacity": 64,
        "mhot_b_rows_capacity": 128,
        "ab_scheduler_mode": "b_mhot_mainline_enhanced",
        "ab_a_min_credit_rows": 16,
        "ab_bias_b": 1,
        "ab_weight_urgency": 4,
        "ab_weight_deficit": 3,
        "ab_weight_reuse": 1,
        "ab_weight_fallback_risk": 2,
        "ab_min_launch_rows_a": 0,
        "ab_min_launch_rows_b": 0,
        "ab_current_protected_b_quota_rows": 4,
    },
    {
        "label": "b_mhot_runtime_first_cut",
        "mode": "b_only",
        "trigger": "compute_launch",
        "rows_a": 0,
        "rows_b": 32,
        "rows_b_by_preset": {
            "ViT-Base-like": 48,
            "ViT-Large-like": 128,
        },
        "carry_over_max_rows": 0,
        "carry_over_inherit_inflight": 0,
        "hole_fill_lead_rows": 32,
        "writec_overlap_b_issue_budget_rows": 0,
        "vip_b_rows_capacity": 64,
        "mhot_b_rows_capacity": 128,
        "ab_scheduler_mode": "b_mhot_runtime_first_cut",
        "ab_a_min_credit_rows": 16,
        "ab_bias_b": 1,
        "ab_weight_urgency": 4,
        "ab_weight_deficit": 3,
        "ab_weight_reuse": 1,
        "ab_weight_fallback_risk": 2,
        "ab_min_launch_rows_a": 0,
        "ab_min_launch_rows_b": 0,
        "ab_current_protected_b_quota_rows": 4,
    },
    {
        "label": "b_mhot_gap_aware_next_cut",
        "mode": "b_only",
        "trigger": "compute_launch",
        "rows_a": 0,
        "rows_b": 32,
        "rows_b_by_preset": {
            "ViT-Base-like": 48,
            "ViT-Large-like": 128,
        },
        "carry_over_max_rows": 0,
        "carry_over_inherit_inflight": 0,
        "hole_fill_lead_rows": 32,
        "writec_overlap_b_issue_budget_rows": 0,
        "vip_b_rows_capacity": 64,
        "mhot_b_rows_capacity": 128,
        "ab_scheduler_mode": "b_mhot_gap_aware_next_cut",
        "ab_a_min_credit_rows": 16,
        "ab_bias_b": 1,
        "ab_weight_urgency": 4,
        "ab_weight_deficit": 3,
        "ab_weight_reuse": 1,
        "ab_weight_fallback_risk": 2,
        "ab_min_launch_rows_a": 0,
        "ab_min_launch_rows_b": 0,
        "ab_current_protected_b_quota_rows": 4,
    },
    {
        "label": "b_mhot_coverage_blindspot_candidate_first_cut",
        "mode": "b_only",
        "trigger": "compute_launch",
        "rows_a": 0,
        "rows_b": 32,
        "rows_b_by_preset": {
            "ViT-Base-like": 48,
            "ViT-Large-like": 128,
        },
        "carry_over_max_rows": 0,
        "carry_over_inherit_inflight": 0,
        "hole_fill_lead_rows": 32,
        "writec_overlap_b_issue_budget_rows": 0,
        "vip_b_rows_capacity": 64,
        "mhot_b_rows_capacity": 128,
        "ab_scheduler_mode": "b_mhot_coverage_blindspot_candidate_first_cut",
        "ab_a_min_credit_rows": 16,
        "ab_bias_b": 1,
        "ab_weight_urgency": 4,
        "ab_weight_deficit": 3,
        "ab_weight_reuse": 1,
        "ab_weight_fallback_risk": 2,
        "ab_min_launch_rows_a": 0,
        "ab_min_launch_rows_b": 0,
        "ab_current_protected_b_quota_rows": 4,
    },
    {
        "label": "b_mhot_coverage_blindspot_candidate_v2",
        "mode": "b_only",
        "trigger": "compute_launch",
        "rows_a": 0,
        "rows_b": 32,
        "rows_b_by_preset": {
            "ViT-Base-like": 48,
            "ViT-Large-like": 128,
        },
        "carry_over_max_rows": 0,
        "carry_over_inherit_inflight": 0,
        "hole_fill_lead_rows": 32,
        "writec_overlap_b_issue_budget_rows": 0,
        "vip_b_rows_capacity": 64,
        "mhot_b_rows_capacity": 128,
        "ab_scheduler_mode": "b_mhot_coverage_blindspot_candidate_v2",
        "ab_a_min_credit_rows": 16,
        "ab_bias_b": 1,
        "ab_weight_urgency": 4,
        "ab_weight_deficit": 3,
        "ab_weight_reuse": 1,
        "ab_weight_fallback_risk": 2,
        "ab_min_launch_rows_a": 0,
        "ab_min_launch_rows_b": 0,
        "ab_current_protected_b_quota_rows": 4,
    },
    {
        "label": "b_coverage_shadow_controller_first_cut",
        "mode": "b_only",
        "trigger": "compute_launch",
        "rows_a": 0,
        "rows_b": 32,
        "rows_b_by_preset": {
            "ViT-Base-like": 48,
            "ViT-Large-like": 128,
        },
        "carry_over_max_rows": 0,
        "carry_over_inherit_inflight": 0,
        "hole_fill_lead_rows": 32,
        "writec_overlap_b_issue_budget_rows": 0,
        "vip_b_rows_capacity": 64,
        "mhot_b_rows_capacity": 128,
        "coverage_shadow_rows_capacity": 128,
        "ab_scheduler_mode": "b_coverage_shadow_controller_first_cut",
        "ab_a_min_credit_rows": 16,
        "ab_bias_b": 1,
        "ab_weight_urgency": 4,
        "ab_weight_deficit": 3,
        "ab_weight_reuse": 1,
        "ab_weight_fallback_risk": 2,
        "ab_min_launch_rows_a": 0,
        "ab_min_launch_rows_b": 0,
        "ab_current_protected_b_quota_rows": 4,
    },
    {
        "label": "b_coverage_shadow_controller_v2",
        "mode": "b_only",
        "trigger": "compute_launch",
        "rows_a": 0,
        "rows_b": 32,
        "rows_b_by_preset": {
            "ViT-Base-like": 48,
            "ViT-Large-like": 128,
        },
        "carry_over_max_rows": 0,
        "carry_over_inherit_inflight": 0,
        "hole_fill_lead_rows": 32,
        "writec_overlap_b_issue_budget_rows": 0,
        "vip_b_rows_capacity": 64,
        "mhot_b_rows_capacity": 128,
        "coverage_shadow_rows_capacity": 128,
        "ab_scheduler_mode": "b_coverage_shadow_controller_v2",
        "ab_a_min_credit_rows": 16,
        "ab_bias_b": 1,
        "ab_weight_urgency": 4,
        "ab_weight_deficit": 3,
        "ab_weight_reuse": 1,
        "ab_weight_fallback_risk": 2,
        "ab_min_launch_rows_a": 0,
        "ab_min_launch_rows_b": 0,
        "ab_current_protected_b_quota_rows": 4,
    },
    {
        "label": "b_coverage_2d_gather_first_cut",
        "mode": "b_only",
        "trigger": "compute_launch",
        "rows_a": 0,
        "rows_b": 32,
        "rows_b_by_preset": {
            "ViT-Base-like": 48,
            "ViT-Large-like": 128,
        },
        "carry_over_max_rows": 0,
        "carry_over_inherit_inflight": 0,
        "hole_fill_lead_rows": 32,
        "writec_overlap_b_issue_budget_rows": 0,
        "vip_b_rows_capacity": 64,
        "mhot_b_rows_capacity": 128,
        "coverage_shadow_rows_capacity": 128,
        "ab_scheduler_mode": "b_coverage_2d_gather_first_cut",
        "ab_a_min_credit_rows": 16,
        "ab_bias_b": 1,
        "ab_weight_urgency": 4,
        "ab_weight_deficit": 3,
        "ab_weight_reuse": 1,
        "ab_weight_fallback_risk": 2,
        "ab_min_launch_rows_a": 0,
        "ab_min_launch_rows_b": 0,
        "ab_current_protected_b_quota_rows": 4,
    },
    {
        "label": "b_coverage_2d_gather_v2",
        "mode": "b_only",
        "trigger": "compute_launch",
        "rows_a": 0,
        "rows_b": 32,
        "rows_b_by_preset": {
            "ViT-Base-like": 48,
            "ViT-Large-like": 128,
        },
        "carry_over_max_rows": 0,
        "carry_over_inherit_inflight": 0,
        "hole_fill_lead_rows": 32,
        "writec_overlap_b_issue_budget_rows": 0,
        "vip_b_rows_capacity": 64,
        "mhot_b_rows_capacity": 128,
        "coverage_shadow_rows_capacity": 128,
        "ab_scheduler_mode": "b_coverage_2d_gather_v2",
        "ab_a_min_credit_rows": 16,
        "ab_bias_b": 1,
        "ab_weight_urgency": 4,
        "ab_weight_deficit": 3,
        "ab_weight_reuse": 1,
        "ab_weight_fallback_risk": 2,
        "ab_min_launch_rows_a": 0,
        "ab_min_launch_rows_b": 0,
        "ab_current_protected_b_quota_rows": 4,
    },
    {
        "label": "b_coverage_2d_gather_no_starvation_v3",
        "mode": "b_only",
        "trigger": "compute_launch",
        "rows_a": 0,
        "rows_b": 32,
        "rows_b_by_preset": {
            "ViT-Base-like": 48,
            "ViT-Large-like": 128,
        },
        "carry_over_max_rows": 0,
        "carry_over_inherit_inflight": 0,
        "hole_fill_lead_rows": 32,
        "writec_overlap_b_issue_budget_rows": 0,
        "vip_b_rows_capacity": 64,
        "mhot_b_rows_capacity": 128,
        "coverage_shadow_rows_capacity": 128,
        "ab_scheduler_mode": "b_coverage_2d_gather_no_starvation_v3",
        "ab_a_min_credit_rows": 16,
        "ab_bias_b": 1,
        "ab_weight_urgency": 4,
        "ab_weight_deficit": 3,
        "ab_weight_reuse": 1,
        "ab_weight_fallback_risk": 2,
        "ab_min_launch_rows_a": 0,
        "ab_min_launch_rows_b": 0,
        "ab_current_protected_b_quota_rows": 4,
    },
    {
        "label": "b_coverage_2d_gather_min_guarantee_first_cut",
        "mode": "b_only",
        "trigger": "compute_launch",
        "rows_a": 0,
        "rows_b": 32,
        "rows_b_by_preset": {
            "ViT-Base-like": 48,
            "ViT-Large-like": 128,
        },
        "carry_over_max_rows": 0,
        "carry_over_inherit_inflight": 0,
        "hole_fill_lead_rows": 32,
        "writec_overlap_b_issue_budget_rows": 0,
        "vip_b_rows_capacity": 64,
        "mhot_b_rows_capacity": 128,
        "coverage_shadow_rows_capacity": 128,
        "coverage_gather_min_issue_budget": 8,
        "ab_scheduler_mode": "b_coverage_2d_gather_min_guarantee_first_cut",
        "ab_a_min_credit_rows": 16,
        "ab_bias_b": 1,
        "ab_weight_urgency": 4,
        "ab_weight_deficit": 3,
        "ab_weight_reuse": 1,
        "ab_weight_fallback_risk": 2,
        "ab_min_launch_rows_a": 0,
        "ab_min_launch_rows_b": 0,
        "ab_current_protected_b_quota_rows": 4,
    },
    {
        "label": "b_coverage_2d_gather_pingpong_first_cut",
        "mode": "b_only",
        "trigger": "compute_launch",
        "rows_a": 0,
        "rows_b": 32,
        "rows_b_by_preset": {
            "ViT-Base-like": 48,
            "ViT-Large-like": 128,
        },
        "carry_over_max_rows": 0,
        "carry_over_inherit_inflight": 0,
        "hole_fill_lead_rows": 32,
        "writec_overlap_b_issue_budget_rows": 0,
        "vip_b_rows_capacity": 64,
        "mhot_b_rows_capacity": 128,
        "coverage_shadow_rows_capacity": 128,
        "ab_scheduler_mode": "b_coverage_2d_gather_pingpong_first_cut",
        "ab_a_min_credit_rows": 16,
        "ab_bias_b": 1,
        "ab_weight_urgency": 4,
        "ab_weight_deficit": 3,
        "ab_weight_reuse": 1,
        "ab_weight_fallback_risk": 2,
        "ab_min_launch_rows_a": 0,
        "ab_min_launch_rows_b": 0,
        "ab_current_protected_b_quota_rows": 4,
    },
    {
        "label": "ab_smart_pattern_prefetch_first_cut",
        "mode": "a_b",
        "trigger": "compute_launch",
        "rows_a": 64,
        "rows_b": 32,
        "rows_b_by_preset": {
            "ViT-Base-like": 48,
            "ViT-Large-like": 128,
        },
        "carry_over_max_rows": 0,
        "carry_over_inherit_inflight": 0,
        "hole_fill_lead_rows": 32,
        "writec_overlap_b_issue_budget_rows": 0,
        "vip_b_rows_capacity": 64,
        "mhot_b_rows_capacity": 128,
        "ab_scheduler_mode": "ab_smart_pattern_prefetch_first_cut",
        "ab_a_min_credit_rows": 16,
        "ab_bias_b": 1,
        "ab_weight_urgency": 4,
        "ab_weight_deficit": 3,
        "ab_weight_reuse": 1,
        "ab_weight_fallback_risk": 2,
        "ab_min_launch_rows_a": 0,
        "ab_min_launch_rows_b": 0,
        "ab_current_protected_b_quota_rows": 4,
    },
    {
        "label": "vip_ab_rescue_buffer_single_run_64",
        "mode": "b_only",
        "trigger": "compute_launch",
        "rows_a": 0,
        "rows_b": 32,
        "rows_b_by_preset": {
            "ViT-Base-like": 48,
            "ViT-Large-like": 128,
        },
        "carry_over_max_rows": 0,
        "carry_over_inherit_inflight": 0,
        "hole_fill_lead_rows": 32,
        "writec_overlap_b_issue_budget_rows": 0,
        "vip_b_rows_capacity": 64,
        "ab_scheduler_mode": "vip_ab_rescue_buffer_single_run",
        "ab_a_min_credit_rows": 16,
        "ab_bias_b": 1,
        "ab_weight_urgency": 4,
        "ab_weight_deficit": 3,
        "ab_weight_reuse": 1,
        "ab_weight_fallback_risk": 2,
        "ab_min_launch_rows_a": 0,
        "ab_min_launch_rows_b": 0,
        "ab_current_protected_b_quota_rows": 4,
    },
    {
        "label": "vip_ab_rescue_buffer_single_run_128",
        "mode": "b_only",
        "trigger": "compute_launch",
        "rows_a": 0,
        "rows_b": 32,
        "rows_b_by_preset": {
            "ViT-Base-like": 48,
            "ViT-Large-like": 128,
        },
        "carry_over_max_rows": 0,
        "carry_over_inherit_inflight": 0,
        "hole_fill_lead_rows": 32,
        "writec_overlap_b_issue_budget_rows": 0,
        "vip_b_rows_capacity": 128,
        "ab_scheduler_mode": "vip_ab_rescue_buffer_single_run",
        "ab_a_min_credit_rows": 16,
        "ab_bias_b": 1,
        "ab_weight_urgency": 4,
        "ab_weight_deficit": 3,
        "ab_weight_reuse": 1,
        "ab_weight_fallback_risk": 2,
        "ab_min_launch_rows_a": 0,
        "ab_min_launch_rows_b": 0,
        "ab_current_protected_b_quota_rows": 4,
    },
]

# 物理时钟频率 (Hz)
CLOCK_FREQ_GHZ = 2.4

# 编译命令：需链接 m5ops 库（libm5.a 需在 util/m5 下 scons build/x86 生成）
COMPILE_CMD = (
    f"gcc -O3 -static -mno-80387 -o trigger_gemm trigger_gemm.c "
    f"-I{GEM5_ROOT}/include -L{GEM5_ROOT}/util/m5/build/x86/out -lm -lm5"
)
INJECT_CMD = "sudo ./inject_trigger_gemm.sh"

# gem5 启动命令模板
GEM5_CMD_TEMPLATE = (
    f"./build/X86/gem5.opt -p {GEM5_ROOT}/src/python "
    "-d {outdir} "
    "configs/example/gem5_library/x86-cxl-type3-with-classic.py "
    "--is_asic True --cpu_type TIMING > {logfile} 2>&1"
)
# =========================================

CHUNK_SEPARATOR = "---------- Begin Simulation Statistics ----------"
LIBM5_PATH = os.path.join(GEM5_ROOT, "util/m5/build/x86/out/libm5.a")


def ensure_libm5():
    """若 libm5.a 不存在，自动在 util/m5 下执行 scons 构建"""
    if os.path.exists(LIBM5_PATH):
        return
    print("[*] libm5.a 未找到，正在自动构建...")
    m5_dir = os.path.join(GEM5_ROOT, "util/m5")
    rc = subprocess.run(
        ["scons", "build/x86/out/libm5.a"],
        cwd=m5_dir,
        capture_output=True,
        text=True,
    )
    if rc.returncode != 0:
        print(rc.stderr or rc.stdout)
        raise SystemExit(
            f"构建 libm5.a 失败。请手动执行: cd {m5_dir} && scons build/x86/out/libm5.a"
        )
    print("[*] libm5.a 构建完成")


def should_run_case(preset, cfg):
    return (
        preset["name"] in RUN_ONLY_PRESET_NAMES
        and cfg["label"] in RUN_ONLY_LABELS
    )


def modify_c_file(preset):
    """写入 ViT-like preset 默认值。"""
    with open(C_FILE) as f:
        content = f.read()

    replacements = {
        r"uint32_t default_seq_len = \d+;": f"uint32_t default_seq_len = {preset['seq_len']};",
        r"uint32_t default_hidden_dim = \d+;": f"uint32_t default_hidden_dim = {preset['hidden_dim']};",
        r"uint32_t default_mlp_dim = \d+;": f"uint32_t default_mlp_dim = {preset['mlp_dim']};",
        r"uint32_t default_num_heads = \d+;": f"uint32_t default_num_heads = {preset['num_heads']};",
        r'const char \*default_phase2_mode = "[^"]+";': f'const char *default_phase2_mode = "{DEFAULT_PHASE2_MODE}";',
        r"uint32_t default_staged_block_bytes = \d+;": f"uint32_t default_staged_block_bytes = {DEFAULT_STAGED_BLOCK_BYTES};",
        r'\.preset_name = "[^"]+",': f'.preset_name = "{preset["name"]}",',
    }

    for pattern, repl in replacements.items():
        content = re.sub(pattern, repl, content)

    with open(C_FILE, "w") as f:
        f.write(content)
    print(
        "[*] 已写入预设: "
        f"{preset['name']} "
        f"(seq_len={preset['seq_len']}, hidden_dim={preset['hidden_dim']}, "
        f"mlp_dim={preset['mlp_dim']}, num_heads={preset['num_heads']})"
    )


def extract_metrics(stats_file, log_file):
    """
    从 stats.txt 中提取纯净 ROI 数据。
    gem5 退出时可能追加空的收尾 stats 块，chunks[-1] 往往无效。
    改为寻找特征块：包含 totalComputeCycles > 0 的那一块（MatrixFlowEngine 真实执行）。
    """
    metrics = {
        "simSeconds": 0.0,
        "dmaRead": 0,
        "dmaWrite": 0,
        "computeCycles": 0,
    }

    if not os.path.exists(stats_file):
        return metrics

    with open(stats_file) as f:
        stats_text = f.read()

    chunks = stats_text.split(CHUNK_SEPARATOR)

    # 寻找特征块：totalComputeCycles > 0 表示 MatrixFlowEngine 真实执行过
    target_chunk = ""
    for chunk in chunks:
        match = re.search(r"totalComputeCycles\s+(\d+)", chunk)
        if match and int(match.group(1)) > 0:
            target_chunk = chunk
            break

    if not target_chunk:
        target_chunk = chunks[-1] if chunks else ""

    target_chunk = target_chunk.strip()
    if not target_chunk:
        return metrics

    # 从特征块中提取
    match_time = re.search(r"simSeconds\s+([0-9.]+)", target_chunk)
    match_read = re.search(r"totalDmaBytesRead\s+(\d+)", target_chunk)
    match_write = re.search(r"totalDmaBytesWritten\s+(\d+)", target_chunk)
    match_cycles = re.search(r"totalComputeCycles\s+(\d+)", target_chunk)

    if match_time:
        metrics["simSeconds"] = float(match_time.group(1))
    if match_read:
        metrics["dmaRead"] = int(match_read.group(1))
    if match_write:
        metrics["dmaWrite"] = int(match_write.group(1))
    if match_cycles:
        metrics["computeCycles"] = int(match_cycles.group(1))

    if os.path.exists(log_file):
        with open(log_file, encoding="utf-8", errors="ignore") as f:
            for line in f:
                if metrics["dmaRead"] == 0:
                    match_read = re.search(r"dmaRead=(\d+)", line)
                    if match_read:
                        metrics["dmaRead"] = int(match_read.group(1))

                if metrics["dmaWrite"] == 0:
                    match_write = re.search(r"dmaWrite=(\d+)", line)
                    if match_write:
                        metrics["dmaWrite"] = int(match_write.group(1))

                if metrics["computeCycles"] == 0:
                    match_cycles = re.search(r"computeCycles=(\d+)", line)
                    if match_cycles:
                        metrics["computeCycles"] = int(match_cycles.group(1))

                if (
                    metrics["dmaRead"] > 0
                    and metrics["dmaWrite"] > 0
                    and metrics["computeCycles"] > 0
                ):
                    break

    return metrics


def extract_bus_dma_stats(stats_file):
    stats = {
        "bus_dma_pkt_count": 0,
        "bus_dma_pkt_bytes": 0,
        "mem_rsp_gap_mean_cycles": 0.0,
        "mem_rsp_gap_stdev_cycles": 0.0,
        "mem_rsp_gap_min_cycles": 0.0,
        "mem_rsp_gap_max_cycles": 0.0,
    }
    if not os.path.exists(stats_file):
        return stats

    patterns = {
        "bus_dma_pkt_count": r"cxl_mem_bus\.pktCount_.*matrix_engine\.dma::total\s+(\d+)",
        "bus_dma_pkt_bytes": r"cxl_mem_bus\.pktSize_.*matrix_engine\.dma::total\s+(\d+)",
        "mem_rsp_gap_mean_cycles": r"memToCXLCtrlRsp::mean\s+([0-9.]+)",
        "mem_rsp_gap_stdev_cycles": r"memToCXLCtrlRsp::stdev\s+([0-9.]+)",
        "mem_rsp_gap_min_cycles": r"memToCXLCtrlRsp::min_value\s+([0-9.]+)",
        "mem_rsp_gap_max_cycles": r"memToCXLCtrlRsp::max_value\s+([0-9.]+)",
    }

    text = open(stats_file, encoding="utf-8", errors="ignore").read()
    for key, pattern in patterns.items():
        m = re.search(pattern, text)
        if not m:
            continue
        stats[key] = (
            float(m.group(1)) if "." in m.group(1) else int(m.group(1))
        )
    return stats


def extract_prefetch_stats(stats_file):
    stats = {
        "next_prefetch_issue_count": 0,
        "next_k_prefetch_issue_count": 0,
        "next_output_prefetch_issue_count": 0,
        "next_prefetch_hit_count": 0,
        "next_output_prefetch_hit_count": 0,
        "next_prefetch_fallback_count": 0,
        "next_output_prefetch_fallback_count": 0,
        "next_prefetch_late_completion_count": 0,
        "next_output_prefetch_late_completion_count": 0,
        "next_prefetch_discard_count": 0,
        "prefetched_b_rows_consumed": 0,
        "fallback_b_rows_fetched": 0,
        "next_output_prefetch_rows_issued": 0,
        "next_k_prefetch_rows_issued": 0,
        "next_output_prefetch_defer_count": 0,
        "next_output_headstart_cycles": 0,
        "next_output_first_issue_to_boundary_cycles": 0,
        "next_output_rows_ready_at_boundary": 0,
        "next_output_consumed_before_fallback_rows": 0,
        "carry_over_rows_at_boundary": 0,
        "carry_over_inflight_rows_at_boundary": 0,
        "carry_over_rows_consumed_post_boundary": 0,
        "normal_fetch_hole_rows": 0,
        "duplicate_b_row_fetch_avoided": 0,
        "duplicate_b_row_fetch_detected": 0,
        "carry_over_late_completion_count": 0,
        "normal_fetch_deferred_by_carry": 0,
        "writec_overlap_cycles": 0,
        "writec_overlap_enabled_count": 0,
        "writec_overlap_success_count": 0,
        "next_output_progress_during_writec": 0,
        "b_rows_issued_during_writec": 0,
        "writec_blocked_b_issue_count": 0,
        "a_rows_issued": 0,
        "a_rows_ready_before_compute": 0,
        "a_rows_inflight_peak": 0,
        "b_rows_inflight_peak": 0,
        "ab_parallel_fetch_overlap_cycles": 0,
        "a_fetch_progress_during_b_fetch": 0,
        "b_fetch_progress_during_a_fetch": 0,
        "a_path_stall_waiting_for_b": 0,
        "b_path_stall_waiting_for_a": 0,
        "a_credit_floor_hits": 0,
        "b_bias_wins": 0,
        "urgency_priority_wins": 0,
        "deficit_priority_wins": 0,
        "reuse_priority_wins": 0,
        "fallback_risk_priority_wins": 0,
        "score_tie_break_count": 0,
        "avg_score_a": 0.0,
        "avg_score_b": 0.0,
        "max_score_a": 0.0,
        "max_score_b": 0.0,
        "protected_b_issue_count": 0,
        "protected_b_ready_count": 0,
        "protected_b_priority_wins": 0,
        "protected_b_blocks_a_count": 0,
        "protected_b_blocks_normal_b_count": 0,
        "protected_b_from_carry_over_count": 0,
        "protected_b_from_next_output_count": 0,
        "protected_b_from_hole_filling_count": 0,
        "protected_b_from_compute_window_count": 0,
        "future_protected_b_issue_count": 0,
        "future_protected_b_priority_wins": 0,
        "future_protected_b_blocks_a_count": 0,
        "future_protected_b_blocks_current_b_count": 0,
        "future_protected_b_from_next_output_count": 0,
        "future_protected_b_from_future_hole_filling_count": 0,
        "future_protected_b_ready_at_boundary_count": 0,
        "current_protected_b_issue_count": 0,
        "current_protected_b_priority_wins": 0,
        "current_protected_b_blocks_a_count": 0,
        "current_protected_b_quota_exhaust_count": 0,
        "current_protected_b_from_compute_window_count": 0,
        "current_protected_b_from_current_hole_filling_count": 0,
        "b_rows_claimed_by_future": 0,
        "future_claim_set_count": 0,
        "future_claim_cleared_count": 0,
        "future_claim_blocked_normal_fetch_count": 0,
        "future_claim_expired_count": 0,
        "future_claim_consumed_success_count": 0,
        "future_claim_invalidated_count": 0,
        "vip_pool_capacity": 0,
        "vip_pool_occupancy_peak": 0,
        "vip_insert_count": 0,
        "vip_hit_count": 0,
        "vip_miss_count": 0,
        "vip_eviction_count": 0,
        "vip_hit_on_next_output_count": 0,
        "vip_hit_on_next_output_immediate_count": 0,
        "vip_hit_on_next_output_near_count": 0,
        "vip_hit_on_claimed_future_b_count": 0,
        "vip_hit_on_carry_over_b_count": 0,
        "vip_insert_from_next_output_count": 0,
        "vip_insert_from_claim_count": 0,
        "vip_insert_from_carry_over_count": 0,
        "vip_materialize_to_current_count": 0,
        "vip_b_rows_served_to_compute": 0,
        "vip_b_rows_prevented_fallback_count": 0,
        "vip_strong_admit_count": 0,
        "vip_weak_admit_count": 0,
        "vip_evict_low_priority_count": 0,
        "vip_evict_weak_admit_count": 0,
        "vip_evict_normal_count": 0,
        "vip_admit_next_output_immediate_count": 0,
        "vip_admit_next_output_near_count": 0,
        "vip_admit_next_output_far_count": 0,
        "vip_admit_claim_count": 0,
        "vip_admit_carry_over_count": 0,
        "vip_admit_current_window_immediate_count": 0,
        "vip_reject_normal_near_count": 0,
        "vip_reject_normal_far_count": 0,
        "vip_reject_other_count": 0,
        "oracle_selected_b_rows_count": 0,
        "oracle_selected_rows_served_by_vip_count": 0,
        "oracle_selected_rows_missed_by_vip_count": 0,
        "vip_rescue_insert_count": 0,
        "vip_rescue_hit_count": 0,
        "vip_rescue_miss_count": 0,
        "vip_rescue_eviction_count": 0,
        "vip_rescue_served_to_compute_count": 0,
        "vip_rescue_prevented_fallback_count": 0,
        "vip_rescue_insert_after_fallback_count": 0,
        "vip_rescue_insert_short_next_use_count": 0,
        "vip_rescue_insert_multi_future_use_count": 0,
        "vip_rescue_reused_count": 0,
        "vip_rescue_a_insert_count": 0,
        "vip_rescue_a_hit_count": 0,
        "vip_rescue_a_miss_count": 0,
        "vip_rescue_a_eviction_count": 0,
        "vip_rescue_a_served_to_compute_count": 0,
        "vip_rescue_a_prevented_remote_fetch_count": 0,
        "vip_rescue_a_insert_short_next_use_count": 0,
        "vip_rescue_a_insert_multi_future_use_count": 0,
        "vip_rescue_a_reused_count": 0,
        "mhot_pool_capacity": 0,
        "mhot_occupancy_peak": 0,
        "mhot_insert_count": 0,
        "mhot_insert_default_count": 0,
        "mhot_insert_enhanced_count": 0,
        "mhot_insert_next_output_immediate_count": 0,
        "mhot_insert_next_output_near_count": 0,
        "mhot_insert_next_output_far_count": 0,
        "mhot_insert_claim_count": 0,
        "mhot_insert_carry_over_count": 0,
        "mhot_insert_normal_near_count": 0,
        "mhot_insert_current_window_far_count": 0,
        "mhot_insert_coverage_blindspot_count": 0,
        "mhot_hit_count": 0,
        "mhot_hit_on_next_output_immediate_count": 0,
        "mhot_hit_on_next_output_near_count": 0,
        "mhot_hit_on_next_output_far_count": 0,
        "mhot_hit_on_claim_count": 0,
        "mhot_hit_on_carry_over_count": 0,
        "mhot_hit_on_normal_near_count": 0,
        "mhot_hit_on_current_window_far_count": 0,
        "mhot_hit_on_coverage_blindspot_count": 0,
        "mhot_rows_served_to_compute_count": 0,
        "mhot_materialize_to_current_count": 0,
        "mhot_eviction_count": 0,
        "mhot_eviction_coverage_blindspot_count": 0,
        "mhot_reuse_hit_count": 0,
        "mhot_mainline_prevented_remote_count": 0,
        "mhot_rows_served_to_compute_from_coverage_blindspot_count": 0,
        "mhot_mainline_prevented_remote_from_coverage_blindspot_count": 0,
        "mhot_checked_on_fallback_count": 0,
        "mhot_miss_then_remote_count": 0,
        "mhot_hit_before_remote_count": 0,
        "coverage_shadow_pool_capacity": 0,
        "coverage_shadow_occupancy_peak": 0,
        "coverage_shadow_insert_count": 0,
        "coverage_shadow_insert_from_next_k_count": 0,
        "coverage_shadow_insert_from_next_output_count": 0,
        "coverage_shadow_insert_from_gather_count": 0,
        "coverage_shadow_hit_count": 0,
        "coverage_shadow_rows_served_to_compute_count": 0,
        "coverage_shadow_prevented_remote_count": 0,
        "coverage_shadow_eviction_count": 0,
        "coverage_shadow_checked_on_fallback_count": 0,
        "coverage_shadow_miss_then_remote_count": 0,
        "coverage_shadow_hit_before_remote_count": 0,
        "coverage_shadow_prime_check_count": 0,
        "coverage_shadow_prime_hit_count": 0,
        "coverage_shadow_prime_miss_count": 0,
        "coverage_shadow_prime_rows_materialized_count": 0,
        "coverage_shadow_fallback_lookup_skipped_count": 0,
        "coverage_gather_issue_count": 0,
        "coverage_gather_completion_count": 0,
        "coverage_gather_late_completion_count": 0,
        "coverage_gather_target_rows": 0,
        "mhot_coverage_blindspot_seen_count": 0,
        "mhot_coverage_blindspot_repeat_count": 0,
        "mhot_coverage_blindspot_promoted_count": 0,
        "mhot_coverage_blindspot_reject_single_count": 0,
        "mhot_coverage_blindspot_reject_long_distance_count": 0,
        "mhot_coverage_blindspot_reject_low_reuse_count": 0,
        "repeat_coverage_victim_count_total": 0,
        "repeat_coverage_victim_promoted_count": 0,
        "coverage_blindspot_short_next_use_count": 0,
        "coverage_blindspot_future_reuse_gt1_count": 0,
        "coverage_blindspot_first_seen_miss_count": 0,
        "coverage_blindspot_repeat_miss_count": 0,
        "coverage_blindspot_distinct_pattern_count": 0,
        "coverage_blindspot_target_miss_count": 0,
        "coverage_blindspot_target_first_seen_miss_count": 0,
        "coverage_blindspot_target_repeat_miss_count": 0,
        "coverage_blindspot_target_distinct_pattern_count": 0,
        "fallback_autopsy_coverage_normal_immediate_count": 0,
        "fallback_autopsy_coverage_normal_near_count": 0,
        "fallback_autopsy_coverage_normal_far_count": 0,
        "coverage_blindspot_seen_distance_immediate_count": 0,
        "coverage_blindspot_seen_distance_near_count": 0,
        "coverage_blindspot_seen_distance_far_count": 0,
        "coverage_blindspot_promoted_distance_immediate_count": 0,
        "coverage_blindspot_promoted_distance_near_count": 0,
        "coverage_blindspot_promoted_distance_far_count": 0,
        "mhot_hit_on_coverage_blindspot_immediate_count": 0,
        "mhot_hit_on_coverage_blindspot_near_count": 0,
        "mhot_hit_on_coverage_blindspot_far_count": 0,
        "coverage_blindspot_seen_reuse_none_count": 0,
        "coverage_blindspot_seen_reuse_one_count": 0,
        "coverage_blindspot_seen_reuse_two_to_three_count": 0,
        "coverage_blindspot_seen_reuse_four_plus_count": 0,
        "coverage_blindspot_promoted_reuse_none_count": 0,
        "coverage_blindspot_promoted_reuse_one_count": 0,
        "coverage_blindspot_promoted_reuse_two_to_three_count": 0,
        "coverage_blindspot_promoted_reuse_four_plus_count": 0,
        "coverage_blindspot_seen_tile_lte4_count": 0,
        "coverage_blindspot_seen_tile_lte12_count": 0,
        "coverage_blindspot_seen_tile_lte24_count": 0,
        "coverage_blindspot_seen_tile_gt24_count": 0,
        "coverage_blindspot_promoted_tile_lte4_count": 0,
        "coverage_blindspot_promoted_tile_lte12_count": 0,
        "coverage_blindspot_promoted_tile_lte24_count": 0,
        "coverage_blindspot_promoted_tile_gt24_count": 0,
        "fallback_autopsy_timeliness_count": 0,
        "fallback_autopsy_churn_count": 0,
        "fallback_autopsy_coverage_count": 0,
        "rx_a_issue_count": 0,
        "rx_b_issue_count": 0,
        "rx_a_ready_count": 0,
        "rx_b_ready_count": 0,
        "rx_a_queue_occupancy_peak": 0,
        "rx_b_queue_occupancy_peak": 0,
        "rx_a_stall_cycles": 0,
        "rx_b_stall_cycles": 0,
        "rx_a_priority_wins": 0,
        "rx_b_priority_wins": 0,
        "rx_b_deficit_wins": 0,
        "rx_a_deficit_wins": 0,
        "rx_b_deadline_wins": 0,
        "rx_a_deadline_wins": 0,
    }
    if not os.path.exists(stats_file):
        return stats

    patterns = {
        "next_prefetch_issue_count": r"matrix_engine\.nextPrefetchIssueCount\s+(\d+)",
        "next_k_prefetch_issue_count": r"matrix_engine\.nextKPrefetchIssueCount\s+(\d+)",
        "next_output_prefetch_issue_count": r"matrix_engine\.nextOutputPrefetchIssueCount\s+(\d+)",
        "next_prefetch_hit_count": r"matrix_engine\.nextPrefetchHitCount\s+(\d+)",
        "next_output_prefetch_hit_count": r"matrix_engine\.nextOutputPrefetchHitCount\s+(\d+)",
        "next_prefetch_fallback_count": r"matrix_engine\.nextPrefetchFallbackCount\s+(\d+)",
        "next_output_prefetch_fallback_count": r"matrix_engine\.nextOutputPrefetchFallbackCount\s+(\d+)",
        "next_prefetch_late_completion_count": r"matrix_engine\.nextPrefetchLateCompletionCount\s+(\d+)",
        "next_output_prefetch_late_completion_count": r"matrix_engine\.nextOutputPrefetchLateCompletionCount\s+(\d+)",
        "next_prefetch_discard_count": r"matrix_engine\.nextPrefetchDiscardCount\s+(\d+)",
        "prefetched_b_rows_consumed": r"matrix_engine\.prefetchedBRowsConsumed\s+(\d+)",
        "fallback_b_rows_fetched": r"matrix_engine\.fallbackBRowsFetched\s+(\d+)",
        "next_output_prefetch_rows_issued": r"matrix_engine\.nextOutputPrefetchRowsIssued\s+(\d+)",
        "next_k_prefetch_rows_issued": r"matrix_engine\.nextKPrefetchRowsIssued\s+(\d+)",
        "next_output_prefetch_defer_count": r"matrix_engine\.nextOutputPrefetchDeferCount\s+(\d+)",
        "next_output_headstart_cycles": r"matrix_engine\.nextOutputHeadstartCycles\s+(\d+)",
        "next_output_first_issue_to_boundary_cycles": r"matrix_engine\.nextOutputFirstIssueToBoundaryCycles\s+(\d+)",
        "next_output_rows_ready_at_boundary": r"matrix_engine\.nextOutputRowsReadyAtBoundary\s+(\d+)",
        "next_output_consumed_before_fallback_rows": r"matrix_engine\.nextOutputConsumedBeforeFallbackRows\s+(\d+)",
        "carry_over_rows_at_boundary": r"matrix_engine\.carryOverRowsAtBoundary\s+(\d+)",
        "carry_over_inflight_rows_at_boundary": r"matrix_engine\.carryOverInflightRowsAtBoundary\s+(\d+)",
        "carry_over_rows_consumed_post_boundary": r"matrix_engine\.carryOverRowsConsumedPostBoundary\s+(\d+)",
        "normal_fetch_hole_rows": r"matrix_engine\.normalFetchHoleRows\s+(\d+)",
        "duplicate_b_row_fetch_avoided": r"matrix_engine\.duplicateBRowFetchAvoided\s+(\d+)",
        "duplicate_b_row_fetch_detected": r"matrix_engine\.duplicateBRowFetchDetected\s+(\d+)",
        "carry_over_late_completion_count": r"matrix_engine\.carryOverLateCompletionCount\s+(\d+)",
        "normal_fetch_deferred_by_carry": r"matrix_engine\.normalFetchDeferredByCarry\s+(\d+)",
        "writec_overlap_cycles": r"matrix_engine\.writeCOverlapCycles\s+(\d+)",
        "writec_overlap_enabled_count": r"matrix_engine\.writeCOverlapEnabledCount\s+(\d+)",
        "writec_overlap_success_count": r"matrix_engine\.writeCOverlapSuccessCount\s+(\d+)",
        "next_output_progress_during_writec": r"matrix_engine\.nextOutputProgressDuringWriteC\s+(\d+)",
        "b_rows_issued_during_writec": r"matrix_engine\.bRowsIssuedDuringWriteC\s+(\d+)",
        "writec_blocked_b_issue_count": r"matrix_engine\.writeCBlockedBIssueCount\s+(\d+)",
        "a_rows_issued": r"matrix_engine\.aRowsIssued\s+(\d+)",
        "a_rows_ready_before_compute": r"matrix_engine\.aRowsReadyBeforeCompute\s+(\d+)",
        "a_rows_inflight_peak": r"matrix_engine\.aRowsInflightPeak\s+(\d+)",
        "b_rows_inflight_peak": r"matrix_engine\.bRowsInflightPeak\s+(\d+)",
        "ab_parallel_fetch_overlap_cycles": r"matrix_engine\.abParallelFetchOverlapCycles\s+(\d+)",
        "a_fetch_progress_during_b_fetch": r"matrix_engine\.aFetchProgressDuringBFetch\s+(\d+)",
        "b_fetch_progress_during_a_fetch": r"matrix_engine\.bFetchProgressDuringAFetch\s+(\d+)",
        "a_path_stall_waiting_for_b": r"matrix_engine\.aPathStallWaitingForB\s+(\d+)",
        "b_path_stall_waiting_for_a": r"matrix_engine\.bPathStallWaitingForA\s+(\d+)",
        "a_credit_floor_hits": r"matrix_engine\.aCreditFloorHits\s+(\d+)",
        "b_bias_wins": r"matrix_engine\.bBiasWins\s+(\d+)",
        "urgency_priority_wins": r"matrix_engine\.urgencyPriorityWins\s+(\d+)",
        "deficit_priority_wins": r"matrix_engine\.deficitPriorityWins\s+(\d+)",
        "reuse_priority_wins": r"matrix_engine\.reusePriorityWins\s+(\d+)",
        "fallback_risk_priority_wins": r"matrix_engine\.fallbackRiskPriorityWins\s+(\d+)",
        "score_tie_break_count": r"matrix_engine\.scoreTieBreakCount\s+(\d+)",
        "avg_score_a": r"matrix_engine\.avgScoreA\s+([0-9.]+)",
        "avg_score_b": r"matrix_engine\.avgScoreB\s+([0-9.]+)",
        "max_score_a": r"matrix_engine\.maxScoreA\s+([0-9.]+)",
        "max_score_b": r"matrix_engine\.maxScoreB\s+([0-9.]+)",
        "protected_b_issue_count": r"matrix_engine\.protectedBIssueCount\s+(\d+)",
        "protected_b_ready_count": r"matrix_engine\.protectedBReadyCount\s+(\d+)",
        "protected_b_priority_wins": r"matrix_engine\.protectedBPriorityWins\s+(\d+)",
        "protected_b_blocks_a_count": r"matrix_engine\.protectedBBlocksACount\s+(\d+)",
        "protected_b_blocks_normal_b_count": r"matrix_engine\.protectedBBlocksNormalBCount\s+(\d+)",
        "protected_b_from_carry_over_count": r"matrix_engine\.protectedBFromCarryOverCount\s+(\d+)",
        "protected_b_from_next_output_count": r"matrix_engine\.protectedBFromNextOutputCount\s+(\d+)",
        "protected_b_from_hole_filling_count": r"matrix_engine\.protectedBFromHoleFillingCount\s+(\d+)",
        "protected_b_from_compute_window_count": r"matrix_engine\.protectedBFromComputeWindowCount\s+(\d+)",
        "future_protected_b_issue_count": r"matrix_engine\.futureProtectedBIssueCount\s+(\d+)",
        "future_protected_b_priority_wins": r"matrix_engine\.futureProtectedBPriorityWins\s+(\d+)",
        "future_protected_b_blocks_a_count": r"matrix_engine\.futureProtectedBBlocksACount\s+(\d+)",
        "future_protected_b_blocks_current_b_count": r"matrix_engine\.futureProtectedBBlocksCurrentBCount\s+(\d+)",
        "future_protected_b_from_next_output_count": r"matrix_engine\.futureProtectedBFromNextOutputCount\s+(\d+)",
        "future_protected_b_from_future_hole_filling_count": r"matrix_engine\.futureProtectedBFromFutureHoleFillingCount\s+(\d+)",
        "future_protected_b_ready_at_boundary_count": r"matrix_engine\.futureProtectedBReadyAtBoundaryCount\s+(\d+)",
        "current_protected_b_issue_count": r"matrix_engine\.currentProtectedBIssueCount\s+(\d+)",
        "current_protected_b_priority_wins": r"matrix_engine\.currentProtectedBPriorityWins\s+(\d+)",
        "current_protected_b_blocks_a_count": r"matrix_engine\.currentProtectedBBlocksACount\s+(\d+)",
        "current_protected_b_quota_exhaust_count": r"matrix_engine\.currentProtectedBQuotaExhaustCount\s+(\d+)",
        "current_protected_b_from_compute_window_count": r"matrix_engine\.currentProtectedBFromComputeWindowCount\s+(\d+)",
        "current_protected_b_from_current_hole_filling_count": r"matrix_engine\.currentProtectedBFromCurrentHoleFillingCount\s+(\d+)",
        "b_rows_claimed_by_future": r"matrix_engine\.bRowsClaimedByFuture\s+(\d+)",
        "future_claim_set_count": r"matrix_engine\.futureClaimSetCount\s+(\d+)",
        "future_claim_cleared_count": r"matrix_engine\.futureClaimClearedCount\s+(\d+)",
        "future_claim_blocked_normal_fetch_count": r"matrix_engine\.futureClaimBlockedNormalFetchCount\s+(\d+)",
        "future_claim_expired_count": r"matrix_engine\.futureClaimExpiredCount\s+(\d+)",
        "future_claim_consumed_success_count": r"matrix_engine\.futureClaimConsumedSuccessCount\s+(\d+)",
        "future_claim_invalidated_count": r"matrix_engine\.futureClaimInvalidatedCount\s+(\d+)",
        "vip_pool_capacity": r"matrix_engine\.vipPoolCapacity\s+(\d+)",
        "vip_pool_occupancy_peak": r"matrix_engine\.vipPoolOccupancyPeak\s+(\d+)",
        "vip_insert_count": r"matrix_engine\.vipInsertCount\s+(\d+)",
        "vip_hit_count": r"matrix_engine\.vipHitCount\s+(\d+)",
        "vip_miss_count": r"matrix_engine\.vipMissCount\s+(\d+)",
        "vip_eviction_count": r"matrix_engine\.vipEvictionCount\s+(\d+)",
        "vip_hit_on_next_output_count": r"matrix_engine\.vipHitOnNextOutputCount\s+(\d+)",
        "vip_hit_on_next_output_immediate_count": r"matrix_engine\.vipHitOnNextOutputImmediateCount\s+(\d+)",
        "vip_hit_on_next_output_near_count": r"matrix_engine\.vipHitOnNextOutputNearCount\s+(\d+)",
        "vip_hit_on_claimed_future_b_count": r"matrix_engine\.vipHitOnClaimedFutureBCount\s+(\d+)",
        "vip_hit_on_carry_over_b_count": r"matrix_engine\.vipHitOnCarryOverBCount\s+(\d+)",
        "vip_insert_from_next_output_count": r"matrix_engine\.vipInsertFromNextOutputCount\s+(\d+)",
        "vip_insert_from_claim_count": r"matrix_engine\.vipInsertFromClaimCount\s+(\d+)",
        "vip_insert_from_carry_over_count": r"matrix_engine\.vipInsertFromCarryOverCount\s+(\d+)",
        "vip_materialize_to_current_count": r"matrix_engine\.vipMaterializeToCurrentCount\s+(\d+)",
        "vip_b_rows_served_to_compute": r"matrix_engine\.vipBRowsServedToCompute\s+(\d+)",
        "vip_b_rows_prevented_fallback_count": r"matrix_engine\.vipBRowsPreventedFallbackCount\s+(\d+)",
        "vip_strong_admit_count": r"matrix_engine\.vipStrongAdmitCount\s+(\d+)",
        "vip_weak_admit_count": r"matrix_engine\.vipWeakAdmitCount\s+(\d+)",
        "vip_evict_low_priority_count": r"matrix_engine\.vipEvictLowPriorityCount\s+(\d+)",
        "vip_evict_weak_admit_count": r"matrix_engine\.vipEvictWeakAdmitCount\s+(\d+)",
        "vip_evict_normal_count": r"matrix_engine\.vipEvictNormalCount\s+(\d+)",
        "vip_admit_next_output_immediate_count": r"matrix_engine\.vipAdmitNextOutputImmediateCount\s+(\d+)",
        "vip_admit_next_output_near_count": r"matrix_engine\.vipAdmitNextOutputNearCount\s+(\d+)",
        "vip_admit_next_output_far_count": r"matrix_engine\.vipAdmitNextOutputFarCount\s+(\d+)",
        "vip_admit_claim_count": r"matrix_engine\.vipAdmitClaimCount\s+(\d+)",
        "vip_admit_carry_over_count": r"matrix_engine\.vipAdmitCarryOverCount\s+(\d+)",
        "vip_admit_current_window_immediate_count": r"matrix_engine\.vipAdmitCurrentWindowImmediateCount\s+(\d+)",
        "vip_reject_normal_near_count": r"matrix_engine\.vipRejectNormalNearCount\s+(\d+)",
        "vip_reject_normal_far_count": r"matrix_engine\.vipRejectNormalFarCount\s+(\d+)",
        "vip_reject_other_count": r"matrix_engine\.vipRejectOtherCount\s+(\d+)",
        "oracle_selected_b_rows_count": r"matrix_engine\.oracleSelectedBRowsCount\s+(\d+)",
        "oracle_selected_rows_served_by_vip_count": r"matrix_engine\.oracleSelectedRowsServedByVipCount\s+(\d+)",
        "oracle_selected_rows_missed_by_vip_count": r"matrix_engine\.oracleSelectedRowsMissedByVipCount\s+(\d+)",
        "vip_rescue_insert_count": r"matrix_engine\.vipRescueInsertCount\s+(\d+)",
        "vip_rescue_hit_count": r"matrix_engine\.vipRescueHitCount\s+(\d+)",
        "vip_rescue_miss_count": r"matrix_engine\.vipRescueMissCount\s+(\d+)",
        "vip_rescue_eviction_count": r"matrix_engine\.vipRescueEvictionCount\s+(\d+)",
        "vip_rescue_served_to_compute_count": r"matrix_engine\.vipRescueServedToComputeCount\s+(\d+)",
        "vip_rescue_prevented_fallback_count": r"matrix_engine\.vipRescuePreventedFallbackCount\s+(\d+)",
        "vip_rescue_insert_after_fallback_count": r"matrix_engine\.vipRescueInsertAfterFallbackCount\s+(\d+)",
        "vip_rescue_insert_short_next_use_count": r"matrix_engine\.vipRescueInsertShortNextUseCount\s+(\d+)",
        "vip_rescue_insert_multi_future_use_count": r"matrix_engine\.vipRescueInsertMultiFutureUseCount\s+(\d+)",
        "vip_rescue_reused_count": r"matrix_engine\.vipRescueReusedCount\s+(\d+)",
        "vip_rescue_insert_strong_count": r"matrix_engine\.vipRescueInsertStrongCount\s+(\d+)",
        "vip_rescue_insert_weak_count": r"matrix_engine\.vipRescueInsertWeakCount\s+(\d+)",
        "vip_rescue_reject_non_fallback_count": r"matrix_engine\.vipRescueRejectNonFallbackCount\s+(\d+)",
        "vip_rescue_reject_not_short_use_count": r"matrix_engine\.vipRescueRejectNotShortUseCount\s+(\d+)",
        "vip_rescue_reject_not_critical_count": r"matrix_engine\.vipRescueRejectNotCriticalCount\s+(\d+)",
        "vip_rescue_reject_not_rescue_critical_count": r"matrix_engine\.vipRescueRejectNotRescueCriticalCount\s+(\d+)",
        "vip_rescue_reject_single_use_count": r"matrix_engine\.vipRescueRejectSingleUseCount\s+(\d+)",
        "vip_rescue_reject_non_repeat_count": r"matrix_engine\.vipRescueRejectNonRepeatCount\s+(\d+)",
        "repeat_victim_count_total": r"matrix_engine\.repeatVictimCountTotal\s+(\d+)",
        "repeat_victim_promoted_count": r"matrix_engine\.repeatVictimPromotedCount\s+(\d+)",
        "critical_window_victim_count": r"matrix_engine\.criticalWindowVictimCount\s+(\d+)",
        "critical_window_vip_insert_count": r"matrix_engine\.criticalWindowVipInsertCount\s+(\d+)",
        "weak_rescue_promoted_on_first_fallback_count": r"matrix_engine\.weakRescuePromotedOnFirstFallbackCount\s+(\d+)",
        "strong_rescue_promoted_on_recurrence_count": r"matrix_engine\.strongRescuePromotedOnRecurrenceCount\s+(\d+)",
        "rescue_criticality_high_count": r"matrix_engine\.rescueCriticalityHighCount\s+(\d+)",
        "next_rescue_opportunity_near_count": r"matrix_engine\.nextRescueOpportunityNearCount\s+(\d+)",
        "eligible_rescue_victims_first_seen_count": r"matrix_engine\.eligibleRescueVictimsFirstSeenCount\s+(\d+)",
        "eligible_rescue_victims_promoted_early_count": r"matrix_engine\.eligibleRescueVictimsPromotedEarlyCount\s+(\d+)",
        "eligible_rescue_victims_hit_after_first_promotion_count": r"matrix_engine\.eligibleRescueVictimsHitAfterFirstPromotionCount\s+(\d+)",
        "vip_rescue_a_insert_count": r"matrix_engine\.vipRescueAInsertCount\s+(\d+)",
        "vip_rescue_a_hit_count": r"matrix_engine\.vipRescueAHitCount\s+(\d+)",
        "vip_rescue_a_miss_count": r"matrix_engine\.vipRescueAMissCount\s+(\d+)",
        "vip_rescue_a_eviction_count": r"matrix_engine\.vipRescueAEvictionCount\s+(\d+)",
        "vip_rescue_a_served_to_compute_count": r"matrix_engine\.vipRescueAServedToComputeCount\s+(\d+)",
        "vip_rescue_a_prevented_remote_fetch_count": r"matrix_engine\.vipRescueAPreventedRemoteFetchCount\s+(\d+)",
        "vip_rescue_a_insert_short_next_use_count": r"matrix_engine\.vipRescueAInsertShortNextUseCount\s+(\d+)",
        "vip_rescue_a_insert_multi_future_use_count": r"matrix_engine\.vipRescueAInsertMultiFutureUseCount\s+(\d+)",
        "vip_rescue_a_reused_count": r"matrix_engine\.vipRescueAReusedCount\s+(\d+)",
        "mhot_pool_capacity": r"matrix_engine\.mhotPoolCapacity\s+(\d+)",
        "mhot_occupancy_peak": r"matrix_engine\.mhotOccupancyPeak\s+(\d+)",
        "mhot_insert_count": r"matrix_engine\.mhotInsertCount\s+(\d+)",
        "mhot_insert_default_count": r"matrix_engine\.mhotInsertDefaultCount\s+(\d+)",
        "mhot_insert_enhanced_count": r"matrix_engine\.mhotInsertEnhancedCount\s+(\d+)",
        "mhot_insert_next_output_immediate_count": r"matrix_engine\.mhotInsertNextOutputImmediateCount\s+(\d+)",
        "mhot_insert_next_output_near_count": r"matrix_engine\.mhotInsertNextOutputNearCount\s+(\d+)",
        "mhot_insert_next_output_far_count": r"matrix_engine\.mhotInsertNextOutputFarCount\s+(\d+)",
        "mhot_insert_claim_count": r"matrix_engine\.mhotInsertClaimCount\s+(\d+)",
        "mhot_insert_carry_over_count": r"matrix_engine\.mhotInsertCarryOverCount\s+(\d+)",
        "mhot_insert_normal_near_count": r"matrix_engine\.mhotInsertNormalNearCount\s+(\d+)",
        "mhot_insert_current_window_far_count": r"matrix_engine\.mhotInsertCurrentWindowFarCount\s+(\d+)",
        "mhot_insert_coverage_blindspot_count": r"matrix_engine\.mhotInsertCoverageBlindspotCount\s+(\d+)",
        "mhot_hit_count": r"matrix_engine\.mhotHitCount\s+(\d+)",
        "mhot_hit_on_next_output_immediate_count": r"matrix_engine\.mhotHitOnNextOutputImmediateCount\s+(\d+)",
        "mhot_hit_on_next_output_near_count": r"matrix_engine\.mhotHitOnNextOutputNearCount\s+(\d+)",
        "mhot_hit_on_next_output_far_count": r"matrix_engine\.mhotHitOnNextOutputFarCount\s+(\d+)",
        "mhot_hit_on_claim_count": r"matrix_engine\.mhotHitOnClaimCount\s+(\d+)",
        "mhot_hit_on_carry_over_count": r"matrix_engine\.mhotHitOnCarryOverCount\s+(\d+)",
        "mhot_hit_on_normal_near_count": r"matrix_engine\.mhotHitOnNormalNearCount\s+(\d+)",
        "mhot_hit_on_current_window_far_count": r"matrix_engine\.mhotHitOnCurrentWindowFarCount\s+(\d+)",
        "mhot_hit_on_coverage_blindspot_count": r"matrix_engine\.mhotHitOnCoverageBlindspotCount\s+(\d+)",
        "mhot_rows_served_to_compute_count": r"matrix_engine\.mhotRowsServedToComputeCount\s+(\d+)",
        "mhot_materialize_to_current_count": r"matrix_engine\.mhotMaterializeToCurrentCount\s+(\d+)",
        "mhot_eviction_count": r"matrix_engine\.mhotEvictionCount\s+(\d+)",
        "mhot_eviction_coverage_blindspot_count": r"matrix_engine\.mhotEvictionCoverageBlindspotCount\s+(\d+)",
        "mhot_reuse_hit_count": r"matrix_engine\.mhotReuseHitCount\s+(\d+)",
        "mhot_mainline_prevented_remote_count": r"matrix_engine\.mhotMainlinePreventedRemoteCount\s+(\d+)",
        "mhot_rows_served_to_compute_from_coverage_blindspot_count": r"matrix_engine\.mhotRowsServedToComputeFromCoverageBlindspotCount\s+(\d+)",
        "mhot_mainline_prevented_remote_from_coverage_blindspot_count": r"matrix_engine\.mhotMainlinePreventedRemoteFromCoverageBlindspotCount\s+(\d+)",
        "mhot_checked_on_fallback_count": r"matrix_engine\.mhotCheckedOnFallbackCount\s+(\d+)",
        "mhot_miss_then_remote_count": r"matrix_engine\.mhotMissThenRemoteCount\s+(\d+)",
        "mhot_hit_before_remote_count": r"matrix_engine\.mhotHitBeforeRemoteCount\s+(\d+)",
        "coverage_shadow_pool_capacity": r"matrix_engine\.coverageShadowPoolCapacity\s+(\d+)",
        "coverage_shadow_occupancy_peak": r"matrix_engine\.coverageShadowOccupancyPeak\s+(\d+)",
        "coverage_shadow_insert_count": r"matrix_engine\.coverageShadowInsertCount\s+(\d+)",
        "coverage_shadow_insert_from_next_k_count": r"matrix_engine\.coverageShadowInsertFromNextKCount\s+(\d+)",
        "coverage_shadow_insert_from_next_output_count": r"matrix_engine\.coverageShadowInsertFromNextOutputCount\s+(\d+)",
        "coverage_shadow_insert_from_gather_count": r"matrix_engine\.coverageShadowInsertFromGatherCount\s+(\d+)",
        "coverage_shadow_hit_count": r"matrix_engine\.coverageShadowHitCount\s+(\d+)",
        "coverage_shadow_rows_served_to_compute_count": r"matrix_engine\.coverageShadowRowsServedToComputeCount\s+(\d+)",
        "coverage_shadow_prevented_remote_count": r"matrix_engine\.coverageShadowPreventedRemoteCount\s+(\d+)",
        "coverage_shadow_eviction_count": r"matrix_engine\.coverageShadowEvictionCount\s+(\d+)",
        "coverage_shadow_checked_on_fallback_count": r"matrix_engine\.coverageShadowCheckedOnFallbackCount\s+(\d+)",
        "coverage_shadow_miss_then_remote_count": r"matrix_engine\.coverageShadowMissThenRemoteCount\s+(\d+)",
        "coverage_shadow_hit_before_remote_count": r"matrix_engine\.coverageShadowHitBeforeRemoteCount\s+(\d+)",
        "coverage_shadow_prime_check_count": r"matrix_engine\.coverageShadowPrimeCheckCount\s+(\d+)",
        "coverage_shadow_prime_hit_count": r"matrix_engine\.coverageShadowPrimeHitCount\s+(\d+)",
        "coverage_shadow_prime_miss_count": r"matrix_engine\.coverageShadowPrimeMissCount\s+(\d+)",
        "coverage_shadow_prime_rows_materialized_count": r"matrix_engine\.coverageShadowPrimeRowsMaterializedCount\s+(\d+)",
        "coverage_shadow_fallback_lookup_skipped_count": r"matrix_engine\.coverageShadowFallbackLookupSkippedCount\s+(\d+)",
        "coverage_gather_issue_count": r"matrix_engine\.coverageGatherIssueCount\s+(\d+)",
        "coverage_gather_completion_count": r"matrix_engine\.coverageGatherCompletionCount\s+(\d+)",
        "coverage_gather_late_completion_count": r"matrix_engine\.coverageGatherLateCompletionCount\s+(\d+)",
        "coverage_gather_target_rows": r"matrix_engine\.coverageGatherTargetRows\s+(\d+)",
        "coverage_gather_completion_while_row_empty_count": r"matrix_engine\.coverageGatherCompletionWhileRowEmptyCount\s+(\d+)",
        "coverage_gather_completion_while_row_inflight_count": r"matrix_engine\.coverageGatherCompletionWhileRowInflightCount\s+(\d+)",
        "coverage_gather_completion_while_row_ready_count": r"matrix_engine\.coverageGatherCompletionWhileRowReadyCount\s+(\d+)",
        "coverage_gather_completion_while_row_consumed_count": r"matrix_engine\.coverageGatherCompletionWhileRowConsumedCount\s+(\d+)",
        "gather_deferred_by_mainline_count": r"matrix_engine\.gatherDeferredByMainlineCount\s+(\d+)",
        "mainline_credit_reserved_count": r"matrix_engine\.mainlineCreditReservedCount\s+(\d+)",
        "gather_blocked_by_mainline_grace_window_count": r"matrix_engine\.gatherBlockedByMainlineGraceWindowCount\s+(\d+)",
        "gather_deferred_by_vip_count": r"matrix_engine\.gatherDeferredByVipCount\s+(\d+)",
        "vip_deferred_by_gather_count": r"matrix_engine\.vipDeferredByGatherCount\s+(\d+)",
        "vip_rescue_mode_active_count": r"matrix_engine\.vipRescueModeActiveCount\s+(\d+)",
        "gather_budget_exhausted_count": r"matrix_engine\.gatherBudgetExhaustedCount\s+(\d+)",
        "gather_max_consecutive_issue_hits_count": r"matrix_engine\.gatherMaxConsecutiveIssueHitsCount\s+(\d+)",
        "gather_candidate_aged_up_count": r"matrix_engine\.gatherCandidateAgedUpCount\s+(\d+)",
        "gather_dropped_due_to_deadline_count": r"matrix_engine\.gatherDroppedDueToDeadlineCount\s+(\d+)",
        "mhot_coverage_blindspot_seen_count": r"matrix_engine\.mhotCoverageBlindspotSeenCount\s+(\d+)",
        "mhot_coverage_blindspot_repeat_count": r"matrix_engine\.mhotCoverageBlindspotRepeatCount\s+(\d+)",
        "mhot_coverage_blindspot_promoted_count": r"matrix_engine\.mhotCoverageBlindspotPromotedCount\s+(\d+)",
        "mhot_coverage_blindspot_reject_single_count": r"matrix_engine\.mhotCoverageBlindspotRejectSingleCount\s+(\d+)",
        "mhot_coverage_blindspot_reject_long_distance_count": r"matrix_engine\.mhotCoverageBlindspotRejectLongDistanceCount\s+(\d+)",
        "mhot_coverage_blindspot_reject_low_reuse_count": r"matrix_engine\.mhotCoverageBlindspotRejectLowReuseCount\s+(\d+)",
        "repeat_coverage_victim_count_total": r"matrix_engine\.repeatCoverageVictimCountTotal\s+(\d+)",
        "repeat_coverage_victim_promoted_count": r"matrix_engine\.repeatCoverageVictimPromotedCount\s+(\d+)",
        "coverage_blindspot_short_next_use_count": r"matrix_engine\.coverageBlindspotShortNextUseCount\s+(\d+)",
        "coverage_blindspot_future_reuse_gt1_count": r"matrix_engine\.coverageBlindspotFutureReuseGt1Count\s+(\d+)",
        "coverage_blindspot_first_seen_miss_count": r"matrix_engine\.coverageBlindspotFirstSeenMissCount\s+(\d+)",
        "coverage_blindspot_repeat_miss_count": r"matrix_engine\.coverageBlindspotRepeatMissCount\s+(\d+)",
        "coverage_blindspot_distinct_pattern_count": r"matrix_engine\.coverageBlindspotDistinctPatternCount\s+(\d+)",
        "coverage_blindspot_target_miss_count": r"matrix_engine\.coverageBlindspotTargetMissCount\s+(\d+)",
        "coverage_blindspot_target_first_seen_miss_count": r"matrix_engine\.coverageBlindspotTargetFirstSeenMissCount\s+(\d+)",
        "coverage_blindspot_target_repeat_miss_count": r"matrix_engine\.coverageBlindspotTargetRepeatMissCount\s+(\d+)",
        "coverage_blindspot_target_distinct_pattern_count": r"matrix_engine\.coverageBlindspotTargetDistinctPatternCount\s+(\d+)",
        "smart_prefetch_a_issue_count": r"matrix_engine\.smartPrefetchAIssueCount\s+(\d+)",
        "smart_prefetch_a_repeat_priority_count": r"matrix_engine\.smartPrefetchARepeatPriorityCount\s+(\d+)",
        "smart_prefetch_a_near_multi_priority_count": r"matrix_engine\.smartPrefetchANearMultiPriorityCount\s+(\d+)",
        "smart_prefetch_b_next_output_priority_issue_count": r"matrix_engine\.smartPrefetchBNextOutputPriorityIssueCount\s+(\d+)",
        "smart_prefetch_b_next_output_coverage_target_issue_count": r"matrix_engine\.smartPrefetchBNextOutputCoverageTargetIssueCount\s+(\d+)",
        "smart_prefetch_b_next_k_priority_issue_count": r"matrix_engine\.smartPrefetchBNextKPriorityIssueCount\s+(\d+)",
        "fallback_autopsy_coverage_normal_immediate_count": r"matrix_engine\.fallbackAutopsyCoverageNormalByDistance::immediate\s+(\d+)",
        "fallback_autopsy_coverage_normal_near_count": r"matrix_engine\.fallbackAutopsyCoverageNormalByDistance::near\s+(\d+)",
        "fallback_autopsy_coverage_normal_far_count": r"matrix_engine\.fallbackAutopsyCoverageNormalByDistance::far\s+(\d+)",
        "coverage_blindspot_seen_distance_immediate_count": r"matrix_engine\.coverageBlindspotSeenByDistance::immediate\s+(\d+)",
        "coverage_blindspot_seen_distance_near_count": r"matrix_engine\.coverageBlindspotSeenByDistance::near\s+(\d+)",
        "coverage_blindspot_seen_distance_far_count": r"matrix_engine\.coverageBlindspotSeenByDistance::far\s+(\d+)",
        "coverage_blindspot_promoted_distance_immediate_count": r"matrix_engine\.coverageBlindspotPromotedByDistance::immediate\s+(\d+)",
        "coverage_blindspot_promoted_distance_near_count": r"matrix_engine\.coverageBlindspotPromotedByDistance::near\s+(\d+)",
        "coverage_blindspot_promoted_distance_far_count": r"matrix_engine\.coverageBlindspotPromotedByDistance::far\s+(\d+)",
        "mhot_hit_on_coverage_blindspot_immediate_count": r"matrix_engine\.mhotHitOnCoverageBlindspotByDistance::immediate\s+(\d+)",
        "mhot_hit_on_coverage_blindspot_near_count": r"matrix_engine\.mhotHitOnCoverageBlindspotByDistance::near\s+(\d+)",
        "mhot_hit_on_coverage_blindspot_far_count": r"matrix_engine\.mhotHitOnCoverageBlindspotByDistance::far\s+(\d+)",
        "coverage_blindspot_seen_reuse_none_count": r"matrix_engine\.coverageBlindspotSeenByReuseBucket::none\s+(\d+)",
        "coverage_blindspot_seen_reuse_one_count": r"matrix_engine\.coverageBlindspotSeenByReuseBucket::one\s+(\d+)",
        "coverage_blindspot_seen_reuse_two_to_three_count": r"matrix_engine\.coverageBlindspotSeenByReuseBucket::two_to_three\s+(\d+)",
        "coverage_blindspot_seen_reuse_four_plus_count": r"matrix_engine\.coverageBlindspotSeenByReuseBucket::four_plus\s+(\d+)",
        "coverage_blindspot_promoted_reuse_none_count": r"matrix_engine\.coverageBlindspotPromotedByReuseBucket::none\s+(\d+)",
        "coverage_blindspot_promoted_reuse_one_count": r"matrix_engine\.coverageBlindspotPromotedByReuseBucket::one\s+(\d+)",
        "coverage_blindspot_promoted_reuse_two_to_three_count": r"matrix_engine\.coverageBlindspotPromotedByReuseBucket::two_to_three\s+(\d+)",
        "coverage_blindspot_promoted_reuse_four_plus_count": r"matrix_engine\.coverageBlindspotPromotedByReuseBucket::four_plus\s+(\d+)",
        "coverage_blindspot_seen_tile_lte4_count": r"matrix_engine\.coverageBlindspotSeenByTileBand::lte4\s+(\d+)",
        "coverage_blindspot_seen_tile_lte12_count": r"matrix_engine\.coverageBlindspotSeenByTileBand::lte12\s+(\d+)",
        "coverage_blindspot_seen_tile_lte24_count": r"matrix_engine\.coverageBlindspotSeenByTileBand::lte24\s+(\d+)",
        "coverage_blindspot_seen_tile_gt24_count": r"matrix_engine\.coverageBlindspotSeenByTileBand::gt24\s+(\d+)",
        "coverage_blindspot_promoted_tile_lte4_count": r"matrix_engine\.coverageBlindspotPromotedByTileBand::lte4\s+(\d+)",
        "coverage_blindspot_promoted_tile_lte12_count": r"matrix_engine\.coverageBlindspotPromotedByTileBand::lte12\s+(\d+)",
        "coverage_blindspot_promoted_tile_lte24_count": r"matrix_engine\.coverageBlindspotPromotedByTileBand::lte24\s+(\d+)",
        "coverage_blindspot_promoted_tile_gt24_count": r"matrix_engine\.coverageBlindspotPromotedByTileBand::gt24\s+(\d+)",
        "fallback_autopsy_timeliness_count": r"matrix_engine\.fallbackAutopsyTimelinessCount\s+(\d+)",
        "fallback_autopsy_churn_count": r"matrix_engine\.fallbackAutopsyChurnCount\s+(\d+)",
        "fallback_autopsy_coverage_count": r"matrix_engine\.fallbackAutopsyCoverageCount\s+(\d+)",
        "rx_a_issue_count": r"matrix_engine\.rxAIssueCount\s+(\d+)",
        "rx_b_issue_count": r"matrix_engine\.rxBIssueCount\s+(\d+)",
        "rx_a_ready_count": r"matrix_engine\.rxAReadyCount\s+(\d+)",
        "rx_b_ready_count": r"matrix_engine\.rxBReadyCount\s+(\d+)",
        "rx_a_queue_occupancy_peak": r"matrix_engine\.rxAQueueOccupancyPeak\s+(\d+)",
        "rx_b_queue_occupancy_peak": r"matrix_engine\.rxBQueueOccupancyPeak\s+(\d+)",
        "rx_a_stall_cycles": r"matrix_engine\.rxAStallCycles\s+(\d+)",
        "rx_b_stall_cycles": r"matrix_engine\.rxBStallCycles\s+(\d+)",
        "rx_a_priority_wins": r"matrix_engine\.rxAPriorityWins\s+(\d+)",
        "rx_b_priority_wins": r"matrix_engine\.rxBPriorityWins\s+(\d+)",
        "rx_b_deficit_wins": r"matrix_engine\.rxBDeficitWins\s+(\d+)",
        "rx_a_deficit_wins": r"matrix_engine\.rxADeficitWins\s+(\d+)",
        "rx_b_deadline_wins": r"matrix_engine\.rxBDeadlineWins\s+(\d+)",
        "rx_a_deadline_wins": r"matrix_engine\.rxADeadlineWins\s+(\d+)",
    }
    text = open(stats_file, encoding="utf-8", errors="ignore").read()
    for key, pattern in patterns.items():
        matches = re.findall(pattern, text)
        if matches:
            last = matches[-1]
            stats[key] = float(last) if "." in last else int(last)
    return stats


def histogram_to_str(counter):
    parts = []
    for size in sorted(counter):
        parts.append(f"{size}B:{counter[size]}")
    return "; ".join(parts)


def compute_request_formation(preset):
    seq_len = preset["seq_len"]
    elem_bytes = 4
    tile_dim = 128
    min_read = DEFAULT_MIN_READ_REQUEST_BYTES
    raw = Counter()
    packetized = Counter()

    def add_packetized(req_bytes):
        remaining = req_bytes
        while remaining > 0:
            chunk = min(DEFAULT_MIN_READ_REQUEST_BYTES, remaining)
            packetized[chunk] += 1
            remaining -= chunk

    # One GEMM only; run_sweep separately reports per-gemm and total counts.
    raw[48] += 1  # descriptor read
    add_packetized(48)
    raw[8] += 1  # completion flag write
    add_packetized(8)

    for i in range(0, seq_len, tile_dim):
        cur_m = min(tile_dim, seq_len - i)
        for j in range(0, seq_len, tile_dim):
            cur_n = min(tile_dim, seq_len - j)
            for k in range(0, seq_len, tile_dim):
                cur_k = min(tile_dim, seq_len - k)

                # A read rows
                row_a = cur_k * elem_bytes
                req_a = max(row_a, min_read) if row_a < min_read else row_a
                raw[row_a] += cur_m
                for _ in range(cur_m):
                    add_packetized(req_a)

                # B read rows
                row_b = cur_n * elem_bytes
                req_b = max(row_b, min_read) if row_b < min_read else row_b
                raw[row_b] += cur_k
                for _ in range(cur_k):
                    add_packetized(req_b)

                # C write rows at end of k-loop only, no flooring yet
                if k + cur_k >= seq_len:
                    row_c = cur_n * elem_bytes
                    raw[row_c] += cur_m
                    for _ in range(cur_m):
                        add_packetized(row_c)

    raw_total_count = sum(raw.values())
    raw_total_bytes = sum(size * count for size, count in raw.items())
    pkt_total_count = sum(packetized.values())
    pkt_total_bytes = sum(size * count for size, count in packetized.items())
    return {
        "raw_hist": histogram_to_str(raw),
        "raw_count": raw_total_count,
        "raw_bytes": raw_total_bytes,
        "raw_avg_size_B": (raw_total_bytes / raw_total_count)
        if raw_total_count
        else 0.0,
        "packet_hist_approx": histogram_to_str(packetized),
        "packet_count_approx": pkt_total_count,
        "packet_bytes_approx": pkt_total_bytes,
        "packet_avg_size_B_approx": (pkt_total_bytes / pkt_total_count)
        if pkt_total_count
        else 0.0,
    }


def extract_gemm_active_time(log_file):
    if not os.path.exists(log_file):
        return 0.0

    start_ticks = []
    completion_flag_ticks = []
    subproblem_done_ticks = []
    with open(log_file, encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = re.search(r"^(\d+): .*startMatrixCompute:", line)
            if match:
                start_ticks.append(int(match.group(1)))
                continue

            match = re.search(
                r"^(\d+): .*MatrixFlow completion flag written:", line
            )
            if match:
                completion_flag_ticks.append(int(match.group(1)))
                continue

            match = re.search(r"^(\d+): .*MatrixFlow subproblem done:", line)
            if match:
                subproblem_done_ticks.append(int(match.group(1)))

    done_ticks = (
        completion_flag_ticks
        if completion_flag_ticks
        else subproblem_done_ticks
    )

    pair_count = min(len(start_ticks), len(done_ticks))
    if pair_count == 0:
        return 0.0

    total_ticks = 0
    for idx in range(pair_count):
        if done_ticks[idx] > start_ticks[idx]:
            total_ticks += done_ticks[idx] - start_ticks[idx]

    return total_ticks / 1.0e12


def extract_phase_timings(serial_log_file):
    timing_keys = [
        "phase1_ms",
        "phase2_d2h_ms",
        "phase2_h2d_ms",
        "phase2_cxl_inplace_ms",
        "phase2_expand_ms",
        "phase2_softmax_ms",
        "phase2_layernorm_ms",
        "phase2_project_ms",
        "phase2_gelu_ms",
        "phase2_residual_ms",
        "phase2_compact_ms",
        "phase2_non_gemm_ms",
        "phase2_total_ms",
        "phase3_ms",
        "end_to_end_ms",
    ]
    byte_keys = [
        "host_mediated_copy_bytes",
        "staged_block_bytes",
        "gemm_tile_count_per_gemm",
        "gemm_tile_count_total",
        "descriptor_launch_count",
        "doorbell_launch_count",
        "phase1_poll_count",
        "phase3_poll_count",
        "cxl_inplace_access_bytes",
        "score_read_bytes",
        "score_write_bytes",
        "score_read_accesses",
        "score_write_accesses",
        "hidden_read_bytes",
        "hidden_write_bytes",
        "hidden_read_accesses",
        "hidden_write_accesses",
        "residual_read_bytes",
        "residual_write_bytes",
        "residual_read_accesses",
        "residual_write_accesses",
        "mlp_read_bytes",
        "mlp_write_bytes",
        "mlp_read_accesses",
        "mlp_write_accesses",
        "proxy_remote_footprint_bytes",
        "total_proxy_movement_bytes",
    ]

    values = {key: 0.0 for key in timing_keys}
    values.update({key: 0 for key in byte_keys})
    values["phase2_mode"] = "unknown"

    if not os.path.exists(serial_log_file):
        return values

    timing_patterns = {
        key: re.compile(rf"\[Timing\] {re.escape(key)}=([0-9.]+)")
        for key in timing_keys
    }
    byte_patterns = {
        key: re.compile(rf"\[Timing\] {re.escape(key)}=(\d+)")
        for key in byte_keys
    }
    mode_pattern = re.compile(r"\[Timing\] phase2_mode=([A-Za-z0-9_]+)")

    with open(serial_log_file, encoding="utf-8", errors="ignore") as f:
        for line in f:
            mode_match = mode_pattern.search(line)
            if mode_match:
                values["phase2_mode"] = mode_match.group(1)
            for key, pattern in timing_patterns.items():
                match = pattern.search(line)
                if match:
                    values[key] = float(match.group(1))

            for key, pattern in byte_patterns.items():
                match = pattern.search(line)
                if match:
                    values[key] = int(match.group(1))

    return values


def serial_log_has_benchmark(serial_log_file):
    if not os.path.exists(serial_log_file):
        return False
    text = open(serial_log_file, encoding="utf-8", errors="ignore").read()
    return (
        "ViT-inspired Layer Proxy Benchmark" in text
        and "[Phase 1] GEMM1 on device HDM" in text
    )


def write_partial_results(results):
    if not results:
        return

    headers = []
    seen = set()
    for row in results:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                headers.append(key)

    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results)

    with open(RAW_TXT_FILE, "w") as f:
        for r in results:
            f.write(str(r) + "\n")


def load_existing_results():
    if not os.path.exists(CSV_FILE) or os.path.getsize(CSV_FILE) == 0:
        return []

    with open(CSV_FILE, newline="") as f:
        return list(csv.DictReader(f))


def completed_case_keys(results):
    keys = set()
    for row in results:
        preset = row.get("Preset")
        label = row.get("Prefetch Label")
        if preset and label:
            keys.add((preset, label))
    return keys


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if FORCE_RERUN:
        results = []
        completed_cases = set()
    else:
        results = load_existing_results()
        completed_cases = completed_case_keys(results)
    active_presets = [
        preset for preset in PRESETS if preset["name"] in RUN_ONLY_PRESET_NAMES
    ]
    active_cfgs = [
        cfg for cfg in PREFETCH_CONFIGS if cfg["label"] in RUN_ONLY_LABELS
    ]

    print("🚀 启动 ViT-inspired Layer Proxy Sweep")
    print("   架构: CXL Type-3 + 设备侧 DDR5 HDM + device-side MatrixFlow")
    print("   口径: timing-mode 已补丁验证；链路为 64 GB/s 级别原型")
    print(
        "   预设: "
        + ", ".join(
            f"{preset['name']}[S={preset['seq_len']},H={preset['hidden_dim']},"
            f"M={preset['mlp_dim']},heads={preset['num_heads']}]"
            for preset in active_presets
        )
    )
    print(
        "   预取配置: "
        + ", ".join(
            f"{cfg['label']}[mode={cfg['mode']},trigger={cfg['trigger']},"
            f"rowsA={cfg['rows_a']},rowsB={cfg.get('rows_b_by_preset', cfg['rows_b'])},"
            f"carryMax={cfg.get('carry_over_max_rows', 0)},"
            f"carryInflight={cfg.get('carry_over_inherit_inflight', 0)},"
            f"holeLead={cfg.get('hole_fill_lead_rows', 0)},"
            f"vipRows={cfg.get('vip_b_rows_capacity', 0)},"
            f"mhotRows={cfg.get('mhot_b_rows_capacity', 0)},"
            f"shadowRows={cfg.get('coverage_shadow_rows_capacity', 0)},"
            f"writeCBudget={cfg.get('writec_overlap_b_issue_budget_rows', 0)},"
            f"scheduler={cfg.get('ab_scheduler_mode', 'baseline')},"
            f"curQuota={cfg.get('ab_current_protected_b_quota_rows', 4)}]"
            for cfg in active_cfgs
        )
    )
    print("=" * 50)
    if FORCE_RERUN:
        print("[*] FORCE_RERUN=1，本轮忽略已有 roofline_data.csv，重新全量跑当前筛选点")

    ensure_libm5()

    for cfg in PREFETCH_CONFIGS:
        if cfg["label"] not in RUN_ONLY_LABELS:
            print(f"[-] 跳过配置: {cfg['label']}")
            continue
        print(
            f"\n=== 预取配置: {cfg['label']} "
            f"(mode={cfg['mode']}, trigger={cfg['trigger']}, "
            f"rowsA={cfg['rows_a']}, rowsB={cfg['rows_b']}) ==="
        )
        for preset in PRESETS:
            if not should_run_case(preset, cfg):
                print(f"[-] 跳过预设: {preset['name']} / 配置: {cfg['label']}")
                continue
            rows_b = cfg.get("rows_b_by_preset", {}).get(
                preset["name"], cfg["rows_b"]
            )
            case_key = (preset["name"], cfg["label"])
            if case_key in completed_cases:
                print(f"[-] 跳过已完成点: {preset['name']} / 配置: {cfg['label']}")
                continue
            preset_tag = preset["name"].replace(" ", "_").replace("/", "_")
            run_tag = f"{preset_tag}__{cfg['label']}"
            seq_len = preset["seq_len"]

            print(f"\n>>> 正在测试预设: {preset['name']} / 配置: {cfg['label']} <<<")
            modify_c_file(preset)

            print("[*] 正在编译 trigger_gemm (需 libm5.a + libm)...")
            subprocess.run(COMPILE_CMD, shell=True, check=True)

            inject_cmd = f"{INJECT_CMD} --no-compile {seq_len}"
            print(
                "[*] 正在注入镜像 "
                f"(script.sh 将执行 trigger_gemm 0x200000000 {seq_len})..."
            )
            subprocess.run(inject_cmd, shell=True, check=True)

            env = os.environ.copy()
            env["MATRIXFLOW_NEXT_PREFETCH_MODE"] = cfg["mode"]
            env["MATRIXFLOW_NEXT_PREFETCH_TRIGGER"] = cfg["trigger"]
            env["MATRIXFLOW_NEXT_PREFETCH_ROWS_A"] = str(cfg["rows_a"])
            env["MATRIXFLOW_NEXT_PREFETCH_ROWS_B"] = str(rows_b)
            env["MATRIXFLOW_CARRY_OVER_MAX_ROWS"] = str(
                cfg.get("carry_over_max_rows", 0)
            )
            env["MATRIXFLOW_CARRY_OVER_INHERIT_INFLIGHT"] = str(
                cfg.get("carry_over_inherit_inflight", 0)
            )
            env["MATRIXFLOW_HOLE_FILL_LEAD_ROWS"] = str(
                cfg.get("hole_fill_lead_rows", 0)
            )
            env["MATRIXFLOW_WRITEC_OVERLAP_B_ISSUE_BUDGET_ROWS"] = str(
                cfg.get("writec_overlap_b_issue_budget_rows", 0)
            )
            env["MATRIXFLOW_VIP_B_ROWS_CAPACITY"] = str(
                cfg.get("vip_b_rows_capacity", 0)
            )
            env["MATRIXFLOW_MHOT_B_ROWS_CAPACITY"] = str(
                cfg.get("mhot_b_rows_capacity", 0)
            )
            env["MATRIXFLOW_COVERAGE_SHADOW_ROWS_CAPACITY"] = str(
                cfg.get("coverage_shadow_rows_capacity", 0)
            )
            env["MATRIXFLOW_COVERAGE_GATHER_MIN_ISSUE_BUDGET"] = str(
                cfg.get("coverage_gather_min_issue_budget", 0)
            )
            env["MATRIXFLOW_AB_SCHEDULER_MODE"] = cfg.get(
                "ab_scheduler_mode", "baseline"
            )
            env["MATRIXFLOW_AB_A_MIN_CREDIT_ROWS"] = str(
                cfg.get("ab_a_min_credit_rows", 16)
            )
            env["MATRIXFLOW_AB_B_BIAS"] = str(cfg.get("ab_bias_b", 1))
            env["MATRIXFLOW_AB_W_URGENCY"] = str(
                cfg.get("ab_weight_urgency", 4)
            )
            env["MATRIXFLOW_AB_W_DEFICIT"] = str(
                cfg.get("ab_weight_deficit", 3)
            )
            env["MATRIXFLOW_AB_W_REUSE"] = str(cfg.get("ab_weight_reuse", 1))
            env["MATRIXFLOW_AB_W_FALLBACK_RISK"] = str(
                cfg.get("ab_weight_fallback_risk", 2)
            )
            env["MATRIXFLOW_AB_MIN_LAUNCH_ROWS_A"] = str(
                cfg.get("ab_min_launch_rows_a", 0)
            )
            env["MATRIXFLOW_AB_MIN_LAUNCH_ROWS_B"] = str(
                cfg.get("ab_min_launch_rows_b", 0)
            )
            env["MATRIXFLOW_AB_CURRENT_PROTECTED_B_QUOTA_ROWS"] = str(
                cfg.get("ab_current_protected_b_quota_rows", 4)
            )

            metrics = None
            bus_dma = None
            prefetch = None
            serial_log = None
            for attempt in range(1, MAX_INVALID_RUN_RETRIES + 1):
                m5out_dir = os.path.join(OUTPUT_DIR, f"m5out_{run_tag}")
                log_file = os.path.join(
                    OUTPUT_DIR, f"terminal_log_{run_tag}.txt"
                )
                if os.path.exists(m5out_dir):
                    shutil.rmtree(m5out_dir)
                if os.path.exists(log_file):
                    os.remove(log_file)

                cmd = GEM5_CMD_TEMPLATE.format(
                    outdir=m5out_dir, logfile=log_file
                )
                print(
                    f"[*] 正在运行 gem5 仿真，日志存入: {log_file} "
                    f"(attempt {attempt}/{MAX_INVALID_RUN_RETRIES})"
                )
                rc = subprocess.run(cmd, shell=True, env=env)
                if rc.returncode != 0:
                    raise SystemExit(
                        f"gem5 仿真失败，返回码={rc.returncode}。请检查日志: {log_file}"
                    )

                stats_file = os.path.join(m5out_dir, "stats.txt")
                serial_log = os.path.join(m5out_dir, "board.pc.com_1.device")
                metrics = extract_metrics(stats_file, log_file)
                bus_dma = extract_bus_dma_stats(stats_file)
                prefetch = extract_prefetch_stats(stats_file)
                valid_metrics = not (
                    metrics["simSeconds"] == 0.0
                    and metrics["dmaRead"] == 0
                    and metrics["dmaWrite"] == 0
                    and metrics["computeCycles"] == 0
                )
                valid_serial = serial_log_has_benchmark(serial_log)
                if valid_metrics and valid_serial:
                    break

                print(
                    "[!] 无效 run：未进入 benchmark 或未写出有效 ROI 统计，" f"准备重试 {run_tag}"
                )
            else:
                raise SystemExit(
                    f"多次重试后仍未拿到有效结果: {run_tag}。" f"请检查日志: {log_file}"
                )

            phase = extract_phase_timings(serial_log)
            formation = compute_request_formation(preset)
            proxy_gemm_flops = 4.0 * (seq_len**3)
            attention_gemm_flops = (
                4.0
                * preset["seq_len"]
                * preset["seq_len"]
                * preset["hidden_dim"]
            )
            mlp_gemm_flops = (
                4.0
                * preset["seq_len"]
                * preset["hidden_dim"]
                * preset["mlp_dim"]
            )
            softmax_flops = 5.0 * preset["seq_len"] * preset["seq_len"]
            layernorm_flops = 6.0 * preset["seq_len"] * preset["hidden_dim"]
            gelu_flops = 8.0 * preset["seq_len"] * preset["mlp_dim"]
            residual_flops = 1.0 * preset["seq_len"] * preset["hidden_dim"]
            non_gemm_flops = (
                softmax_flops + layernorm_flops + gelu_flops + residual_flops
            )
            total_layer_flops = (
                attention_gemm_flops + mlp_gemm_flops + non_gemm_flops
            )
            roi_latency_s = metrics["simSeconds"]
            gemm_active_s = extract_gemm_active_time(log_file)
            dma_bytes = metrics["dmaRead"] + metrics["dmaWrite"]
            total_movement_bytes = (
                dma_bytes + phase["total_proxy_movement_bytes"]
            )
            compute_time_total_s = (
                (
                    metrics["totalComputeCycles"]
                    if "totalComputeCycles" in metrics
                    else metrics["computeCycles"]
                )
                * 2.0
                / (CLOCK_FREQ_GHZ * 1e9)
                if metrics["computeCycles"] > 0
                else 0.0
            )
            gemm_wall_s = gemm_active_s
            gemm_compute_share = (
                100.0 * compute_time_total_s / gemm_wall_s
                if gemm_wall_s > 0
                else 0.0
            )
            gemm_noncompute_share = (
                100.0 - gemm_compute_share if gemm_wall_s > 0 else 0.0
            )

            peak_gflops = (
                ((proxy_gemm_flops / 2.0) / 1e9)
                / (metrics["computeCycles"] / (CLOCK_FREQ_GHZ * 1e9))
                if metrics["computeCycles"] > 0
                else 0.0
            )
            gemm_effective_gflops = (
                (proxy_gemm_flops / 1e9) / gemm_active_s
                if gemm_active_s > 0
                else 0.0
            )
            gemm_dma_bw_gbps = (
                (dma_bytes / 1e9) / gemm_active_s if gemm_active_s > 0 else 0.0
            )
            layer_gemm_normalized_gflops = (
                ((attention_gemm_flops + mlp_gemm_flops) / 1e9) / gemm_active_s
                if gemm_active_s > 0
                else 0.0
            )
            system_effective_gflops = (
                (total_layer_flops / 1e9) / roi_latency_s
                if roi_latency_s > 0
                else 0.0
            )
            oi = (
                total_layer_flops / total_movement_bytes
                if total_movement_bytes > 0
                else 0.0
            )

            def avg_size(byte_key, acc_key):
                acc = phase[acc_key]
                return phase[byte_key] / acc if acc > 0 else 0.0

            result_row = {
                "Preset": preset["name"],
                "Prefetch Label": cfg["label"],
                "Prefetch Mode": cfg["mode"],
                "Prefetch Trigger": cfg["trigger"],
                "Prefetch Rows A": cfg["rows_a"],
                "Prefetch Rows B": rows_b,
                "Scheduler Mode": cfg.get("ab_scheduler_mode", "baseline"),
                "A Min Credit Rows": cfg.get("ab_a_min_credit_rows", 16),
                "B Bias": cfg.get("ab_bias_b", 1),
                "Score W Urgency": cfg.get("ab_weight_urgency", 4),
                "Score W Deficit": cfg.get("ab_weight_deficit", 3),
                "Score W Reuse": cfg.get("ab_weight_reuse", 1),
                "Score W Fallback Risk": cfg.get("ab_weight_fallback_risk", 2),
                "Min Launch Rows A": cfg.get("ab_min_launch_rows_a", 0),
                "Min Launch Rows B": cfg.get("ab_min_launch_rows_b", 0),
                "Current Protected B Quota": cfg.get(
                    "ab_current_protected_b_quota_rows", 4
                ),
                "VIP B Rows Capacity": cfg.get("vip_b_rows_capacity", 0),
                "MHot B Rows Capacity": cfg.get("mhot_b_rows_capacity", 0),
                "Coverage Shadow Rows Capacity": cfg.get(
                    "coverage_shadow_rows_capacity", 0
                ),
                "SeqLen": preset["seq_len"],
                "HiddenDim": preset["hidden_dim"],
                "MLPDim": preset["mlp_dim"],
                "NumHeads": preset["num_heads"],
                "Phase2 Mode": phase["phase2_mode"],
                "Staged Block Bytes": phase["staged_block_bytes"],
                "GEMM Proxy Size": seq_len,
                "Proxy GEMM FLOPs": int(proxy_gemm_flops),
                "Attention GEMM FLOPs": int(attention_gemm_flops),
                "MLP GEMM FLOPs": int(mlp_gemm_flops),
                "Non-GEMM FLOPs": int(non_gemm_flops),
                "Total FLOPs": int(total_layer_flops),
                "End-to-End ROI Latency (s)": round(roi_latency_s, 6),
                "Total Latency (ms)": round(phase["end_to_end_ms"], 6),
                "Phase 2 Share of Total (%)": round(
                    100.0 * phase["phase2_total_ms"] / phase["end_to_end_ms"]
                    if phase["end_to_end_ms"] > 0
                    else 0.0,
                    4,
                ),
                "GEMM Tiles Per GEMM": phase["gemm_tile_count_per_gemm"],
                "GEMM Total Tiles": phase["gemm_tile_count_total"],
                "Phase-1 GEMM Time (ms)": round(phase["phase1_ms"], 6),
                "D2H Copy Time (ms)": round(phase["phase2_d2h_ms"], 6),
                "H2D Copy Time (ms)": round(phase["phase2_h2d_ms"], 6),
                "CXL In-Place Phase Time (ms)": round(
                    phase["phase2_cxl_inplace_ms"], 6
                ),
                "Expand Time (ms)": round(phase["phase2_expand_ms"], 6),
                "Softmax Time (ms)": round(phase["phase2_softmax_ms"], 6),
                "LayerNorm Time (ms)": round(phase["phase2_layernorm_ms"], 6),
                "Project Time (ms)": round(phase["phase2_project_ms"], 6),
                "GeLU Time (ms)": round(phase["phase2_gelu_ms"], 6),
                "Residual Time (ms)": round(phase["phase2_residual_ms"], 6),
                "Compact Time (ms)": round(phase["phase2_compact_ms"], 6),
                "Non-GEMM Time (ms)": round(phase["phase2_non_gemm_ms"], 6),
                "Phase-2 Total (ms)": round(phase["phase2_total_ms"], 6),
                "Phase-3 GEMM Time (ms)": round(phase["phase3_ms"], 6),
                "Descriptor Launch Count": phase["descriptor_launch_count"],
                "Doorbell Launch Count": phase["doorbell_launch_count"],
                "Phase1 Poll Count": phase["phase1_poll_count"],
                "Phase3 Poll Count": phase["phase3_poll_count"],
                "Device DMA Read Bytes": metrics["dmaRead"],
                "Device DMA Write Bytes": metrics["dmaWrite"],
                "GEMM DMA Bytes": dma_bytes,
                "next_prefetch_issue_count": prefetch[
                    "next_prefetch_issue_count"
                ],
                "next_k_prefetch_issue_count": prefetch[
                    "next_k_prefetch_issue_count"
                ],
                "next_output_prefetch_issue_count": prefetch[
                    "next_output_prefetch_issue_count"
                ],
                "next_prefetch_hit_count": prefetch["next_prefetch_hit_count"],
                "next_output_prefetch_hit_count": prefetch[
                    "next_output_prefetch_hit_count"
                ],
                "next_prefetch_fallback_count": prefetch[
                    "next_prefetch_fallback_count"
                ],
                "next_output_prefetch_fallback_count": prefetch[
                    "next_output_prefetch_fallback_count"
                ],
                "next_prefetch_late_completion_count": prefetch[
                    "next_prefetch_late_completion_count"
                ],
                "next_output_prefetch_late_completion_count": prefetch[
                    "next_output_prefetch_late_completion_count"
                ],
                "next_prefetch_discard_count": prefetch[
                    "next_prefetch_discard_count"
                ],
                "prefetched_b_rows_consumed": prefetch[
                    "prefetched_b_rows_consumed"
                ],
                "fallback_b_rows_fetched": prefetch["fallback_b_rows_fetched"],
                "next_output_prefetch_rows_issued": prefetch[
                    "next_output_prefetch_rows_issued"
                ],
                "next_k_prefetch_rows_issued": prefetch[
                    "next_k_prefetch_rows_issued"
                ],
                "next_output_prefetch_defer_count": prefetch[
                    "next_output_prefetch_defer_count"
                ],
                "next_output_headstart_cycles": prefetch[
                    "next_output_headstart_cycles"
                ],
                "next_output_first_issue_to_boundary_cycles": prefetch[
                    "next_output_first_issue_to_boundary_cycles"
                ],
                "next_output_rows_ready_at_boundary": prefetch[
                    "next_output_rows_ready_at_boundary"
                ],
                "next_output_consumed_before_fallback_rows": prefetch[
                    "next_output_consumed_before_fallback_rows"
                ],
                "carry_over_rows_at_boundary": prefetch[
                    "carry_over_rows_at_boundary"
                ],
                "carry_over_inflight_rows_at_boundary": prefetch[
                    "carry_over_inflight_rows_at_boundary"
                ],
                "carry_over_rows_consumed_post_boundary": prefetch[
                    "carry_over_rows_consumed_post_boundary"
                ],
                "normal_fetch_hole_rows": prefetch["normal_fetch_hole_rows"],
                "duplicate_b_row_fetch_avoided": prefetch[
                    "duplicate_b_row_fetch_avoided"
                ],
                "duplicate_b_row_fetch_detected": prefetch[
                    "duplicate_b_row_fetch_detected"
                ],
                "carry_over_late_completion_count": prefetch[
                    "carry_over_late_completion_count"
                ],
                "normal_fetch_deferred_by_carry": prefetch[
                    "normal_fetch_deferred_by_carry"
                ],
                "writec_overlap_cycles": prefetch["writec_overlap_cycles"],
                "writec_overlap_enabled_count": prefetch[
                    "writec_overlap_enabled_count"
                ],
                "writec_overlap_success_count": prefetch[
                    "writec_overlap_success_count"
                ],
                "next_output_progress_during_writec": prefetch[
                    "next_output_progress_during_writec"
                ],
                "b_rows_issued_during_writec": prefetch[
                    "b_rows_issued_during_writec"
                ],
                "writec_blocked_b_issue_count": prefetch[
                    "writec_blocked_b_issue_count"
                ],
                "a_rows_issued": prefetch["a_rows_issued"],
                "a_rows_ready_before_compute": prefetch[
                    "a_rows_ready_before_compute"
                ],
                "a_rows_inflight_peak": prefetch["a_rows_inflight_peak"],
                "b_rows_inflight_peak": prefetch["b_rows_inflight_peak"],
                "ab_parallel_fetch_overlap_cycles": prefetch[
                    "ab_parallel_fetch_overlap_cycles"
                ],
                "a_fetch_progress_during_b_fetch": prefetch[
                    "a_fetch_progress_during_b_fetch"
                ],
                "b_fetch_progress_during_a_fetch": prefetch[
                    "b_fetch_progress_during_a_fetch"
                ],
                "a_path_stall_waiting_for_b": prefetch[
                    "a_path_stall_waiting_for_b"
                ],
                "b_path_stall_waiting_for_a": prefetch[
                    "b_path_stall_waiting_for_a"
                ],
                "a_credit_floor_hits": prefetch["a_credit_floor_hits"],
                "b_bias_wins": prefetch["b_bias_wins"],
                "urgency_priority_wins": prefetch["urgency_priority_wins"],
                "deficit_priority_wins": prefetch["deficit_priority_wins"],
                "reuse_priority_wins": prefetch["reuse_priority_wins"],
                "fallback_risk_priority_wins": prefetch[
                    "fallback_risk_priority_wins"
                ],
                "score_tie_break_count": prefetch["score_tie_break_count"],
                "avg_score_A": round(float(prefetch["avg_score_a"]), 4),
                "avg_score_B": round(float(prefetch["avg_score_b"]), 4),
                "max_score_A": round(float(prefetch["max_score_a"]), 4),
                "max_score_B": round(float(prefetch["max_score_b"]), 4),
                "protected_b_issue_count": prefetch["protected_b_issue_count"],
                "protected_b_ready_count": prefetch["protected_b_ready_count"],
                "protected_b_priority_wins": prefetch[
                    "protected_b_priority_wins"
                ],
                "protected_b_blocks_a_count": prefetch[
                    "protected_b_blocks_a_count"
                ],
                "protected_b_blocks_normal_b_count": prefetch[
                    "protected_b_blocks_normal_b_count"
                ],
                "protected_b_from_carry_over_count": prefetch[
                    "protected_b_from_carry_over_count"
                ],
                "protected_b_from_next_output_count": prefetch[
                    "protected_b_from_next_output_count"
                ],
                "protected_b_from_hole_filling_count": prefetch[
                    "protected_b_from_hole_filling_count"
                ],
                "protected_b_from_compute_window_count": prefetch[
                    "protected_b_from_compute_window_count"
                ],
                "future_protected_b_issue_count": prefetch[
                    "future_protected_b_issue_count"
                ],
                "future_protected_b_priority_wins": prefetch[
                    "future_protected_b_priority_wins"
                ],
                "future_protected_b_blocks_a_count": prefetch[
                    "future_protected_b_blocks_a_count"
                ],
                "future_protected_b_blocks_current_b_count": prefetch[
                    "future_protected_b_blocks_current_b_count"
                ],
                "future_protected_b_from_next_output_count": prefetch[
                    "future_protected_b_from_next_output_count"
                ],
                "future_protected_b_from_future_hole_filling_count": prefetch[
                    "future_protected_b_from_future_hole_filling_count"
                ],
                "future_protected_b_ready_at_boundary_count": prefetch[
                    "future_protected_b_ready_at_boundary_count"
                ],
                "current_protected_b_issue_count": prefetch[
                    "current_protected_b_issue_count"
                ],
                "current_protected_b_priority_wins": prefetch[
                    "current_protected_b_priority_wins"
                ],
                "current_protected_b_blocks_a_count": prefetch[
                    "current_protected_b_blocks_a_count"
                ],
                "current_protected_b_quota_exhaust_count": prefetch[
                    "current_protected_b_quota_exhaust_count"
                ],
                "current_protected_b_from_compute_window_count": prefetch[
                    "current_protected_b_from_compute_window_count"
                ],
                "current_protected_b_from_current_hole_filling_count": prefetch[
                    "current_protected_b_from_current_hole_filling_count"
                ],
                "b_rows_claimed_by_future": prefetch[
                    "b_rows_claimed_by_future"
                ],
                "future_claim_set_count": prefetch["future_claim_set_count"],
                "future_claim_cleared_count": prefetch[
                    "future_claim_cleared_count"
                ],
                "future_claim_blocked_normal_fetch_count": prefetch[
                    "future_claim_blocked_normal_fetch_count"
                ],
                "future_claim_expired_count": prefetch[
                    "future_claim_expired_count"
                ],
                "future_claim_consumed_success_count": prefetch[
                    "future_claim_consumed_success_count"
                ],
                "future_claim_invalidated_count": prefetch[
                    "future_claim_invalidated_count"
                ],
                "vip_pool_capacity": prefetch["vip_pool_capacity"],
                "vip_pool_occupancy_peak": prefetch["vip_pool_occupancy_peak"],
                "vip_insert_count": prefetch["vip_insert_count"],
                "vip_hit_count": prefetch["vip_hit_count"],
                "vip_miss_count": prefetch["vip_miss_count"],
                "vip_eviction_count": prefetch["vip_eviction_count"],
                "vip_hit_on_next_output_count": prefetch[
                    "vip_hit_on_next_output_count"
                ],
                "vip_hit_on_next_output_immediate_count": prefetch[
                    "vip_hit_on_next_output_immediate_count"
                ],
                "vip_hit_on_next_output_near_count": prefetch[
                    "vip_hit_on_next_output_near_count"
                ],
                "vip_hit_on_claimed_future_b_count": prefetch[
                    "vip_hit_on_claimed_future_b_count"
                ],
                "vip_hit_on_carry_over_b_count": prefetch[
                    "vip_hit_on_carry_over_b_count"
                ],
                "vip_insert_from_next_output_count": prefetch[
                    "vip_insert_from_next_output_count"
                ],
                "vip_insert_from_claim_count": prefetch[
                    "vip_insert_from_claim_count"
                ],
                "vip_insert_from_carry_over_count": prefetch[
                    "vip_insert_from_carry_over_count"
                ],
                "vip_materialize_to_current_count": prefetch[
                    "vip_materialize_to_current_count"
                ],
                "vip_b_rows_served_to_compute": prefetch[
                    "vip_b_rows_served_to_compute"
                ],
                "vip_b_rows_prevented_fallback_count": prefetch[
                    "vip_b_rows_prevented_fallback_count"
                ],
                "vip_strong_admit_count": prefetch["vip_strong_admit_count"],
                "vip_weak_admit_count": prefetch["vip_weak_admit_count"],
                "vip_evict_low_priority_count": prefetch[
                    "vip_evict_low_priority_count"
                ],
                "vip_evict_weak_admit_count": prefetch[
                    "vip_evict_weak_admit_count"
                ],
                "vip_evict_normal_count": prefetch["vip_evict_normal_count"],
                "vip_admit_next_output_immediate_count": prefetch[
                    "vip_admit_next_output_immediate_count"
                ],
                "vip_admit_next_output_near_count": prefetch[
                    "vip_admit_next_output_near_count"
                ],
                "vip_admit_next_output_far_count": prefetch[
                    "vip_admit_next_output_far_count"
                ],
                "vip_admit_claim_count": prefetch["vip_admit_claim_count"],
                "vip_admit_carry_over_count": prefetch[
                    "vip_admit_carry_over_count"
                ],
                "vip_admit_current_window_immediate_count": prefetch[
                    "vip_admit_current_window_immediate_count"
                ],
                "vip_reject_normal_near_count": prefetch[
                    "vip_reject_normal_near_count"
                ],
                "vip_reject_normal_far_count": prefetch[
                    "vip_reject_normal_far_count"
                ],
                "vip_reject_other_count": prefetch["vip_reject_other_count"],
                "oracle_selected_b_rows_count": prefetch[
                    "oracle_selected_b_rows_count"
                ],
                "oracle_selected_rows_served_by_vip_count": prefetch[
                    "oracle_selected_rows_served_by_vip_count"
                ],
                "oracle_selected_rows_missed_by_vip_count": prefetch[
                    "oracle_selected_rows_missed_by_vip_count"
                ],
                "vip_rescue_insert_count": prefetch["vip_rescue_insert_count"],
                "vip_rescue_hit_count": prefetch["vip_rescue_hit_count"],
                "vip_rescue_miss_count": prefetch["vip_rescue_miss_count"],
                "vip_rescue_eviction_count": prefetch[
                    "vip_rescue_eviction_count"
                ],
                "vip_rescue_served_to_compute_count": prefetch[
                    "vip_rescue_served_to_compute_count"
                ],
                "vip_rescue_prevented_fallback_count": prefetch[
                    "vip_rescue_prevented_fallback_count"
                ],
                "vip_rescue_insert_after_fallback_count": prefetch[
                    "vip_rescue_insert_after_fallback_count"
                ],
                "vip_rescue_insert_short_next_use_count": prefetch[
                    "vip_rescue_insert_short_next_use_count"
                ],
                "vip_rescue_insert_multi_future_use_count": prefetch[
                    "vip_rescue_insert_multi_future_use_count"
                ],
                "vip_rescue_reused_count": prefetch["vip_rescue_reused_count"],
                "vip_rescue_a_insert_count": prefetch[
                    "vip_rescue_a_insert_count"
                ],
                "vip_rescue_a_hit_count": prefetch["vip_rescue_a_hit_count"],
                "vip_rescue_a_miss_count": prefetch["vip_rescue_a_miss_count"],
                "vip_rescue_a_eviction_count": prefetch[
                    "vip_rescue_a_eviction_count"
                ],
                "vip_rescue_a_served_to_compute_count": prefetch[
                    "vip_rescue_a_served_to_compute_count"
                ],
                "vip_rescue_a_prevented_remote_fetch_count": prefetch[
                    "vip_rescue_a_prevented_remote_fetch_count"
                ],
                "vip_rescue_a_insert_short_next_use_count": prefetch[
                    "vip_rescue_a_insert_short_next_use_count"
                ],
                "vip_rescue_a_insert_multi_future_use_count": prefetch[
                    "vip_rescue_a_insert_multi_future_use_count"
                ],
                "vip_rescue_a_reused_count": prefetch[
                    "vip_rescue_a_reused_count"
                ],
                "mhot_pool_capacity": prefetch["mhot_pool_capacity"],
                "mhot_occupancy_peak": prefetch["mhot_occupancy_peak"],
                "mhot_insert_count": prefetch["mhot_insert_count"],
                "mhot_insert_default_count": prefetch[
                    "mhot_insert_default_count"
                ],
                "mhot_insert_enhanced_count": prefetch[
                    "mhot_insert_enhanced_count"
                ],
                "mhot_insert_next_output_immediate_count": prefetch[
                    "mhot_insert_next_output_immediate_count"
                ],
                "mhot_insert_next_output_near_count": prefetch[
                    "mhot_insert_next_output_near_count"
                ],
                "mhot_insert_next_output_far_count": prefetch[
                    "mhot_insert_next_output_far_count"
                ],
                "mhot_insert_claim_count": prefetch["mhot_insert_claim_count"],
                "mhot_insert_carry_over_count": prefetch[
                    "mhot_insert_carry_over_count"
                ],
                "mhot_hit_count": prefetch["mhot_hit_count"],
                "mhot_hit_on_next_output_immediate_count": prefetch[
                    "mhot_hit_on_next_output_immediate_count"
                ],
                "mhot_hit_on_next_output_near_count": prefetch[
                    "mhot_hit_on_next_output_near_count"
                ],
                "mhot_hit_on_next_output_far_count": prefetch[
                    "mhot_hit_on_next_output_far_count"
                ],
                "mhot_hit_on_claim_count": prefetch["mhot_hit_on_claim_count"],
                "mhot_hit_on_carry_over_count": prefetch[
                    "mhot_hit_on_carry_over_count"
                ],
                "mhot_rows_served_to_compute_count": prefetch[
                    "mhot_rows_served_to_compute_count"
                ],
                "mhot_materialize_to_current_count": prefetch[
                    "mhot_materialize_to_current_count"
                ],
                "mhot_eviction_count": prefetch["mhot_eviction_count"],
                "mhot_reuse_hit_count": prefetch["mhot_reuse_hit_count"],
                "mhot_mainline_prevented_remote_count": prefetch[
                    "mhot_mainline_prevented_remote_count"
                ],
                "mhot_checked_on_fallback_count": prefetch[
                    "mhot_checked_on_fallback_count"
                ],
                "mhot_miss_then_remote_count": prefetch[
                    "mhot_miss_then_remote_count"
                ],
                "mhot_hit_before_remote_count": prefetch[
                    "mhot_hit_before_remote_count"
                ],
                "rx_a_issue_count": prefetch["rx_a_issue_count"],
                "rx_b_issue_count": prefetch["rx_b_issue_count"],
                "rx_a_ready_count": prefetch["rx_a_ready_count"],
                "rx_b_ready_count": prefetch["rx_b_ready_count"],
                "rx_a_queue_occupancy_peak": prefetch[
                    "rx_a_queue_occupancy_peak"
                ],
                "rx_b_queue_occupancy_peak": prefetch[
                    "rx_b_queue_occupancy_peak"
                ],
                "rx_a_stall_cycles": prefetch["rx_a_stall_cycles"],
                "rx_b_stall_cycles": prefetch["rx_b_stall_cycles"],
                "rx_a_priority_wins": prefetch["rx_a_priority_wins"],
                "rx_b_priority_wins": prefetch["rx_b_priority_wins"],
                "rx_b_deficit_wins": prefetch["rx_b_deficit_wins"],
                "rx_a_deficit_wins": prefetch["rx_a_deficit_wins"],
                "rx_b_deadline_wins": prefetch["rx_b_deadline_wins"],
                "rx_a_deadline_wins": prefetch["rx_a_deadline_wins"],
                "Raw DMA Action Histogram": formation["raw_hist"],
                "Raw DMA Action Count": formation["raw_count"],
                "Raw DMA Action Bytes": formation["raw_bytes"],
                "Raw DMA Action Avg Size (B)": round(
                    formation["raw_avg_size_B"], 4
                ),
                "Approx 64B Packet Histogram": formation["packet_hist_approx"],
                "Approx Packet Count": formation["packet_count_approx"],
                "Approx Packet Bytes": formation["packet_bytes_approx"],
                "Approx Packet Avg Size (B)": round(
                    formation["packet_avg_size_B_approx"], 4
                ),
                "Bus DMA Packet Count": bus_dma["bus_dma_pkt_count"],
                "Bus DMA Packet Bytes": bus_dma["bus_dma_pkt_bytes"],
                "Bus Avg Packet Size (B)": round(
                    (
                        bus_dma["bus_dma_pkt_bytes"]
                        / bus_dma["bus_dma_pkt_count"]
                    )
                    if bus_dma["bus_dma_pkt_count"] > 0
                    else 0.0,
                    4,
                ),
                "Bus Avg Request Payload Size (B)": round(
                    (
                        bus_dma["bus_dma_pkt_bytes"]
                        / (bus_dma["bus_dma_pkt_count"] / 2.0)
                    )
                    if bus_dma["bus_dma_pkt_count"] > 0
                    else 0.0,
                    4,
                ),
                "memToCXLCtrlRsp mean (cycles)": round(
                    float(bus_dma["mem_rsp_gap_mean_cycles"]), 6
                ),
                "memToCXLCtrlRsp stdev (cycles)": round(
                    float(bus_dma["mem_rsp_gap_stdev_cycles"]), 6
                ),
                "memToCXLCtrlRsp min (cycles)": round(
                    float(bus_dma["mem_rsp_gap_min_cycles"]), 6
                ),
                "memToCXLCtrlRsp max (cycles)": round(
                    float(bus_dma["mem_rsp_gap_max_cycles"]), 6
                ),
                "Host-Mediated Copy Bytes": phase["host_mediated_copy_bytes"],
                "CXL In-Place Access Bytes": phase["cxl_inplace_access_bytes"],
                "Score Read Bytes": phase["score_read_bytes"],
                "Score Write Bytes": phase["score_write_bytes"],
                "Score Read Accesses": phase["score_read_accesses"],
                "Score Write Accesses": phase["score_write_accesses"],
                "Score Avg Read Size (B/access)": round(
                    avg_size("score_read_bytes", "score_read_accesses"), 4
                ),
                "Score Avg Write Size (B/access)": round(
                    avg_size("score_write_bytes", "score_write_accesses"), 4
                ),
                "Hidden Read Bytes": phase["hidden_read_bytes"],
                "Hidden Write Bytes": phase["hidden_write_bytes"],
                "Hidden Read Accesses": phase["hidden_read_accesses"],
                "Hidden Write Accesses": phase["hidden_write_accesses"],
                "Hidden Avg Read Size (B/access)": round(
                    avg_size("hidden_read_bytes", "hidden_read_accesses"), 4
                ),
                "Hidden Avg Write Size (B/access)": round(
                    avg_size("hidden_write_bytes", "hidden_write_accesses"), 4
                ),
                "Residual Read Bytes": phase["residual_read_bytes"],
                "Residual Write Bytes": phase["residual_write_bytes"],
                "Residual Read Accesses": phase["residual_read_accesses"],
                "Residual Write Accesses": phase["residual_write_accesses"],
                "Residual Avg Read Size (B/access)": round(
                    avg_size("residual_read_bytes", "residual_read_accesses"),
                    4,
                ),
                "Residual Avg Write Size (B/access)": round(
                    avg_size(
                        "residual_write_bytes", "residual_write_accesses"
                    ),
                    4,
                ),
                "MLP Read Bytes": phase["mlp_read_bytes"],
                "MLP Write Bytes": phase["mlp_write_bytes"],
                "MLP Read Accesses": phase["mlp_read_accesses"],
                "MLP Write Accesses": phase["mlp_write_accesses"],
                "MLP Avg Read Size (B/access)": round(
                    avg_size("mlp_read_bytes", "mlp_read_accesses"), 4
                ),
                "MLP Avg Write Size (B/access)": round(
                    avg_size("mlp_write_bytes", "mlp_write_accesses"), 4
                ),
                "Proxy Remote Footprint Bytes": phase[
                    "proxy_remote_footprint_bytes"
                ],
                "Phase2 Total Bytes": phase["total_proxy_movement_bytes"],
                "Total Data Movement Bytes": total_movement_bytes,
                "totalComputeCycles": metrics["computeCycles"],
                "GEMM Compute Share (%)": round(gemm_compute_share, 4),
                "GEMM Non-Compute Share (%)": round(gemm_noncompute_share, 4),
                "Operational Intensity (OI)": round(oi, 4),
                "Peak MAC Throughput (GFLOPS)": round(peak_gflops, 6),
                "GEMM DMA BW (GB/s)": round(gemm_dma_bw_gbps, 6),
                "GEMM-Effective (GFLOPS)": round(gemm_effective_gflops, 6),
                "Layer-GEMM-Normalized (GFLOPS)": round(
                    layer_gemm_normalized_gflops, 6
                ),
                "End-to-End Effective (GFLOPS)": round(
                    system_effective_gflops, 6
                ),
            }
            results.append(result_row)
            completed_cases.add(case_key)

            print(
                f"[√] {preset['name']} / {cfg['label']} 完成! "
                f"Phase1={result_row['Phase-1 GEMM Time (ms)']:.3f}ms, "
                f"Phase2={result_row['Phase-2 Total (ms)']:.3f}ms, "
                f"GEMM-Effective={result_row['GEMM-Effective (GFLOPS)']:.3f} GFLOPS, "
                f"End-to-End Effective={result_row['End-to-End Effective (GFLOPS)']:.3f} GFLOPS"
            )
            write_partial_results(results)

    # 5. 保存到 CSV 和 TXT
    if results:
        write_partial_results(results)
        print("\n" + "=" * 50)
        print(f"🎉 Sweep 完成！数据已保存至: {CSV_FILE}")


if __name__ == "__main__":
    main()
