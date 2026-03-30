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
