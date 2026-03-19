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
