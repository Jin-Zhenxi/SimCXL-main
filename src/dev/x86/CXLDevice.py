import os

from m5.objects.MatrixFlowEngine import MatrixFlowEngine
from m5.objects.PciDevice import *
from m5.objects.XBar import CXLMemBar
from m5.params import *
from m5.proxy import *


class CXLMemCtrl(PciDevice):
    type = "CXLMemCtrl"
    cxx_header = "dev/x86/cxl_mem_ctrl.hh"
    cxx_class = "gem5::CXLMemCtrl"

    cxl_rsp_port = ResponsePort(
        "This port sends responses to and receives requests from the Host"
    )
    mem_req_port = RequestPort(
        "This port sends requests to and receives responses from the back-end memory media"
    )

    rsp_size = Param.Unsigned(48, "The number of responses to buffer")
    req_size = Param.Unsigned(48, "The number of requests to buffer")

    proto_proc_lat = Param.Latency(
        "15ns",
        "Latency of the CXL controller processing CXL.mem sub-protocol packets",
    )
    cxl_mem_range = Param.AddrRange(
        "2GB",
        "CXL expander memory range that can be identified as system memory",
    )

    matrix_engine = Param.MatrixFlowEngine(
        MatrixFlowEngine(
            mac_array_size=16,
            compute_latency_per_op=1,
            min_read_request_bytes=64,
            next_prefetch_mode=os.getenv(
                "MATRIXFLOW_NEXT_PREFETCH_MODE", "b_only"
            ),
            next_prefetch_trigger=os.getenv(
                "MATRIXFLOW_NEXT_PREFETCH_TRIGGER", "b_ready"
            ),
            next_prefetch_rows_a=int(
                os.getenv("MATRIXFLOW_NEXT_PREFETCH_ROWS_A", "0")
            ),
            next_prefetch_rows_b=int(
                os.getenv("MATRIXFLOW_NEXT_PREFETCH_ROWS_B", "0")
            ),
            carry_over_max_rows=int(
                os.getenv("MATRIXFLOW_CARRY_OVER_MAX_ROWS", "0")
            ),
            carry_over_inherit_inflight=(
                os.getenv("MATRIXFLOW_CARRY_OVER_INHERIT_INFLIGHT", "1") != "0"
            ),
            hole_fill_lead_rows=int(
                os.getenv("MATRIXFLOW_HOLE_FILL_LEAD_ROWS", "0")
            ),
            writec_overlap_b_issue_budget_rows=int(
                os.getenv("MATRIXFLOW_WRITEC_OVERLAP_B_ISSUE_BUDGET_ROWS", "0")
            ),
        ),
        "MatrixFlow compute engine (PNM accelerator)",
    )

    VendorID = 0x8086
    DeviceID = 0x7890
    Command = 0x0
    Status = 0x280
    Revision = 0x0
    ClassCode = 0x05
    SubClassCode = 0x00
    ProgIF = 0x00
    InterruptLine = 0x1F
    InterruptPin = 0x01

    # Primary
    BAR0 = PciMemBar(size="2GB")
    BAR1 = PciMemUpperBar()
    BAR2 = PciMemBar(size="2MiB")
    BAR3 = PciMemUpperBar()
    BAR4 = PciMemBar(size="512KiB")
    BAR5 = PciMemUpperBar()

    def connectMemory(self, cxl_mem_range, cxl_memory):
        self.cxl_mem_range = cxl_mem_range
        self.BAR0.size = cxl_memory.get_size_str()
        self.cxl_mem_bus = CXLMemBar()
        self.cxl_mem_bus.cpu_side_ports = [
            self.mem_req_port,
            self.matrix_engine.dma_port,
        ]
        for _, port in cxl_memory.get_mem_ports():
            self.cxl_mem_bus.mem_side_ports = port

    def configCXL(self, proc_lat, queue_size):
        self.proto_proc_lat = proc_lat
        self.rsp_size = queue_size
        self.req_size = queue_size


class CXLType1Accel(PciDevice):
    type = "CXLType1Accel"
    cxx_header = "dev/x86/cxl_type1_accel.hh"
    cxx_class = "gem5::CXLType1Accel"
    dcache_port = RequestPort("Data Port")
    icache_port = RequestPort("Intr Port")
    cacheline_size = Param.Int(64, "Device cache line size ")
    lsu_mode = Param.Int(
        1,
        "1 for single-point access; 2 for sequential access; 3 for random access",
    )
    lsu_num = Param.Int(1, "Number of LSU sending requests to the host")
    load_store = Param.Int(1, "1 for load; 2 for store")

    VendorID = 0x8086
    DeviceID = 0x7890
    Command = 0x0
    Status = 0x280
    Revision = 0x0
    ClassCode = 0x05
    SubClassCode = 0x00
    ProgIF = 0x00
    InterruptLine = 0x1F
    InterruptPin = 0x01

    # Primary
    BAR0 = PciMemBar(size="4MiB")
    BAR1 = PciMemUpperBar()

    def connectCachedPorts(self, in_ports):
        self.dcache_port = in_ports

    def configCXL(self, lsu_mode, lsu_num, load_store):
        self.lsu_mode = lsu_mode
        self.lsu_num = lsu_num
        self.load_store = load_store


class CXLType2Accel(PciDevice):
    type = "CXLType2Accel"
    cxx_header = "dev/x86/cxl_type2_accel.hh"
    cxx_class = "gem5::CXLType2Accel"
    dcache_port = RequestPort("Data Port")
    icache_port = RequestPort("Intr Port")
    cacheline_size = Param.Int(64, "Device cache line size ")
    cxl_rsp_port = ResponsePort(
        "This port sends responses to and receives requests from the Host"
    )
    mem_req_port = RequestPort(
        "This port sends requests to and receives responses from the back-end memory media"
    )
    rsp_size = Param.Unsigned(48, "The number of responses to buffer")
    req_size = Param.Unsigned(48, "The number of requests to buffer")
    proto_proc_lat = Param.Latency(
        "15ns",
        "Latency of the CXL controller processing CXL.mem sub-protocol packets",
    )
    cxl_mem_range = Param.AddrRange(
        "2GB",
        "CXL expander memory range that can be identified as system memory",
    )
    lsu_mode = Param.Int(
        1,
        "1 for single-point access; 2 for sequential access; 3 for random access",
    )
    lsu_num = Param.Int(1, "Number of LSU sending requests to the host")
    load_store = Param.Int(1, "1 for load; 2 for store")

    VendorID = 0x8086
    DeviceID = 0x7890
    Command = 0x0
    Status = 0x280
    Revision = 0x0
    ClassCode = 0x05
    SubClassCode = 0x00
    ProgIF = 0x00
    InterruptLine = 0x1F
    InterruptPin = 0x01

    # Primary
    BAR0 = PciMemBar(size="2GB")
    BAR1 = PciMemUpperBar()
    BAR2 = PciMemBar(size="2MiB")
    BAR3 = PciMemUpperBar()
    BAR4 = PciMemBar(size="512KiB")
    BAR5 = PciMemUpperBar()

    def connectCachedPorts(self, in_ports):
        self.dcache_port = in_ports

    def connectMemory(self, cxl_mem_range, cxl_memory):
        self.cxl_mem_range = cxl_mem_range
        self.BAR0.size = cxl_memory.get_size_str()
        self.cxl_mem_bus = CXLMemBar()
        self.cxl_mem_bus.cpu_side_ports = self.mem_req_port
        for _, port in cxl_memory.get_mem_ports():
            self.cxl_mem_bus.mem_side_ports = port

    def configCXL(self, proc_lat, queue_size, lsu_mode, lsu_num, load_store):
        self.proto_proc_lat = proc_lat
        self.rsp_size = queue_size
        self.req_size = queue_size
        self.lsu_mode = lsu_mode
        self.lsu_num = lsu_num
        self.load_store = load_store
