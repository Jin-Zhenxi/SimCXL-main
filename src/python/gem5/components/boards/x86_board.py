# Copyright (c) 2021 The Regents of the University of California
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met: redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer;
# redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution;
# neither the name of the copyright holders nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import os
from typing import (
    List,
    Sequence,
)

from m5.objects import (
    Addr,
    AddrRange,
    BaseXBar,
    Bridge,
    CowDiskImage,
    CXLBridge,
    CXLMemCtrl,
    IdeDisk,
    IOXBar,
    NoncoherentXBar,
    Pc,
    Port,
    RawDiskImage,
    SrcClockDomain,
    X86E820Entry,
    X86FsLinux,
    X86IntelMPBus,
    X86IntelMPBusHierarchy,
    X86IntelMPIOAPIC,
    X86IntelMPIOIntAssignment,
    X86IntelMPProcessor,
    X86SMBiosBiosInformation,
)
from m5.params import Latency
from m5.util.convert import toMemorySize

# 已移除 EtherTap/EtherLink/IGbE：使用磁盘镜像传入测试程序，无需网络
from ...isas import ISA
from ...resources.resource import AbstractResource
from ...utils.override import overrides
from ..cachehierarchies.abstract_cache_hierarchy import AbstractCacheHierarchy
from ..memory.abstract_memory_system import AbstractMemorySystem
from ..processors.abstract_processor import AbstractProcessor
from .abstract_system_board import AbstractSystemBoard
from .kernel_disk_workload import KernelDiskWorkload


class X86Board(AbstractSystemBoard, KernelDiskWorkload):
    """
    A board capable of full system simulation for X86.

    **Limitations**
    * Currently, this board's memory is hardcoded to 3GB.
    * Much of the I/O subsystem is hard coded.
    """

    def __init__(
        self,
        clk_freq: str,
        processor: AbstractProcessor,
        memory: AbstractMemorySystem,
        cache_hierarchy: AbstractCacheHierarchy,
        cxl_memory: AbstractMemorySystem,
        is_asic: bool = True,
        cxl_host_link_width_bytes: int = 32,
        host_link_window_bytes: int = 0,
    ) -> None:
        self._cxl_memory_ptr = cxl_memory
        self._is_asic = is_asic
        self._cxl_host_link_width_bytes = cxl_host_link_width_bytes
        self._host_link_window_bytes = host_link_window_bytes
        self._host_link_window_range = None
        self._advertise_cxl_mem_to_guest = False
        self._cxl_mem_range = None
        self._enable_cxl_host_dma_bridge = (
            os.environ.get("MATRIXFLOW_HOST_DMA_BRIDGE", "0") == "1"
        )

        super().__init__(
            clk_freq=clk_freq,
            processor=processor,
            memory=memory,
            cache_hierarchy=cache_hierarchy,
        )
        self.cxl_memory = self._cxl_memory_ptr

        if self.get_processor().get_isa() != ISA.X86:
            raise Exception(
                "The X86Board requires a processor using the X86 "
                f"ISA. Current processor ISA: '{processor.get_isa().name}'."
            )

    @overrides(AbstractSystemBoard)
    def _setup_board(self) -> None:
        self.pc = Pc()
        # cxl_device is dynamically initialized and attached
        self.pc.south_bridge.cxl_device = CXLMemCtrl(
            pci_func=0, pci_dev=6, pci_bus=0
        )

        self.workload = X86FsLinux()

        # North Bridge
        self.iobus = IOXBar()

        # Set up all of the I/O.
        self._setup_io_devices()

        self.m5ops_base = 0xFFFF0000

    def _setup_io_devices(self):
        """Sets up the x86 IO devices.

        .. note::

            This is mostly copy-paste from prior X86 FS setups. Some of it
            may not be documented and there may be bugs.
        """

        # Constants similar to x86_traits.hh
        IO_address_space_base = 0x8000000000000000
        pci_config_address_space_base = 0xC000000000000000
        interrupts_address_space_base = 0xA000000000000000
        APIC_range_size = 1 << 12

        # Configure CXL Device
        cxl_dram = self._cxl_memory_ptr
        cxl_mem_range = AddrRange(Addr(0x400000000), size=cxl_dram.get_size())
        self._cxl_mem_range = cxl_mem_range
        cxl_dram.set_memory_range([cxl_mem_range])
        cxl_mem_ctrl = self.pc.south_bridge.cxl_device
        cxl_mem_ctrl.connectMemory(cxl_mem_range, cxl_dram)
        cxl_abstract_mems = []
        for mc in cxl_dram.get_memory_controllers():
            cxl_abstract_mems.append(mc.dram)
        self.memories.extend(cxl_abstract_mems)

        # ASIC：设备协议延迟按硅侧常见值；Phase2 CPU 经 CXLBridge 承受真实链路惩罚
        if self._is_asic:
            cxl_mem_ctrl.configCXL(Latency("15ns"), 128)
        else:
            cxl_mem_ctrl.configCXL(Latency("60ns"), 36)

        # Setup memory system specific settings.
        if self.get_cache_hierarchy().is_ruby():
            self.pc.attachIO(
                self.get_io_bus(),
                [
                    self.pc.south_bridge.ide.dma,
                    cxl_mem_ctrl.dma,
                ],
            )
        else:
            # # Constants similar to x86_traits.hh
            IO_address_space_base = 0x8000000000000000
            pci_config_address_space_base = 0xC000000000000000
            interrupts_address_space_base = 0xA000000000000000
            APIC_range_size = 1 << 12
            cxl_bar0_range = AddrRange(0x200000000, size=cxl_dram.get_size())

            # Keep the legacy system bridge for normal southbridge/PCI/IO traffic.
            self.bridge = Bridge(delay="50ns")
            self.bridge.mem_side_port = self.get_io_bus().cpu_side_ports
            self.bridge.cpu_side_port = (
                self.get_cache_hierarchy().get_mem_side_port()
            )
            self.bridge.ranges = [
                AddrRange(0xC0000000, 0xFFFF0000),
                AddrRange(
                    IO_address_space_base, interrupts_address_space_base - 1
                ),
                AddrRange(pci_config_address_space_base, Addr.max),
            ]

            # Model the host-device link with a dedicated 2 GHz fabric clock.
            # Effective bandwidth is width(B) * 2 GHz.
            self.pcie_fabric_clk_domain = SrcClockDomain()
            self.pcie_fabric_clk_domain.clock = "2GHz"
            self.pcie_fabric_clk_domain.voltage_domain = (
                self.clk_domain.voltage_domain
            )

            # XBar：保持 1 cycle（本地 fabric 极薄）；CXLBridge：恢复长链路延迟供 CPU 访存吃满惩罚
            self.cxl_xbar = NoncoherentXBar(
                width=self._cxl_host_link_width_bytes,
                frontend_latency=1,
                forward_latency=1,
                response_latency=1,
                header_latency=1,
            )
            self.cxl_xbar.clk_domain = self.pcie_fabric_clk_domain

            self.cxl_bridge = CXLBridge(
                bridge_lat="50ns",
                proto_proc_lat="12ns",
                req_fifo_depth=128,
                resp_fifo_depth=128,
                optimal_pkt_size=256,
                small_pkt_size=64,
                small_pkt_overhead_pct=0,
                large_pkt_size=4096,
                large_pkt_overhead_pct=36,
            )
            self.cxl_bridge.clk_domain = self.pcie_fabric_clk_domain
            self.cxl_bridge.cpu_side_port = (
                self.get_cache_hierarchy().get_mem_side_port()
            )
            self.cxl_bridge.mem_side_port = self.cxl_xbar.cpu_side_ports
            self.cxl_bridge.ranges = [
                cxl_bar0_range,
                cxl_mem_range,
            ]

            if self._enable_cxl_host_dma_bridge:
                self.host_dma_bridge = CXLBridge(
                    bridge_lat="50ns",
                    proto_proc_lat="12ns",
                    req_fifo_depth=128,
                    resp_fifo_depth=128,
                    optimal_pkt_size=256,
                    small_pkt_size=64,
                    small_pkt_overhead_pct=0,
                    large_pkt_size=4096,
                    large_pkt_overhead_pct=36,
                )
                self.host_dma_bridge.clk_domain = self.pcie_fabric_clk_domain
                self.host_dma_bridge.cpu_side_port = (
                    self.cxl_xbar.mem_side_ports
                )
                self.host_dma_bridge.mem_side_port = (
                    self.get_cache_hierarchy().get_cpu_side_port()
                )
                self.host_dma_bridge.ranges = [self.mem_ranges[0]]
                cxl_mem_ctrl.dma = self.cxl_xbar.cpu_side_ports

            self.apicbridge = Bridge(delay="50ns")
            self.apicbridge.cpu_side_port = self.get_io_bus().mem_side_ports
            self.apicbridge.mem_side_port = (
                self.get_cache_hierarchy().get_cpu_side_port()
            )
            self.apicbridge.ranges = [
                AddrRange(
                    interrupts_address_space_base,
                    interrupts_address_space_base
                    + self.get_processor().get_num_cores() * APIC_range_size
                    - 1,
                )
            ]
            self.pc.south_bridge.skip_cxl_attach = True
            self.pc.attachIO(self.get_io_bus())

            # SouthBridge.attachIO() temporarily wires the CXL device back onto
            # the shared iobus unless explicitly skipped above; keep BAR0 PIO
            # accesses and the HDM data-plane requests on the private CXL fabric.
            cxl_mem_ctrl.pio = self.cxl_xbar.mem_side_ports
            cxl_mem_ctrl.cxl_rsp_port = self.cxl_xbar.mem_side_ports

        # Add in a Bios information structure.
        self.workload.smbios_table.structures = [X86SMBiosBiosInformation()]

        # Set up the Intel MP table
        base_entries = []
        ext_entries = []
        for i in range(self.get_processor().get_num_cores()):
            bp = X86IntelMPProcessor(
                local_apic_id=i,
                local_apic_version=0x14,
                enable=True,
                bootstrap=(i == 0),
            )
            base_entries.append(bp)
        io_apic = X86IntelMPIOAPIC(
            id=self.get_processor().get_num_cores(),
            version=0x11,
            enable=True,
            address=0xFEC00000,
        )

        self.pc.south_bridge.io_apic.apic_id = io_apic.id
        base_entries.append(io_apic)
        pci_bus = X86IntelMPBus(bus_id=0, bus_type="PCI   ")
        base_entries.append(pci_bus)
        isa_bus = X86IntelMPBus(bus_id=1, bus_type="ISA   ")
        base_entries.append(isa_bus)
        connect_busses = X86IntelMPBusHierarchy(
            bus_id=1, subtractive_decode=True, parent_bus=0
        )
        ext_entries.append(connect_busses)

        # pci_dev3 (ethernet) 已移除，不再添加中断
        pci_dev4_inta = X86IntelMPIOIntAssignment(
            interrupt_type="INT",
            polarity="ConformPolarity",
            trigger="ConformTrigger",
            source_bus_id=0,
            source_bus_irq=0 + (4 << 2),
            dest_io_apic_id=io_apic.id,
            dest_io_apic_intin=16,
        )

        base_entries.append(pci_dev4_inta)

        def assignISAInt(irq, apicPin):
            assign_8259_to_apic = X86IntelMPIOIntAssignment(
                interrupt_type="ExtInt",
                polarity="ConformPolarity",
                trigger="ConformTrigger",
                source_bus_id=1,
                source_bus_irq=irq,
                dest_io_apic_id=io_apic.id,
                dest_io_apic_intin=0,
            )
            base_entries.append(assign_8259_to_apic)

            assign_to_apic = X86IntelMPIOIntAssignment(
                interrupt_type="INT",
                polarity="ConformPolarity",
                trigger="ConformTrigger",
                source_bus_id=1,
                source_bus_irq=irq,
                dest_io_apic_id=io_apic.id,
                dest_io_apic_intin=apicPin,
            )
            base_entries.append(assign_to_apic)

        assignISAInt(0, 2)
        assignISAInt(1, 1)

        for i in range(3, 15):
            assignISAInt(i, i)

        self.workload.intel_mp_table.base_entries = base_entries
        self.workload.intel_mp_table.ext_entries = ext_entries

        host_link_window_start = None
        if self._host_link_window_bytes:
            total_mem_size = int(self.mem_ranges[0].size())
            if self._host_link_window_bytes >= total_mem_size:
                raise Exception(
                    "host_link_window_bytes must be smaller than system memory"
                )
            host_link_window_start = (
                int(self.mem_ranges[0].end())
                + 1
                - self._host_link_window_bytes
            )
            self._host_link_window_range = AddrRange(
                Addr(host_link_window_start),
                size=self._host_link_window_bytes,
            )

        usable_mem_size = (
            host_link_window_start - 0x100000
            if host_link_window_start is not None
            else int(self.mem_ranges[0].size()) - 0x100000
        )

        entries = [
            # Mark the first megabyte of memory as reserved
            X86E820Entry(addr=0, size="639kB", range_type=1),
            X86E820Entry(addr=0x9FC00, size="385kB", range_type=2),
            # Mark the rest of physical memory as available
            X86E820Entry(
                addr=0x100000,
                size=f"{usable_mem_size:d}B",
                range_type=1,
            ),
        ]

        # Reserve the last 16kB of the 32-bit address space for m5ops
        entries.append(
            X86E820Entry(addr=0xFFFF0000, size="64kB", range_type=2)
        )

        # Keep the CXL.mem range physically connected to the controller and
        # bridge, but do not advertise it to the guest OS as system memory.
        # This isolates background Linux traffic from the hidden HDM region so
        # the MatrixEngine doorbell path can be validated in a clean setup.
        if (
            getattr(self, "_advertise_cxl_mem_to_guest", False)
            and self._cxl_mem_range is not None
        ):
            entries.append(
                X86E820Entry(
                    addr=self._cxl_mem_range.start(),
                    size=f"{self._cxl_mem_range.size()}B",
                    range_type=20,
                )
            )

        if self._host_link_window_range is not None:
            entries.append(
                X86E820Entry(
                    addr=self._host_link_window_range.start(),
                    size=f"{self._host_link_window_range.size()}B",
                    range_type=2,
                )
            )

        self.workload.e820_table.entries = entries

    @overrides(AbstractSystemBoard)
    def has_io_bus(self) -> bool:
        return True

    @overrides(AbstractSystemBoard)
    def get_io_bus(self) -> BaseXBar:
        return self.iobus

    @overrides(AbstractSystemBoard)
    def has_dma_ports(self) -> bool:
        return True

    @overrides(AbstractSystemBoard)
    def get_dma_ports(self) -> Sequence[Port]:
        return [
            self.pc.south_bridge.ide.dma,
            self.iobus.mem_side_ports,
            self.pc.south_bridge.cxl_device.dma,
        ]

    @overrides(AbstractSystemBoard)
    def has_coherent_io(self) -> bool:
        return True

    @overrides(AbstractSystemBoard)
    def get_mem_side_coherent_io_port(self) -> Port:
        return self.iobus.mem_side_ports

    @overrides(AbstractSystemBoard)
    def _setup_memory_ranges(self):
        memory = self.get_memory()

        if memory.get_size() > toMemorySize("3GB"):
            raise Exception(
                "X86Board currently only supports memory sizes up "
                "to 3GB because of the I/O hole."
            )
        data_range = AddrRange(memory.get_size())
        memory.set_memory_range([data_range])
        cpu_abstract_mems = []
        for mc in memory.get_memory_controllers():
            cpu_abstract_mems.append(mc.dram)
        self.memories = cpu_abstract_mems

        # Add the address range for the IO
        self.mem_ranges = [
            data_range,  # All data
            AddrRange(0xC0000000, size=0x100000),  # For I/0
        ]

    @overrides(KernelDiskWorkload)
    def get_disk_device(self):
        return "/dev/sda1"

    @overrides(KernelDiskWorkload)
    def _add_disk_to_board(self, disk_image: AbstractResource):
        ide_disk = IdeDisk()
        ide_disk.driveID = "device0"
        ide_disk.image = CowDiskImage(
            child=RawDiskImage(read_only=True), read_only=False
        )
        ide_disk.image.child.image_file = disk_image.get_local_path()

        # Attach the SimObject to the system.
        self.pc.south_bridge.ide.disks = [ide_disk]

    @overrides(KernelDiskWorkload)
    def get_default_kernel_args(self) -> List[str]:
        return [
            "earlyprintk=ttyS0",
            "console=ttyS0",
            "lpj=7999923",
            "root={root_value}",
            "disk_device={disk_device}",
        ]
