# Copyright (c) 2021 The Regents of the University of California
# All rights reserved.

"""
PCIe + HBM baseline for MatrixFlow full-system runs.

This variant mirrors x86-cxl-pcie-hbm.py but automatically falls back to an
ATOMIC boot CPU when /dev/kvm is unavailable, which is common on shared
servers and inside restricted environments.
"""

import argparse
import os

import m5
from gem5.components.boards.x86_board import X86Board
from gem5.components.boards.mem_mode import MemMode
from gem5.components.cachehierarchies.classic.private_l1_private_l2_shared_l3_cache_hierarchy import (
    PrivateL1PrivateL2SharedL3CacheHierarchy,
)
from gem5.components.memory.memory import ChanneledMemory
from gem5.components.memory.dram_interfaces.hbm import Ideal_CXL_HBM2
from gem5.components.memory.single_channel import (
    DIMM_DDR5_4400,
    SingleChannelDDR4_3200,
)
from gem5.components.processors.cpu_types import CPUTypes
from gem5.components.processors.simple_switchable_processor import (
    SimpleSwitchableProcessor,
)
from gem5.isas import ISA
from gem5.resources.resource import DiskImageResource, KernelResource
from gem5.simulate.exit_event import ExitEvent
from gem5.simulate.simulator import Simulator
from gem5.utils.requires import requires


requires(isa_required=ISA.X86)

parser = argparse.ArgumentParser(description="PCIe-HBM baseline parameters.")
parser.add_argument(
    "--is_asic",
    action="store",
    type=str,
    nargs="?",
    choices=["True", "False"],
    default="True",
    help="Choose to simulate CXL ASIC Device or FPGA Device.",
)
parser.add_argument(
    "--num_cpus", type=int, default=1, help="Number of CPUs"
)
parser.add_argument(
    "--cpu_type",
    type=str,
    choices=["TIMING", "O3"],
    default="TIMING",
    help="CPU type",
)
parser.add_argument(
    "--manual",
    action="store_true",
    help="Manual debug mode: no m5 exit, keep sim running for telnet",
)
parser.add_argument(
    "--no-network",
    action="store_true",
    help="Use EtherTapStub instead of EtherTap.",
)
args = parser.parse_args()

if args.no_network:
    os.environ["GEM5_USE_ETHERtap_STUB"] = "1"

cache_hierarchy = PrivateL1PrivateL2SharedL3CacheHierarchy(
    l1d_size="48kB",
    l1d_assoc=6,
    l1i_size="32kB",
    l1i_assoc=8,
    l2_size="2MB",
    l2_assoc=16,
    l3_size="96MB",
    l3_assoc=48,
)

# X86Board hard-limits host-visible DRAM to 3GB because of the legacy I/O hole.
# Keep the host side minimal and put the large capacity on the device-side HBM.
memory = DIMM_DDR5_4400(size="3GB")

# Device-attached memory models the expensive 8-channel HBM on the accelerator.
cxl_dram = ChanneledMemory(Ideal_CXL_HBM2, 8, 64, size="8GB")
if args.is_asic == "False":
    cxl_dram = SingleChannelDDR4_3200(size="8GB")

starting_core_type = CPUTypes.KVM
if not (os.path.exists("/dev/kvm") and os.access("/dev/kvm", os.R_OK | os.W_OK)):
    print("WARN: /dev/kvm unavailable; falling back to ATOMIC boot CPU.")
    starting_core_type = CPUTypes.ATOMIC

processor = SimpleSwitchableProcessor(
    starting_core_type=starting_core_type,
    switch_core_type=CPUTypes.O3 if args.cpu_type == "O3" else CPUTypes.TIMING,
    isa=ISA.X86,
    num_cores=args.num_cpus,
)

if starting_core_type == CPUTypes.KVM:
    for proc in processor.start:
        proc.core.usePerf = False

board = X86Board(
    clk_freq="2.4GHz",
    processor=processor,
    memory=memory,
    cache_hierarchy=cache_hierarchy,
    cxl_memory=cxl_dram,
    is_asic=(args.is_asic == "True"),
)

_script_dir = os.path.dirname(os.path.abspath(__file__))
_simcxl_root = os.path.abspath(os.path.join(_script_dir, "..", "..", ".."))
_kernel_path = os.path.join(_simcxl_root, "vmlinux")
_disk_path = os.path.join(_simcxl_root, "parsec.img")
_script_path = os.path.join(_simcxl_root, "script.sh")

if args.manual:
    command = "sleep 999999"
elif os.path.exists(_script_path):
    command = "m5 exit;" + open(_script_path).read()
else:
    command = (
        "m5 exit;"
        + "numactl -H;"
        + "m5 resetstats;"
        + "numactl -N 0 -m 1 /home/test_code/simple_test;"
    )

board.set_kernel_disk_workload(
    kernel=KernelResource(local_path=_kernel_path),
    disk_image=DiskImageResource(local_path=_disk_path),
    readfile_contents=command,
    kernel_args=board.get_default_kernel_args(),
)


def switch_and_force_timing() -> bool:
    processor.switch()
    board.set_mem_mode(MemMode.TIMING)
    print(f"after switch mem_mode={board.mem_mode}")
    return False

simulator = Simulator(
    board=board,
    on_exit_event={ExitEvent.EXIT: [switch_and_force_timing]},
)
m5.stats.reset()
simulator.run()
