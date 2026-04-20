# Copyright (c) 2021 The Regents of the University of California
# All rights reserved.

import argparse
import os

from matrixflow_workload import build_matrixflow_command

import m5

from gem5.components.boards.mem_mode import MemMode
from gem5.components.boards.x86_board import X86Board
from gem5.components.cachehierarchies.classic.private_l1_private_l2_shared_l3_cache_hierarchy import (
    PrivateL1PrivateL2SharedL3CacheHierarchy,
)
from gem5.components.memory.dram_interfaces.hbm import HBM_1000_4H_1x128
from gem5.components.memory.memory import ChanneledMemory
from gem5.components.memory.single_channel import (
    DIMM_DDR5_4400,
    SingleChannelDDR4_3200,
)
from gem5.components.processors.cpu_types import CPUTypes
from gem5.components.processors.simple_switchable_processor import (
    SimpleSwitchableProcessor,
)
from gem5.isas import ISA
from gem5.resources.resource import (
    DiskImageResource,
    KernelResource,
)
from gem5.simulate.exit_event import ExitEvent
from gem5.simulate.simulator import Simulator
from gem5.utils.requires import requires

requires(isa_required=ISA.X86)


class IdealCXLHBM2(HBM_1000_4H_1x128):
    device_size = "1GiB"
    tCK = "1ns"
    tBURST = "2ns"
    tRP = "14ns"
    tRCD = "14ns"
    tCL = "14ns"
    tRAS = "33ns"
    read_buffer_size = 128
    write_buffer_size = 128


def device_link_width_bytes(device_link_gbs: int) -> int:
    if device_link_gbs % 2 != 0:
        raise ValueError("device_link_gbs 必须与 2 GHz fabric 对齐，当前只支持偶数 GB/s。")
    return device_link_gbs // 2


parser = argparse.ArgumentParser(
    description="PCIe+HBM pure GEMM with configurable host-device bandwidth."
)
parser.add_argument(
    "--is_asic",
    action="store",
    type=str,
    nargs="?",
    choices=["True", "False"],
    default="True",
)
parser.add_argument("--num_cpus", type=int, default=1)
parser.add_argument(
    "--cpu_type",
    type=str,
    choices=["TIMING", "O3"],
    default="TIMING",
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
parser.add_argument(
    "--matrixflow-size",
    "--matrixflow_size",
    dest="matrixflow_size",
    type=int,
    default=2048,
)
parser.add_argument(
    "--matrixflow-binary",
    "--matrixflow_binary",
    dest="matrixflow_binary",
    default="/root/trigger_gemm",
)
parser.add_argument(
    "--allow-local-trigger",
    "--allow_local_trigger",
    dest="allow_local_trigger",
    action="store_true",
)
parser.add_argument(
    "--matrixflow-workload",
    "--matrixflow_workload",
    dest="matrixflow_workload",
    choices=["pure_gemm", "host_link_gemm", "vit_proxy"],
    default="pure_gemm",
)
parser.add_argument(
    "--device-link-gbs",
    "--device_link_gbs",
    "--host-link-gbs",
    "--host_link_gbs",
    dest="device_link_gbs",
    type=int,
    choices=[16, 32, 64, 128, 256],
    default=64,
)
parser.add_argument(
    "--phase2-mode",
    "--phase2_mode",
    dest="phase2_mode",
    choices=[
        "devm_copy",
        "correct_devm_copy_good_path",
        "devmem_5x_non_gemm_remote_access",
    ],
    default="devm_copy",
)
args = parser.parse_args()

if args.no_network:
    os.environ["GEM5_USE_ETHERtap_STUB"] = "1"

os.environ.pop("MATRIXFLOW_HOST_DMA_BRIDGE", None)

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

memory = DIMM_DDR5_4400(size="3GB")
# Keep the HBM interleave granularity aligned with MatrixFlow's 256B DMA
# chunk size so a single DMA packet maps cleanly onto one downstream range.
cxl_dram = ChanneledMemory(IdealCXLHBM2, 8, 256, size="8GB")
if args.is_asic == "False":
    cxl_dram = SingleChannelDDR4_3200(size="8GB")

starting_core_type = CPUTypes.KVM
if not (
    os.path.exists("/dev/kvm") and os.access("/dev/kvm", os.R_OK | os.W_OK)
):
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
cxl_dev = board.pc.south_bridge.cxl_device
cxl_dev.cxl_mem_bus.width = device_link_width_bytes(args.device_link_gbs)
fabric_clk_domain = board.__dict__.get(
    "pcie_fabric_clk_domain", board.clk_domain
)
cxl_dev.cxl_mem_bus.clk_domain = fabric_clk_domain
print(
    "[HBM+PCIe Config] "
    f"device_link_gbs={args.device_link_gbs}, "
    "fabric_clock=2GHz, "
    f"cxl_mem_bus_width={device_link_width_bytes(args.device_link_gbs)}B, "
    f"workload={args.matrixflow_workload}, "
    f"phase2_mode={args.phase2_mode}"
)

script_dir = os.path.dirname(os.path.abspath(__file__))
simcxl_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
kernel_path = os.path.join(simcxl_root, "vmlinux")
disk_path = os.path.join(simcxl_root, "parsec.img")
script_path = os.path.join(simcxl_root, "script.sh")

trigger_extra_args = [str(args.device_link_gbs), args.phase2_mode]

command = build_matrixflow_command(
    manual=args.manual,
    script_path=script_path,
    workload=args.matrixflow_workload,
    matrix_size=args.matrixflow_size,
    trigger_binary=args.matrixflow_binary,
    allow_guest_fallback=args.allow_local_trigger,
    trigger_extra_args=trigger_extra_args,
)

board.set_kernel_disk_workload(
    kernel=KernelResource(local_path=kernel_path),
    disk_image=DiskImageResource(local_path=disk_path),
    readfile_contents=command,
    kernel_args=board.get_default_kernel_args(),
)


def switch_and_force_timing() -> bool:
    processor.switch()
    board.set_mem_mode(MemMode.TIMING)
    print(f"after switch mem_mode={board.mem_mode}")
    return False


def exit_on_benchmark_completion() -> bool:
    print("benchmark completed, exiting simulator")
    return True


simulator = Simulator(
    board=board,
    on_exit_event={
        ExitEvent.EXIT: [switch_and_force_timing, exit_on_benchmark_completion]
    },
)
m5.stats.reset()
simulator.run()
