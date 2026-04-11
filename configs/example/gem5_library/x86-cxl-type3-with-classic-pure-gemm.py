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

import argparse
import os

import m5
from gem5.utils.requires import requires
from matrixflow_workload import build_matrixflow_command
from gem5.components.boards.x86_board import X86Board
from gem5.components.memory.single_channel import (
    DIMM_DDR5_4400,
    DIMM_DDR5_8400,
    SingleChannelDDR4_3200,
)
from gem5.components.processors.simple_switchable_processor import (
    SimpleSwitchableProcessor,
)
from gem5.components.processors.cpu_types import CPUTypes
from gem5.components.cachehierarchies.classic.private_l1_private_l2_shared_l3_cache_hierarchy import (
    PrivateL1PrivateL2SharedL3CacheHierarchy,
)
from gem5.isas import ISA
from gem5.simulate.simulator import Simulator
from gem5.simulate.exit_event import ExitEvent
from gem5.resources.resource import DiskImageResource, KernelResource

requires(isa_required=ISA.X86)

parser = argparse.ArgumentParser(description="Pure GEMM on CXL Type-3 classic.")
parser.add_argument(
    "--is_asic",
    action="store",
    type=str,
    nargs="?",
    choices=["True", "False"],
    default="True",
    help="Choose to simulate CXL ASIC Device or FPGA Device.",
)
parser.add_argument("--num_cpus", type=int, default=1, help="Number of CPUs")
parser.add_argument(
    "--cpu_type",
    type=str,
    choices=["TIMING", "O3"],
    default="TIMING",
    help="CPU type",
)
parser.add_argument(
    "--boot-cpu-type",
    "--boot_cpu_type",
    dest="boot_cpu_type",
    type=str,
    choices=["KVM", "ATOMIC"],
    default="KVM",
    help="Boot CPU type before switching to the detailed core.",
)
parser.add_argument(
    "--manual",
    action="store_true",
    help="Manual debug mode: no m5 exit, keep sim running for telnet",
)
parser.add_argument(
    "--no-network",
    action="store_true",
    help="Use EtherTapStub instead of EtherTap (WSL2/无 root 时避免 tap 设备 panic)",
)
parser.add_argument(
    "--matrixflow-size",
    "--matrixflow_size",
    dest="matrixflow_size",
    type=int,
    default=2048,
    help="Matrix size passed to trigger_gemm pure_gemm.",
)
parser.add_argument(
    "--matrixflow-binary",
    "--matrixflow_binary",
    dest="matrixflow_binary",
    default="/root/trigger_gemm",
    help="Preferred guest path to trigger_gemm.",
)
parser.add_argument(
    "--allow-local-trigger",
    "--allow_local_trigger",
    dest="allow_local_trigger",
    action="store_true",
    help="Allow fallback to an already injected guest trigger_gemm binary.",
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
    l2_mshrs=64,
)

memory = DIMM_DDR5_4400(size="3GB")
cxl_dram = DIMM_DDR5_8400(size="8GB")
if args.is_asic == "False":
    cxl_dram = SingleChannelDDR4_3200(size="8GB")

processor = SimpleSwitchableProcessor(
    starting_core_type=(
        CPUTypes.KVM if args.boot_cpu_type == "KVM" else CPUTypes.ATOMIC
    ),
    switch_core_type=CPUTypes.O3 if args.cpu_type == "O3" else CPUTypes.TIMING,
    isa=ISA.X86,
    num_cores=args.num_cpus,
)

for proc in processor.start:
    if hasattr(proc.core, "usePerf"):
        proc.core.usePerf = False

board = X86Board(
    clk_freq="2.4GHz",
    processor=processor,
    memory=memory,
    cache_hierarchy=cache_hierarchy,
    cxl_memory=cxl_dram,
    is_asic=(args.is_asic == "True"),
)

script_dir = os.path.dirname(os.path.abspath(__file__))
simcxl_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
kernel_path = os.path.join(simcxl_root, "vmlinux")
disk_path = os.path.join(simcxl_root, "parsec.img")
script_path = os.path.join(simcxl_root, "script.sh")

command = build_matrixflow_command(
    manual=args.manual,
    script_path=script_path,
    workload="pure_gemm",
    matrix_size=args.matrixflow_size,
    trigger_binary=args.matrixflow_binary,
    allow_guest_fallback=args.allow_local_trigger,
)

board.set_kernel_disk_workload(
    kernel=KernelResource(local_path=kernel_path),
    disk_image=DiskImageResource(local_path=disk_path),
    readfile_contents=command,
    kernel_args=board.get_default_kernel_args() + ["memmap=8G$16G"],
)

simulator = Simulator(
    board=board,
    on_exit_event={ExitEvent.EXIT: [processor.switch]},
)
m5.stats.reset()
simulator.run()
