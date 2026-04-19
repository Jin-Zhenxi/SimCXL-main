# Copyright (c) 2021 The Regents of the University of California
# All rights reserved.

import argparse
import os

from matrixflow_workload import build_matrixflow_command

import m5
import m5.internal.params as m5_internal_params
from m5.objects import SrcClockDomain

from gem5.components.boards.mem_mode import MemMode
from gem5.components.boards.x86_board import X86Board
from gem5.components.cachehierarchies.classic.private_l1_private_l2_shared_l3_cache_hierarchy import (
    PrivateL1PrivateL2SharedL3CacheHierarchy,
)
from gem5.components.memory.single_channel import (
    DIMM_DDR5_4400,
    DIMM_DDR5_8400,
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


def device_link_width_bytes(device_link_gbs: int) -> int:
    if device_link_gbs % 2 != 0:
        raise ValueError("device_link_gbs 必须与 2GHz fabric 对齐，当前仅支持偶数 GB/s。")
    return device_link_gbs // 2


parser = argparse.ArgumentParser(
    description="CXL Type-3 classic config for MatrixFlow pure_gemm / vit_proxy."
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
    "--boot-cpu-type",
    "--boot_cpu_type",
    dest="boot_cpu_type",
    type=str,
    choices=["KVM", "ATOMIC"],
    default="KVM",
)
parser.add_argument("--manual", action="store_true")
parser.add_argument("--no-network", action="store_true")
parser.add_argument(
    "--matrixflow-workload",
    "--matrixflow_workload",
    dest="matrixflow_workload",
    choices=["script", "vit_proxy", "pure_gemm"],
    default="script",
)
parser.add_argument(
    "--matrixflow-size",
    "--matrixflow_size",
    dest="matrixflow_size",
    type=int,
    default=257,
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
    "--device-link-gbs",
    "--device_link_gbs",
    dest="device_link_gbs",
    type=int,
    choices=[16, 32, 64, 128, 256],
    default=32,
)
parser.add_argument(
    "--pure-gemm-pack-tail-align16",
    "--pure_gemm_pack_tail_align16",
    dest="pure_gemm_pack_tail_align16",
    action="store_true",
)
parser.add_argument(
    "--pure-gemm-peeled-rect-v1",
    "--pure_gemm_peeled_rect_v1",
    dest="pure_gemm_peeled_rect_v1",
    action="store_true",
)
parser.add_argument(
    "--pure-gemm-peeled-rect-v2-right-edge-rectified",
    "--pure_gemm_peeled_rect_v2_right_edge_rectified",
    dest="pure_gemm_peeled_rect_v2_right_edge_rectified",
    action="store_true",
)
parser.add_argument(
    "--pure-gemm-peeled-rect-batched-single-doorbell-first-cut",
    "--pure_gemm_peeled_rect_batched_single_doorbell_first_cut",
    dest="pure_gemm_peeled_rect_batched_single_doorbell_first_cut",
    action="store_true",
)
parser.add_argument(
    "--irregular-gemm-b-tail-scratchpad-output-hold-first-cut",
    "--irregular_gemm_b_tail_scratchpad_output_hold_first_cut",
    dest="irregular_gemm_b_tail_scratchpad_output_hold_first_cut",
    action="store_true",
)
parser.add_argument(
    "--irregular-gemm-fused-edges-completion-optimized-first-cut",
    "--irregular_gemm_fused_edges_completion_optimized_first_cut",
    dest="irregular_gemm_fused_edges_completion_optimized_first_cut",
    action="store_true",
)
parser.add_argument(
    "--irregular-gemm-fused-right-edge-clean-timing-first-cut",
    "--irregular_gemm_fused_right_edge_clean_timing_first_cut",
    dest="irregular_gemm_fused_right_edge_clean_timing_first_cut",
    action="store_true",
)
parser.add_argument(
    "--irregular-gemm-residual-device-overhead-autopsy-first-cut",
    "--irregular_gemm_residual_device_overhead_autopsy_first_cut",
    dest="irregular_gemm_residual_device_overhead_autopsy_first_cut",
    action="store_true",
)
parser.add_argument(
    "--irregular-gemm-no-wait-fused-right-first-cut",
    "--irregular_gemm_no_wait_fused_right_first_cut",
    dest="irregular_gemm_no_wait_fused_right_first_cut",
    action="store_true",
)
parser.add_argument(
    "--irregular-gemm-no-wait-fused-bottom-first-cut",
    "--irregular_gemm_no_wait_fused_bottom_first_cut",
    dest="irregular_gemm_no_wait_fused_bottom_first_cut",
    action="store_true",
)
parser.add_argument(
    "--irregular-gemm-single-fused-descriptor-corner-collapse-first-cut",
    "--irregular_gemm_single_fused_descriptor_corner_collapse_first_cut",
    dest="irregular_gemm_single_fused_descriptor_corner_collapse_first_cut",
    action="store_true",
)
parser.add_argument(
    "--irregular-gemm-final-completion-chain-autopsy-first-cut",
    "--irregular_gemm_final_completion_chain_autopsy_first_cut",
    dest="irregular_gemm_final_completion_chain_autopsy_first_cut",
    action="store_true",
)
parser.add_argument(
    "--irregular-gemm-boundary-only-hold-early-body-writeback-first-cut",
    "--irregular_gemm_boundary_only_hold_early_body_writeback_first_cut",
    dest="irregular_gemm_boundary_only_hold_early_body_writeback_first_cut",
    action="store_true",
)
parser.add_argument(
    "--irregular-gemm-static-output-tile-classifier-boundary-hold-first-cut",
    "--irregular_gemm_static_output_tile_classifier_boundary_hold_first_cut",
    dest="irregular_gemm_static_output_tile_classifier_boundary_hold_first_cut",
    action="store_true",
)
parser.add_argument(
    "--irregular-gemm-boundary-writeback-coalescing-first-cut",
    "--irregular_gemm_boundary_writeback_coalescing_first_cut",
    dest="irregular_gemm_boundary_writeback_coalescing_first_cut",
    action="store_true",
)
parser.add_argument(
    "--irregular-gemm-streaming-body-writeback-first-cut",
    "--irregular_gemm_streaming_body_writeback_first_cut",
    dest="irregular_gemm_streaming_body_writeback_first_cut",
    action="store_true",
)
parser.add_argument(
    "--irregular-gemm-small-full-residency-pad256-first-cut",
    "--irregular_gemm_small_full_residency_pad256_first_cut",
    dest="irregular_gemm_small_full_residency_pad256_first_cut",
    action="store_true",
)
parser.add_argument(
    "--irregular-gemm-logical-zero-fill-clipped-execution-first-cut",
    "--irregular_gemm_logical_zero_fill_clipped_execution_first_cut",
    dest="irregular_gemm_logical_zero_fill_clipped_execution_first_cut",
    action="store_true",
)
parser.add_argument(
    "--interior-writeback-stripe-rows",
    "--interior_writeback_stripe_rows",
    dest="interior_writeback_stripe_rows",
    type=int,
    default=0,
)
parser.add_argument(
    "--max-outstanding-stripes",
    "--max_outstanding_stripes",
    dest="max_outstanding_stripes",
    type=int,
    default=0,
)
parser.add_argument(
    "--boundary-right-writeback-bytes",
    "--boundary_right_writeback_bytes",
    dest="boundary_right_writeback_bytes",
    type=int,
    default=4,
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

starting_core_type = (
    CPUTypes.KVM if args.boot_cpu_type == "KVM" else CPUTypes.ATOMIC
)
if starting_core_type == CPUTypes.KVM:
    kvm_device_usable = os.path.exists("/dev/kvm") and os.access(
        "/dev/kvm", os.R_OK | os.W_OK
    )
    kvm_params_usable = hasattr(m5_internal_params, "X86KvmCPUParams")
    if not (kvm_device_usable and kvm_params_usable):
        print(
            "WARN: /dev/kvm or X86KvmCPUParams unavailable; falling back to ATOMIC boot CPU."
        )
        starting_core_type = CPUTypes.ATOMIC

processor = SimpleSwitchableProcessor(
    starting_core_type=starting_core_type,
    switch_core_type=CPUTypes.O3 if args.cpu_type == "O3" else CPUTypes.TIMING,
    isa=ISA.X86,
    num_cores=args.num_cpus,
)

if starting_core_type == CPUTypes.KVM:
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

cxl_dev = board.pc.south_bridge.cxl_device
cxl_dev.matrix_engine.body_interior_writeback_stripe_rows = (
    args.interior_writeback_stripe_rows
)
cxl_dev.matrix_engine.body_interior_writeback_max_outstanding_stripes = (
    args.max_outstanding_stripes
)
cxl_dev.matrix_engine.boundary_right_writeback_bytes = (
    args.boundary_right_writeback_bytes
)
board.matrixflow_cxl_mem_clk_domain = SrcClockDomain(
    clock="2GHz",
    voltage_domain=board.clk_domain.voltage_domain,
)
cxl_dev.cxl_mem_bus.width = device_link_width_bytes(args.device_link_gbs)
cxl_dev.cxl_mem_bus.clk_domain = board.matrixflow_cxl_mem_clk_domain
print(
    "[CXL MatrixFlow Config] "
    f"device_link_gbs={args.device_link_gbs}, "
    "fabric_clock=2GHz, "
    f"cxl_mem_bus_width={device_link_width_bytes(args.device_link_gbs)}B, "
    f"workload={args.matrixflow_workload}"
)

script_dir = os.path.dirname(os.path.abspath(__file__))
simcxl_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
kernel_path = os.path.join(simcxl_root, "vmlinux")
disk_path = os.path.join(simcxl_root, "parsec.img")
script_path = os.path.join(simcxl_root, "script.sh")
workload_is_pure_gemm = args.matrixflow_workload == "pure_gemm"
logical_zero_fill_mode = (
    args.irregular_gemm_logical_zero_fill_clipped_execution_first_cut
)
small_full_residency_mode = (
    args.irregular_gemm_small_full_residency_pad256_first_cut
)
boundary_coalescing_mode = (
    args.irregular_gemm_boundary_writeback_coalescing_first_cut
)
static_classifier_mode = (
    args.irregular_gemm_static_output_tile_classifier_boundary_hold_first_cut
)
boundary_hold_body_wb_mode = (
    args.irregular_gemm_boundary_only_hold_early_body_writeback_first_cut
)
completion_autopsy_mode = (
    args.irregular_gemm_final_completion_chain_autopsy_first_cut
)
single_fused_corner_mode = (
    args.irregular_gemm_single_fused_descriptor_corner_collapse_first_cut
)
b_tail_hold_mode = args.irregular_gemm_b_tail_scratchpad_output_hold_first_cut
pack_tail_align16_mode = args.pure_gemm_pack_tail_align16
trigger_mode_args = []
if logical_zero_fill_mode:
    trigger_mode_args = [
        "irregular_gemm_logical_zero_fill_clipped_execution_first_cut"
    ]
elif small_full_residency_mode:
    trigger_mode_args = [
        "irregular_gemm_small_full_residency_pad256_first_cut"
    ]
elif (
    args.irregular_gemm_streaming_body_writeback_first_cut
    and workload_is_pure_gemm
):
    trigger_mode_args = ["irregular_gemm_streaming_body_writeback_first_cut"]
elif boundary_coalescing_mode and workload_is_pure_gemm:
    trigger_mode_args = [
        "irregular_gemm_boundary_writeback_coalescing_first_cut"
    ]
elif static_classifier_mode:
    trigger_mode_args = [
        "irregular_gemm_static_output_tile_classifier_boundary_hold_first_cut"
    ]
elif boundary_hold_body_wb_mode and workload_is_pure_gemm:
    trigger_mode_args = [
        "irregular_gemm_boundary_only_hold_early_body_writeback_first_cut"
    ]
elif completion_autopsy_mode and workload_is_pure_gemm:
    trigger_mode_args = [
        "irregular_gemm_final_completion_chain_autopsy_first_cut"
    ]
elif single_fused_corner_mode and workload_is_pure_gemm:
    trigger_mode_args = [
        "irregular_gemm_single_fused_descriptor_corner_collapse_first_cut"
    ]
elif (
    args.irregular_gemm_no_wait_fused_bottom_first_cut
    and workload_is_pure_gemm
):
    trigger_mode_args = ["irregular_gemm_no_wait_fused_bottom_first_cut"]
elif (
    args.irregular_gemm_no_wait_fused_right_first_cut and workload_is_pure_gemm
):
    trigger_mode_args = ["irregular_gemm_no_wait_fused_right_first_cut"]
elif (
    args.irregular_gemm_residual_device_overhead_autopsy_first_cut
    and workload_is_pure_gemm
):
    trigger_mode_args = [
        "irregular_gemm_residual_device_overhead_autopsy_first_cut"
    ]
elif (
    args.irregular_gemm_fused_right_edge_clean_timing_first_cut
    and not args.irregular_gemm_residual_device_overhead_autopsy_first_cut
    and workload_is_pure_gemm
):
    trigger_mode_args = [
        "irregular_gemm_fused_right_edge_clean_timing_first_cut"
    ]
elif (
    args.irregular_gemm_fused_edges_completion_optimized_first_cut
    and workload_is_pure_gemm
):
    trigger_mode_args = [
        "irregular_gemm_fused_edges_completion_optimized_first_cut"
    ]
elif b_tail_hold_mode and workload_is_pure_gemm:
    trigger_mode_args = [
        "irregular_gemm_b_tail_scratchpad_output_hold_first_cut"
    ]
elif (
    args.pure_gemm_peeled_rect_batched_single_doorbell_first_cut
    and workload_is_pure_gemm
):
    trigger_mode_args = ["peeled_rect_batched_single_doorbell_first_cut"]
elif (
    args.pure_gemm_peeled_rect_v2_right_edge_rectified
    and workload_is_pure_gemm
):
    trigger_mode_args = ["peeled_rect_v2_right_edge_rectified"]
elif args.pure_gemm_peeled_rect_v1 and workload_is_pure_gemm:
    trigger_mode_args = ["peeled_rect_v1"]
elif pack_tail_align16_mode and workload_is_pure_gemm:
    trigger_mode_args = ["pack_tail_align16"]

command = build_matrixflow_command(
    manual=args.manual,
    script_path=script_path,
    workload=args.matrixflow_workload,
    matrix_size=args.matrixflow_size,
    trigger_binary=args.matrixflow_binary,
    allow_guest_fallback=args.allow_local_trigger,
    trigger_extra_args=[
        str(args.device_link_gbs),
        *trigger_mode_args,
    ],
)

board.set_kernel_disk_workload(
    kernel=KernelResource(local_path=kernel_path),
    disk_image=DiskImageResource(local_path=disk_path),
    readfile_contents=command,
    kernel_args=board.get_default_kernel_args() + ["memmap=8G$16G"],
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
