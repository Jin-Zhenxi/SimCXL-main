#!/usr/bin/env python3
"""Run ViT-like proxy presets on the current HBM+PCIe baseline."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
from pathlib import Path

from run_sweep import (
    PRESETS,
    ensure_libm5,
    extract_gemm_active_time,
    extract_metrics,
    extract_phase_timings,
    modify_c_file,
)

GEM5_ROOT = Path("/home/jzx8091/SimCXL-main")
OUTPUT_DIR = GEM5_ROOT / "my_outputData"
TRIGGER_SRC = GEM5_ROOT / "trigger_gemm.c"
TRIGGER_BIN = GEM5_ROOT / "trigger_gemm"
GEM5_BIN = GEM5_ROOT / "build/X86/gem5.opt"
GEM5_SCRIPT = "configs/example/gem5_library/x86-cxl-pcie-hbm-pure-gemm.py"
COMPILE_CMD = (
    f"gcc -O3 -static -mno-80387 -o {TRIGGER_BIN} {TRIGGER_SRC} "
    f"-I{GEM5_ROOT}/include -L{GEM5_ROOT}/util/m5/build/x86/out -lm -lm5"
)
CLOCK_FREQ_GHZ = 2.4
SUMMARY_CSV = OUTPUT_DIR / "hbm_vit_proxy_data.csv"
SUMMARY_TXT = OUTPUT_DIR / "hbm_vit_proxy_summary_raw.txt"


def run_cmd(cmd: str) -> None:
    print(f"[*] {cmd}")
    subprocess.run(cmd, shell=True, cwd=GEM5_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ViT Base/Large/Huge proxy presets on HBM+PCIe."
    )
    parser.add_argument(
        "--device-link-gbs",
        "--device_link_gbs",
        dest="device_link_gbs",
        type=int,
        choices=[16, 32, 64, 128, 256],
        default=64,
    )
    parser.add_argument(
        "--cpu-type",
        "--cpu_type",
        dest="cpu_type",
        choices=["TIMING", "O3"],
        default="TIMING",
    )
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--skip-inject", action="store_true")
    parser.add_argument("--keep-existing", action="store_true")
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
    parser.add_argument(
        "--only-preset",
        choices=[preset["name"] for preset in PRESETS],
        help="Run only one ViT preset for quick validation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_libm5()
    OUTPUT_DIR.mkdir(exist_ok=True)
    if args.phase2_mode == "devm_copy":
        summary_csv = SUMMARY_CSV
        summary_txt = SUMMARY_TXT
    else:
        summary_csv = OUTPUT_DIR / f"hbm_vit_proxy_{args.phase2_mode}_data.csv"
        summary_txt = OUTPUT_DIR / f"hbm_vit_proxy_{args.phase2_mode}_raw.txt"

    original_trigger = TRIGGER_SRC.read_text(encoding="utf-8")
    results: list[dict[str, object]] = []

    try:
        print("🚀 启动 HBM ViT proxy sweep")
        selected_presets = [
            preset for preset in PRESETS
            if args.only_preset is None or preset["name"] == args.only_preset
        ]
        print(
            "   预设: "
            + ", ".join(
                f"{preset['name']}[S={preset['seq_len']},H={preset['hidden_dim']},"
                f"M={preset['mlp_dim']},heads={preset['num_heads']}]"
                for preset in selected_presets
            )
        )
        print(f"   device_link_gbs={args.device_link_gbs}")
        print(f"   phase2_mode={args.phase2_mode}")
        print("=" * 50)

        for preset in selected_presets:
            preset_tag = preset["name"].replace(" ", "_").replace("/", "_")
            seq_len = preset["seq_len"]
            run_tag = (
                f"{preset_tag}__pcie_hbm_{args.device_link_gbs}g__"
                f"{args.phase2_mode}"
            )
            m5out_dir = OUTPUT_DIR / f"m5out_{run_tag}"
            log_file = OUTPUT_DIR / f"terminal_log_{run_tag}.txt"
            serial_log = m5out_dir / "board.pc.com_1.device"
            stats_file = m5out_dir / "stats.txt"

            print(f"\n>>> 正在测试预设: {preset['name']} <<<")
            modify_c_file(preset)

            if not args.skip_compile:
                print("[*] 正在编译 trigger_gemm...")
                run_cmd(COMPILE_CMD)

            if not args.skip_inject:
                inject_cmd = (
                    f"sudo ./inject_trigger_gemm.sh --no-compile {seq_len}"
                )
                print("[*] 正在注入镜像...")
                run_cmd(inject_cmd)

            if not args.keep_existing:
                if m5out_dir.exists():
                    shutil.rmtree(m5out_dir)
                if log_file.exists():
                    log_file.unlink()

            cmd = (
                f"{GEM5_BIN} -p {GEM5_ROOT / 'src/python'} "
                f"--debug-flags=MatrixFlow "
                f"-d {m5out_dir} "
                f"{GEM5_SCRIPT} "
                "--is_asic True "
                f"--cpu_type {args.cpu_type} "
                "--no-network "
                f"--matrixflow_size {seq_len} "
                "--matrixflow_workload vit_proxy "
                f"--device-link-gbs {args.device_link_gbs} "
                f"--phase2-mode {args.phase2_mode} "
                "--allow-local-trigger "
                f"> {log_file} 2>&1"
            )
            print(f"[*] 正在运行 gem5 仿真，日志存入: {log_file}")
            rc = subprocess.run(cmd, shell=True, cwd=GEM5_ROOT)
            if rc.returncode != 0:
                raise SystemExit(
                    f"gem5 仿真失败，返回码={rc.returncode}。请检查日志: {log_file}"
                )

            metrics = extract_metrics(str(stats_file), str(log_file))
            phase_timings = extract_phase_timings(str(serial_log))
            gemm_active_s = extract_gemm_active_time(str(log_file))
            flops = 4.0 * (seq_len**3)
            roi_latency_s = metrics["simSeconds"]
            total_bytes = metrics["dmaRead"] + metrics["dmaWrite"]

            peak_gflops = (
                (flops / 1e9)
                / (metrics["computeCycles"] / (CLOCK_FREQ_GHZ * 1e9))
                if metrics["computeCycles"] > 0
                else 0.0
            )
            gemm_effective_peak_gflops = (
                (flops / 1e9) / gemm_active_s if gemm_active_s > 0 else 0.0
            )
            effective_gflops = (
                (flops / 1e9) / roi_latency_s if roi_latency_s > 0 else 0.0
            )
            oi = flops / total_bytes if total_bytes > 0 else 0.0

            workload_ok = (
                metrics["computeCycles"] > 0
                and metrics["dmaRead"] > 0
                and phase_timings["end_to_end_ms"] > 0.0
            )
            result_row = {
                "Preset": preset["name"],
                "Device Link (GB/s)": args.device_link_gbs,
                "Workload OK": workload_ok,
                "SeqLen": preset["seq_len"],
                "HiddenDim": preset["hidden_dim"],
                "MLPDim": preset["mlp_dim"],
                "NumHeads": preset["num_heads"],
                "GEMM Proxy Size": seq_len,
                "Phase2 Mode": phase_timings["phase2_mode"],
                "system_name": phase_timings["system_name"],
                "GEMM_location": phase_timings["GEMM_location"],
                "NonGEMM_location": phase_timings["NonGEMM_location"],
                "data_home": phase_timings["data_home"],
                "data_home_before_nongemm": phase_timings[
                    "data_home_before_nongemm"
                ],
                "FLOPs": int(flops),
                "End-to-End ROI Latency (s)": round(roi_latency_s, 6),
                "end_to_end_ms": round(phase_timings["end_to_end_ms"], 6),
                "phase1_ms": round(phase_timings["phase1_ms"], 6),
                "phase2_d2h_ms": round(phase_timings["phase2_d2h_ms"], 6),
                "phase2_softmax_ms": round(
                    phase_timings["phase2_softmax_ms"], 6
                ),
                "phase2_layernorm_ms": round(
                    phase_timings["phase2_layernorm_ms"], 6
                ),
                "phase2_gelu_ms": round(phase_timings["phase2_gelu_ms"], 6),
                "phase2_residual_ms": round(
                    phase_timings["phase2_residual_ms"], 6
                ),
                "phase2_h2d_ms": round(phase_timings["phase2_h2d_ms"], 6),
                "phase2_copy_ms": round(phase_timings["phase2_copy_ms"], 6),
                "phase2_non_gemm_ms": round(
                    phase_timings["phase2_non_gemm_ms"], 6
                ),
                "phase2_total_ms": round(phase_timings["phase2_total_ms"], 6),
                "phase3_ms": round(phase_timings["phase3_ms"], 6),
                "phase2_share_pct": round(
                    100.0
                    * phase_timings["phase2_total_ms"]
                    / phase_timings["end_to_end_ms"]
                    if phase_timings["end_to_end_ms"] > 0
                    else 0.0,
                    6,
                ),
                "host_mediated_copy_bytes": phase_timings[
                    "host_mediated_copy_bytes"
                ],
                "phase2_cpu_reads_remote_mem_bytes": phase_timings[
                    "phase2_cpu_reads_remote_mem_bytes"
                ],
                "phase2_cpu_writes_remote_mem_bytes": phase_timings[
                    "phase2_cpu_writes_remote_mem_bytes"
                ],
                "remote_memory_cacheable": phase_timings[
                    "remote_memory_cacheable"
                ],
                "remote_memory_coherent": phase_timings[
                    "remote_memory_coherent"
                ],
                "explicit_host_mediated_copy_used": phase_timings[
                    "explicit_host_mediated_copy_used"
                ],
                "phase2_remote_first_pass_bytes": phase_timings[
                    "phase2_remote_first_pass_bytes"
                ],
                "phase2_remote_revisit_bytes": phase_timings[
                    "phase2_remote_revisit_bytes"
                ],
                "phase2_remote_cache_hit_like_count": phase_timings[
                    "phase2_remote_cache_hit_like_count"
                ],
                "phase2_remote_cache_miss_like_count": phase_timings[
                    "phase2_remote_cache_miss_like_count"
                ],
                "phase2_read_bytes": phase_timings["phase2_read_bytes"],
                "phase2_write_bytes": phase_timings["phase2_write_bytes"],
                "phase2_read_accesses": phase_timings["phase2_read_accesses"],
                "phase2_write_accesses": phase_timings[
                    "phase2_write_accesses"
                ],
                "phase2_total_bytes": (
                    phase_timings["phase2_read_bytes"]
                    + phase_timings["phase2_write_bytes"]
                ),
                "phase2_avg_read_size_B": round(
                    phase_timings["phase2_read_bytes"]
                    / phase_timings["phase2_read_accesses"]
                    if phase_timings["phase2_read_accesses"] > 0
                    else 0.0,
                    6,
                ),
                "phase2_avg_write_size_B": round(
                    phase_timings["phase2_write_bytes"]
                    / phase_timings["phase2_write_accesses"]
                    if phase_timings["phase2_write_accesses"] > 0
                    else 0.0,
                    6,
                ),
                "totalDmaRead": metrics["dmaRead"],
                "totalDmaWrite": metrics["dmaWrite"],
                "totalComputeCycles": metrics["computeCycles"],
                "Operational Intensity (OI)": round(oi, 4),
                "Peak MAC Throughput (GFLOPS)": round(peak_gflops, 6),
                "Effective GEMM Peak (GFLOPS)": round(
                    gemm_effective_peak_gflops, 6
                ),
                "Effective System Throughput (GFLOPS)": round(
                    effective_gflops, 6
                ),
                "m5out_dir": str(m5out_dir),
                "terminal_log": str(log_file),
                "serial_log": str(serial_log),
            }
            results.append(result_row)

            print(
                f"[{'√' if workload_ok else '!'}] {preset['name']} 完成! "
                f"end_to_end_ms={result_row['end_to_end_ms']}, "
                f"phase2_ms={result_row['phase2_total_ms']}, "
                f"GEMM-Effective={result_row['Effective GEMM Peak (GFLOPS)']} GFLOPS, "
                f"System-Effective={result_row['Effective System Throughput (GFLOPS)']} GFLOPS"
            )

    finally:
        TRIGGER_SRC.write_text(original_trigger, encoding="utf-8")

    if results:
        headers = list(results[0].keys())
        with summary_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=headers)
            writer.writeheader()
            writer.writerows(results)

        with summary_txt.open("w", encoding="utf-8") as fh:
            for row in results:
                fh.write(str(row) + "\n")

        print("\n" + "=" * 50)
        print(f"🎉 HBM ViT proxy 完成！数据已保存至: {summary_csv}")


if __name__ == "__main__":
    main()
