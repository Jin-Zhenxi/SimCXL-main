#!/usr/bin/env python3
"""Run ViT-like proxy presets on the CXL Type-3 DDR5 HDM baseline."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
from pathlib import Path

from run_sweep import (
    PREFETCH_CONFIGS,
    ensure_libm5,
    extract_gemm_active_time,
    extract_metrics,
    extract_phase_timings,
    modify_c_file,
)

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
    {
        "name": "ViT-Huge-like",
        "seq_len": 257,
        "hidden_dim": 1280,
        "mlp_dim": 5120,
        "num_heads": 16,
    },
]

GEM5_ROOT = Path("/home/jzx8091/SimCXL-main")
OUTPUT_DIR = GEM5_ROOT / "my_outputData"
TRIGGER_SRC = GEM5_ROOT / "trigger_gemm.c"
TRIGGER_BIN = GEM5_ROOT / "trigger_gemm"
GEM5_BIN = GEM5_ROOT / "build/X86/gem5.opt"
GEM5_SCRIPT = (
    "configs/example/gem5_library/x86-cxl-type3-with-classic-matrixflow.py"
)
COMPILE_CMD = (
    f"gcc -O3 -static -mno-80387 -o {TRIGGER_BIN} {TRIGGER_SRC} "
    f"-I{GEM5_ROOT}/include -L{GEM5_ROOT}/util/m5/build/x86/out -lm -lm5"
)
CLOCK_FREQ_GHZ = 2.4
SUMMARY_CSV = OUTPUT_DIR / "cxl_vit_proxy_data.csv"
SUMMARY_TXT = OUTPUT_DIR / "cxl_vit_proxy_summary_raw.txt"
DEFAULT_PREFETCH_LABEL = "baseline_serial"


def get_prefetch_cfg(label: str) -> dict[str, object]:
    for cfg in PREFETCH_CONFIGS:
        if cfg["label"] == label:
            return cfg
    raise SystemExit(f"未知 prefetch label: {label}")


def build_matrixflow_env(
    base_env: dict[str, str], cfg: dict[str, object], preset_name: str
) -> dict[str, str]:
    rows_b = cfg.get("rows_b_by_preset", {}).get(preset_name, cfg["rows_b"])
    env = base_env.copy()
    env["MATRIXFLOW_NEXT_PREFETCH_MODE"] = str(cfg["mode"])
    env["MATRIXFLOW_NEXT_PREFETCH_TRIGGER"] = str(cfg["trigger"])
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
    env["MATRIXFLOW_AB_SCHEDULER_MODE"] = str(
        cfg.get("ab_scheduler_mode", "baseline")
    )
    env["MATRIXFLOW_AB_A_MIN_CREDIT_ROWS"] = str(
        cfg.get("ab_a_min_credit_rows", 16)
    )
    env["MATRIXFLOW_AB_B_BIAS"] = str(cfg.get("ab_bias_b", 1))
    env["MATRIXFLOW_AB_W_URGENCY"] = str(cfg.get("ab_weight_urgency", 4))
    env["MATRIXFLOW_AB_W_DEFICIT"] = str(cfg.get("ab_weight_deficit", 3))
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
    return env


def run_cmd(cmd: str) -> None:
    print(f"[*] {cmd}")
    subprocess.run(cmd, shell=True, cwd=GEM5_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ViT Base/Large/Huge proxy presets on CXL Type-3 DDR5 HDM."
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
        "--cpu-type",
        "--cpu_type",
        dest="cpu_type",
        choices=["TIMING", "O3"],
        default="TIMING",
    )
    parser.add_argument(
        "--preset-names",
        dest="preset_names",
        nargs="+",
        default=None,
        help="只运行指定预设名，例如: --preset-names ViT-Large-like",
    )
    parser.add_argument(
        "--prefetch-label",
        dest="prefetch_label",
        default=DEFAULT_PREFETCH_LABEL,
        help="复用 run_sweep.py 里的预取/调度 preset，默认 baseline_serial",
    )
    parser.add_argument(
        "--irregular-gemm-static-output-tile-classifier-boundary-hold-first-cut",
        "--irregular_gemm_static_output_tile_classifier_boundary_hold_first_cut",
        dest="irregular_gemm_static_output_tile_classifier_boundary_hold_first_cut",
        action="store_true",
    )
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--skip-inject", action="store_true")
    parser.add_argument("--keep-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_libm5()
    OUTPUT_DIR.mkdir(exist_ok=True)
    prefetch_cfg = get_prefetch_cfg(args.prefetch_label)

    selected_presets = PRESETS
    if args.preset_names:
        selected = set(args.preset_names)
        selected_presets = [
            preset for preset in PRESETS if preset["name"] in selected
        ]
        missing = sorted(
            selected - {preset["name"] for preset in selected_presets}
        )
        if missing:
            raise SystemExit(f"未知 preset: {', '.join(missing)}")

    original_trigger = TRIGGER_SRC.read_text(encoding="utf-8")
    results: list[dict[str, object]] = []

    try:
        print("🚀 启动 CXL ViT proxy sweep")
        print(
            "   预设: "
            + ", ".join(
                f"{preset['name']}[S={preset['seq_len']},H={preset['hidden_dim']},"
                f"M={preset['mlp_dim']},heads={preset['num_heads']}]"
                for preset in selected_presets
            )
        )
        print(f"   device_link_gbs={args.device_link_gbs}")
        print(f"   prefetch_label={args.prefetch_label}")
        if (
            args.irregular_gemm_static_output_tile_classifier_boundary_hold_first_cut
        ):
            print("   mode=static_classifier")
        print("=" * 50)

        for preset in selected_presets:
            preset_tag = preset["name"].replace(" ", "_").replace("/", "_")
            seq_len = preset["seq_len"]
            run_tag = f"{preset_tag}__{args.prefetch_label}__cxl_{args.device_link_gbs}g"
            if (
                args.irregular_gemm_static_output_tile_classifier_boundary_hold_first_cut
            ):
                run_tag += "__static_classifier"
            m5out_dir = OUTPUT_DIR / f"m5out_{run_tag}"
            log_file = OUTPUT_DIR / f"terminal_log_{run_tag}.txt"
            serial_log = m5out_dir / "board.pc.com_1.device"
            stats_file = m5out_dir / "stats.txt"

            print(f"\n>>> 正在测试预设: {preset['name']} <<<")
            modify_c_file(preset)

            if not args.skip_compile:
                run_cmd(COMPILE_CMD)

            if not args.skip_inject:
                run_cmd(
                    f"sudo ./inject_trigger_gemm.sh --no-compile {seq_len}"
                )

            if not args.keep_existing:
                if m5out_dir.exists():
                    shutil.rmtree(m5out_dir)
                if log_file.exists():
                    log_file.unlink()

            env = build_matrixflow_env(
                os.environ.copy(), prefetch_cfg, preset["name"]
            )
            cmd = (
                f"{GEM5_BIN} -p {GEM5_ROOT / 'src/python'} "
                "--debug-flags=MatrixFlowTiming "
                f"-d {m5out_dir} "
                f"{GEM5_SCRIPT} "
                "--is_asic True "
                f"--cpu_type {args.cpu_type} "
                "--no-network "
                f"--matrixflow_size {seq_len} "
                "--matrixflow_workload vit_proxy "
                f"--device-link-gbs {args.device_link_gbs} "
                f"{'--irregular-gemm-static-output-tile-classifier-boundary-hold-first-cut ' if args.irregular_gemm_static_output_tile_classifier_boundary_hold_first_cut else ''}"
                "--allow-local-trigger "
                f"> {log_file} 2>&1"
            )
            rc = subprocess.run(cmd, shell=True, cwd=GEM5_ROOT, env=env)
            run_status = (
                "ok" if rc.returncode == 0 else f"failed({rc.returncode})"
            )

            metrics = extract_metrics(str(stats_file), str(log_file))
            phase = extract_phase_timings(str(serial_log))
            gemm_active_s = extract_gemm_active_time(str(log_file))
            proxy_gemm_flops = 4.0 * (seq_len**3)
            softmax_flops = 5.0 * seq_len * seq_len
            layernorm_flops = 6.0 * seq_len * preset["hidden_dim"]
            gelu_flops = 8.0 * seq_len * preset["mlp_dim"]
            residual_flops = 1.0 * seq_len * preset["hidden_dim"]
            non_gemm_flops = (
                softmax_flops + layernorm_flops + gelu_flops + residual_flops
            )
            roi_latency_s = metrics["simSeconds"]
            compute_time_total_s = (
                metrics["computeCycles"] / (CLOCK_FREQ_GHZ * 1e9)
                if metrics["computeCycles"] > 0
                else 0.0
            )
            gemm_effective = (
                (proxy_gemm_flops / 1e9) / gemm_active_s
                if gemm_active_s > 0
                else 0.0
            )
            system_effective = (
                (proxy_gemm_flops / 1e9) / roi_latency_s
                if roi_latency_s > 0
                else 0.0
            )
            phase2_s = phase["phase2_total_ms"] / 1.0e3
            phase2_non_gemm_s = phase["phase2_non_gemm_ms"] / 1.0e3
            phase2_proxy_throughput = (
                (proxy_gemm_flops / 1e9) / phase2_s if phase2_s > 0 else 0.0
            )
            non_gemm_effective = (
                (non_gemm_flops / 1e9) / phase2_non_gemm_s
                if phase2_non_gemm_s > 0
                else 0.0
            )
            compute_share = (
                100.0 * compute_time_total_s / gemm_active_s
                if gemm_active_s > 0
                else 0.0
            )
            workload_ok = (
                metrics["computeCycles"] > 0
                and metrics["dmaRead"] > 0
                and phase["end_to_end_ms"] > 0.0
            )
            if run_status == "ok" and not workload_ok:
                run_status = "invalid_false_completion"

            result_row = {
                "Preset": preset["name"],
                "Device Link (GB/s)": args.device_link_gbs,
                "Prefetch Label": args.prefetch_label,
                "Static Classifier": (
                    args.irregular_gemm_static_output_tile_classifier_boundary_hold_first_cut
                ),
                "Workload OK": workload_ok,
                "Run Status": run_status,
                "SeqLen": seq_len,
                "HiddenDim": preset["hidden_dim"],
                "MLPDim": preset["mlp_dim"],
                "NumHeads": preset["num_heads"],
                "GEMM Proxy Size": seq_len,
                "phase1_ms": round(phase["phase1_ms"], 6),
                "phase2_total_ms": round(phase["phase2_total_ms"], 6),
                "phase2_non_gemm_ms": round(phase["phase2_non_gemm_ms"], 6),
                "phase3_ms": round(phase["phase3_ms"], 6),
                "end_to_end_ms": round(phase["end_to_end_ms"], 6),
                "GEMM Active Time (s)": round(gemm_active_s, 9),
                "GEMM-Effective (GFLOPS)": round(
                    gemm_effective if workload_ok else 0.0, 6
                ),
                "Phase2 Proxy Throughput (GFLOPS)": round(
                    phase2_proxy_throughput if workload_ok else 0.0, 6
                ),
                "Non-GEMM Effective (GFLOPS)": round(
                    non_gemm_effective if workload_ok else 0.0, 6
                ),
                "System-Effective (GFLOPS)": round(
                    system_effective if workload_ok else 0.0, 6
                ),
                "ComputeShare (%)": round(
                    compute_share if workload_ok else 0.0, 6
                ),
                "totalDmaRead": int(metrics["dmaRead"]),
                "totalDmaWrite": int(metrics["dmaWrite"]),
                "totalComputeCycles": int(metrics["computeCycles"]),
                "m5out_dir": str(m5out_dir),
                "terminal_log": str(log_file),
                "serial_log": str(serial_log),
            }
            results.append(result_row)

            print(
                f"[{'√' if workload_ok else '!'}] {preset['name']} 完成: "
                f"end_to_end_ms={result_row['end_to_end_ms']}, "
                f"phase2_ms={result_row['phase2_total_ms']}, "
                f"GEMM-Effective={result_row['GEMM-Effective (GFLOPS)']} GFLOPS, "
                f"Phase2-Proxy={result_row['Phase2 Proxy Throughput (GFLOPS)']} GFLOPS, "
                f"Non-GEMM-Effective={result_row['Non-GEMM Effective (GFLOPS)']} GFLOPS, "
                f"status={run_status}"
            )

    finally:
        TRIGGER_SRC.write_text(original_trigger, encoding="utf-8")

    if results:
        headers = list(results[0].keys())
        with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=headers)
            writer.writeheader()
            writer.writerows(results)

        with SUMMARY_TXT.open("w", encoding="utf-8") as fh:
            for row in results:
                fh.write(str(row) + "\n")

        print("\n" + "=" * 50)
        print(f"🎉 CXL ViT proxy 完成！数据已保存至: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
