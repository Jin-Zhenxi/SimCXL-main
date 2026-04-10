#!/usr/bin/env python3
"""
Oracle/profiler sweep for high-value B analysis.

Goal:
- run a small but diverse workload set
- keep the architectural path fixed
- collect new oracle profiler stats for B source/value analysis
"""

import csv
import os
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path

import run_sweep as rs

OUTPUT_DIR = "my_outputData"
CSV_FILE = os.path.join(OUTPUT_DIR, "b_value_oracle_sweep.csv")
RAW_TXT_FILE = os.path.join(OUTPUT_DIR, "b_value_oracle_sweep_raw.txt")
GEM5_BIN = Path("build/X86/gem5.opt")
BUILD_P2P = Path("./build_p2p.sh")
ORACLE_SOURCE_FILES = [
    Path("src/mem/matrixflow_engine.cc"),
    Path("src/mem/matrixflow_engine.hh"),
    Path("src/mem/MatrixFlowEngine.py"),
    Path("src/dev/x86/CXLDevice.py"),
]

FORCE_RERUN = False
MAX_INVALID_RUN_RETRIES = 3

MODE_LABELS = [
    "baseline_serial",
    "ab_hierarchical_claim_hole_filling",
]

ANALYSIS_PRESETS = [
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
        "name": "Kernel-S128",
        "seq_len": 128,
        "hidden_dim": 1024,
        "mlp_dim": 4096,
        "num_heads": 16,
    },
    {
        "name": "Kernel-S192",
        "seq_len": 192,
        "hidden_dim": 1024,
        "mlp_dim": 4096,
        "num_heads": 16,
    },
    {
        "name": "Kernel-S256",
        "seq_len": 256,
        "hidden_dim": 1024,
        "mlp_dim": 4096,
        "num_heads": 16,
    },
    {
        "name": "Kernel-S320",
        "seq_len": 320,
        "hidden_dim": 1024,
        "mlp_dim": 4096,
        "num_heads": 16,
    },
]

ORACLE_SOURCES = [
    "next_output",
    "claim",
    "carry_over",
    "current_window",
    "normal",
]
ORACLE_DISTANCES = ["immediate", "near", "far"]


def write_results(path, rows):
    if not rows:
        return
    headers = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                headers.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def load_existing_results():
    if not os.path.exists(CSV_FILE) or os.path.getsize(CSV_FILE) == 0:
        return []
    with open(CSV_FILE, newline="") as f:
        return list(csv.DictReader(f))


def completed_case_keys(rows):
    keys = set()
    for row in rows:
        if row.get("Preset") and row.get("Prefetch Label"):
            keys.add((row["Preset"], row["Prefetch Label"]))
    return keys


def existing_results_are_stale():
    if FORCE_RERUN or not GEM5_BIN.exists():
        return False
    if not os.path.exists(CSV_FILE) or os.path.getsize(CSV_FILE) == 0:
        return False
    return os.path.getmtime(CSV_FILE) < GEM5_BIN.stat().st_mtime


def kernel_best_rows_b():
    path = Path(OUTPUT_DIR) / "small_tile_search.csv"
    if not path.exists():
        return {}
    rows = list(csv.DictReader(path.open()))
    best = {}
    for preset in {r["Preset"] for r in rows if r["Preset"].startswith("Kernel-")}:
        items = [r for r in rows if r["Preset"] == preset]
        items.sort(
            key=lambda r: (
                float(r["GEMM Compute Share (%)"]),
                float(r["GEMM-Effective (GFLOPS)"]),
                -float(r["Phase1/3 time (ms)"]),
            ),
            reverse=True,
        )
        best[preset] = int(items[0]["Prefetch Rows B"])
    return best


def ensure_gem5_binary():
    rebuild = not GEM5_BIN.exists()
    if not rebuild:
        gem5_mtime = GEM5_BIN.stat().st_mtime
        rebuild = any(
            src.exists() and src.stat().st_mtime > gem5_mtime
            for src in ORACLE_SOURCE_FILES
        )

    if not rebuild:
        print(f"[*] gem5.opt 已是最新，跳过重编: {GEM5_BIN}")
        return

    print("[*] 检测到 Oracle 相关源码更新，正在通过 build_p2p.sh 重建 gem5.opt ...")
    if BUILD_P2P.exists():
        subprocess.run(
            "./build_p2p.sh",
            shell=True,
            check=True,
        )
    else:
        subprocess.run(
            "scons build/X86/gem5.opt -j4 USE_TCMALLOC=False",
            shell=True,
            check=True,
        )


def active_configs():
    cfgs = []
    for label in MODE_LABELS:
        cfg = next(
            deepcopy(item) for item in rs.PREFETCH_CONFIGS if item["label"] == label
        )
        cfgs.append(cfg)
    return cfgs


def resolve_rows_b(preset, cfg, kernel_rows):
    preset_name = preset["name"]
    if preset_name in cfg.get("rows_b_by_preset", {}):
        return cfg["rows_b_by_preset"][preset_name]
    return kernel_rows.get(preset_name, cfg["rows_b"])


def parse_oracle_vector2d(stats_file, base_name):
    values = {}
    if not os.path.exists(stats_file):
        return values
    text = Path(stats_file).read_text(encoding="utf-8", errors="ignore")
    for src in ORACLE_SOURCES:
        for dist in ORACLE_DISTANCES:
            key = f"{base_name}_{src}_{dist}"
            pattern = (
                rf"matrix_engine\.{base_name}_{src}::"
                rf"{dist}\s+([0-9.]+)"
            )
            m = None
            for match in re_finditer_last(pattern, text):
                m = match
            values[key] = float(m.group(1)) if m else 0.0
    return values


def re_finditer_last(pattern, text):
    import re

    return re.finditer(pattern, text)


def build_row_from_outputs(preset, cfg, rows_b, m5out_dir, log_file):
    stats_file = os.path.join(m5out_dir, "stats.txt")
    serial_log = os.path.join(m5out_dir, "board.pc.com_1.device")

    metrics = rs.extract_metrics(stats_file, log_file)
    prefetch = rs.extract_prefetch_stats(stats_file)
    oracle = {}
    for base in [
        "oracleLocalRows",
        "oracleFallbackRows",
        "oracleLocalReuseWeight",
        "oracleFallbackReuseWeight",
    ]:
        oracle.update(parse_oracle_vector2d(stats_file, base))

    valid_metrics = not (
        metrics["simSeconds"] == 0.0
        and metrics["dmaRead"] == 0
        and metrics["dmaWrite"] == 0
        and metrics["computeCycles"] == 0
    )
    valid_serial = rs.serial_log_has_benchmark(serial_log)
    if not (valid_metrics and valid_serial):
        raise RuntimeError(f"现有结果无效: {m5out_dir}")

    phase = rs.extract_phase_timings(serial_log)
    gemm_active_s = rs.extract_gemm_active_time(serial_log)
    dma_bytes = metrics["dmaRead"] + metrics["dmaWrite"]
    compute_cycles = metrics["computeCycles"]
    compute_time_total_s = (
        compute_cycles * 2.0 / (rs.CLOCK_FREQ_GHZ * 1e9)
        if compute_cycles > 0
        else 0.0
    )
    gemm_compute_share = (
        100.0 * compute_time_total_s / gemm_active_s if gemm_active_s > 0 else 0.0
    )
    gemm_effective_gflops = (
        (4.0 * (preset["seq_len"] ** 3) / 1e9) / gemm_active_s
        if gemm_active_s > 0
        else 0.0
    )
    gemm_dma_bw = (dma_bytes / 1e9) / gemm_active_s if gemm_active_s > 0 else 0.0

    row = {
        "Preset": preset["name"],
        "Preset Class": "ViT" if preset["name"].startswith("ViT-") else "Kernel",
        "SeqLen": preset["seq_len"],
        "HiddenDim": preset["hidden_dim"],
        "MLPDim": preset["mlp_dim"],
        "NumHeads": preset["num_heads"],
        "Prefetch Label": cfg["label"],
        "Scheduler Mode": cfg.get("ab_scheduler_mode", "baseline"),
        "Prefetch Rows B": rows_b,
        "Phase1/3 time (ms)": round(phase["phase1_ms"] + phase["phase3_ms"], 6),
        "GEMM Compute Share (%)": round(gemm_compute_share, 4),
        "GEMM-Effective (GFLOPS)": round(gemm_effective_gflops, 6),
        "GEMM DMA BW (GB/s)": round(gemm_dma_bw, 6),
        "prefetched_b_rows_consumed": prefetch["prefetched_b_rows_consumed"],
        "fallback_b_rows_fetched": prefetch["fallback_b_rows_fetched"],
        "normal_fetch_hole_rows": prefetch["normal_fetch_hole_rows"],
        "next_output_prefetch_issue_count": prefetch[
            "next_output_prefetch_issue_count"
        ],
        "next_output_rows_ready_at_boundary": prefetch[
            "next_output_rows_ready_at_boundary"
        ],
        "next_output_consumed_before_fallback_rows": prefetch[
            "next_output_consumed_before_fallback_rows"
        ],
        "next_output_prefetch_rows_issued": prefetch[
            "next_output_prefetch_rows_issued"
        ],
        "carry_over_rows_consumed_post_boundary": prefetch[
            "carry_over_rows_consumed_post_boundary"
        ],
        "future_claim_set_count": prefetch["future_claim_set_count"],
        "future_claim_blocked_normal_fetch_count": prefetch[
            "future_claim_blocked_normal_fetch_count"
        ],
        "future_claim_consumed_success_count": prefetch[
            "future_claim_consumed_success_count"
        ],
    }
    row.update(oracle)
    return row


def oracle_row_has_signal(row):
    return any(
        key.startswith("oracle") and float(value) != 0.0
        for key, value in row.items()
    )


def run_one(preset, cfg, rows_b):
    preset_tag = preset["name"].replace(" ", "_").replace("/", "_")
    run_tag = f"{preset_tag}__oracle__{cfg['label']}"

    rs.modify_c_file(preset)

    print(
        f"\n>>> Oracle sweep: {preset['name']} / {cfg['label']} "
        f"(rows_b={rows_b}) <<<"
    )
    print("[*] 正在编译 trigger_gemm...")
    subprocess.run(rs.COMPILE_CMD, shell=True, check=True)

    inject_cmd = f"{rs.INJECT_CMD} --no-compile {preset['seq_len']}"
    print(f"[*] 正在注入镜像 (seq_len={preset['seq_len']})...")
    subprocess.run(inject_cmd, shell=True, check=True)

    env = os.environ.copy()
    env["MATRIXFLOW_NEXT_PREFETCH_MODE"] = cfg["mode"]
    env["MATRIXFLOW_NEXT_PREFETCH_TRIGGER"] = cfg["trigger"]
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
    env["MATRIXFLOW_AB_SCHEDULER_MODE"] = cfg.get(
        "ab_scheduler_mode", "baseline"
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

    metrics = None
    bus_dma = None
    prefetch = None
    oracle = None
    serial_log = None
    log_file = None

    for attempt in range(1, MAX_INVALID_RUN_RETRIES + 1):
        m5out_dir = os.path.join(OUTPUT_DIR, f"m5out_{run_tag}")
        log_file = os.path.join(OUTPUT_DIR, f"terminal_log_{run_tag}.txt")
        if os.path.exists(m5out_dir):
            shutil.rmtree(m5out_dir)
        if os.path.exists(log_file):
            os.remove(log_file)

        cmd = rs.GEM5_CMD_TEMPLATE.format(outdir=m5out_dir, logfile=log_file)
        print(
            f"[*] 正在运行 gem5 仿真，日志存入: {log_file} "
            f"(attempt {attempt}/{MAX_INVALID_RUN_RETRIES})"
        )
        rc = subprocess.run(cmd, shell=True, env=env)
        if rc.returncode != 0:
            raise SystemExit(
                f"gem5 仿真失败，返回码={rc.returncode}。请检查日志: {log_file}"
            )

        stats_file = os.path.join(m5out_dir, "stats.txt")
        serial_log = os.path.join(m5out_dir, "board.pc.com_1.device")
        metrics = rs.extract_metrics(stats_file, log_file)
        bus_dma = rs.extract_bus_dma_stats(stats_file)
        valid_metrics = not (
            metrics["simSeconds"] == 0.0
            and metrics["dmaRead"] == 0
            and metrics["dmaWrite"] == 0
            and metrics["computeCycles"] == 0
        )
        valid_serial = rs.serial_log_has_benchmark(serial_log)
        if valid_metrics and valid_serial:
            break
        print(f"[!] 无效 run：{run_tag}，准备重试")
    else:
        raise SystemExit(f"多次重试后仍未拿到有效结果: {run_tag}")

    return build_row_from_outputs(preset, cfg, rows_b, m5out_dir, log_file)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rs.ensure_libm5()
    ensure_gem5_binary()

    kernel_rows = kernel_best_rows_b()
    configs = active_configs()
    stale_results = existing_results_are_stale()
    if FORCE_RERUN:
        print("[*] FORCE_RERUN=1，本轮忽略已有 Oracle sweep 结果。")
    elif stale_results:
        print("[*] 检测到已有 Oracle sweep 结果早于当前 gem5.opt，将重新全量跑。")
    results = [] if (FORCE_RERUN or stale_results) else load_existing_results()
    completed = set() if (FORCE_RERUN or stale_results) else completed_case_keys(results)

    print("🚀 Oracle / B-value Profiler Sweep")
    print(
        "   工作集: "
        + ", ".join(
            f"{p['name']}[S={p['seq_len']},H={p['hidden_dim']}]"
            for p in ANALYSIS_PRESETS
        )
    )
    print("   模式: " + ", ".join(cfg["label"] for cfg in configs))
    print("   Kernel rows_b 基于 small_tile_search.csv 最佳点自动选择")
    print("=" * 60)

    by_key = {
        (row["Preset"], row["Prefetch Label"]): row
        for row in results
    }
    for cfg in configs:
        for preset in ANALYSIS_PRESETS:
            case_key = (preset["name"], cfg["label"])
            rows_b = resolve_rows_b(preset, cfg, kernel_rows)
            if case_key in completed:
                existing = by_key.get(case_key)
                preset_tag = preset["name"].replace(" ", "_").replace("/", "_")
                run_tag = f"{preset_tag}__oracle__{cfg['label']}"
                m5out_dir = os.path.join(OUTPUT_DIR, f"m5out_{run_tag}")
                log_file = os.path.join(OUTPUT_DIR, f"terminal_log_{run_tag}.txt")
                if existing and oracle_row_has_signal(existing):
                    print(f"[-] 跳过已完成点: {preset['name']} / {cfg['label']}")
                    continue
                print(
                    f"[*] 重新解析已有结果: {preset['name']} / {cfg['label']}"
                )
                row = build_row_from_outputs(preset, cfg, rows_b, m5out_dir, log_file)
                for i, item in enumerate(results):
                    if (
                        item.get("Preset") == preset["name"]
                        and item.get("Prefetch Label") == cfg["label"]
                    ):
                        results[i] = row
                        break
                by_key[case_key] = row
                write_results(CSV_FILE, results)
                with open(RAW_TXT_FILE, "w") as f:
                    for item in results:
                        f.write(str(item) + "\n")
                continue

            row = run_one(preset, cfg, rows_b)
            results.append(row)
            by_key[case_key] = row
            write_results(CSV_FILE, results)
            with open(RAW_TXT_FILE, "w") as f:
                for item in results:
                    f.write(str(item) + "\n")
            print(
                f"[√] {preset['name']} / {cfg['label']} 完成: "
                f"share={row['GEMM Compute Share (%)']:.4f}% "
                f"gemm={row['GEMM-Effective (GFLOPS)']:.3f} GFLOPS"
            )

    print("\n" + "=" * 60)
    print(f"✅ Oracle sweep 完成，结果写入: {CSV_FILE}")


if __name__ == "__main__":
    main()
