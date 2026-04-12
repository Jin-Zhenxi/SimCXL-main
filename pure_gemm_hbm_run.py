#!/usr/bin/env python3
"""Sweep device-side HBM data-plane bandwidth for pure GEMM."""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
from pathlib import Path

from run_sweep import extract_gemm_active_time

GEM5_ROOT = Path("/home/jzx8091/SimCXL-main")
OUTPUT_DIR = GEM5_ROOT / "my_outputData"
TRIGGER_BIN = GEM5_ROOT / "trigger_gemm"
TRIGGER_SRC = GEM5_ROOT / "trigger_gemm.c"
GEM5_BIN = GEM5_ROOT / "build/X86/gem5.opt"
GEM5_SCRIPT = "configs/example/gem5_library/x86-cxl-pcie-hbm-pure-gemm.py"
LIBM5_PATH = GEM5_ROOT / "util/m5/build/x86/out/libm5.a"
COMPILE_CMD = (
    f"gcc -O3 -static -mno-80387 -o {TRIGGER_BIN} {TRIGGER_SRC} "
    f"-I{GEM5_ROOT}/include -L{GEM5_ROOT}/util/m5/build/x86/out -lm -lm5"
)
INJECT_CMD_TEMPLATE = (
    "sudo ./inject_trigger_gemm.sh --no-compile {matrix_size}"
)
CHUNK_SEPARATOR = "---------- Begin Simulation Statistics ----------"
CLOCK_FREQ_GHZ = 2.4
SUMMARY_CSV = OUTPUT_DIR / "pure_gemm_hbm_bandwidth_sweep.csv"
SUMMARY_TXT = OUTPUT_DIR / "pure_gemm_hbm_bandwidth_sweep.txt"
SUMMARY_SVG = OUTPUT_DIR / "pure_gemm_hbm_bandwidth_sweep.svg"


def ensure_libm5() -> None:
    if LIBM5_PATH.exists():
        return
    raise SystemExit(
        "libm5.a 未找到。请先执行: "
        f"cd {GEM5_ROOT / 'util/m5'} && scons build/x86/out/libm5.a"
    )


def run_cmd(cmd: str) -> None:
    print(f"[*] {cmd}")
    subprocess.run(cmd, shell=True, cwd=GEM5_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep device-side HBM data-plane bandwidth for pure_gemm."
    )
    parser.add_argument(
        "--matrix-size",
        "--matrix_size",
        "--matrixflow-size",
        "--matrixflow_size",
        dest="matrix_size",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--cpu-type",
        "--cpu_type",
        dest="cpu_type",
        choices=["TIMING", "O3"],
        default="TIMING",
    )
    parser.add_argument(
        "--bandwidths",
        default="16,32,64,128",
        help="Comma-separated device-side data-plane bandwidths in GB/s.",
    )
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--skip-inject", action="store_true")
    parser.add_argument("--keep-existing", action="store_true")
    return parser.parse_args()


def parse_bandwidths(raw: str) -> list[int]:
    values = []
    seen = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value not in {16, 32, 64, 128, 256}:
            raise SystemExit(
                "只支持 16,32,64,128,256 GB/s 五个点；" f"收到非法值 {value}。"
            )
        if value not in seen:
            values.append(value)
            seen.add(value)
    if not values:
        raise SystemExit("bandwidths 不能为空。")
    return values


def extract_roi_metrics(stats_file: Path) -> dict[str, float]:
    metrics = {
        "simSeconds": 0.0,
        "dmaRead": 0.0,
        "dmaWrite": 0.0,
        "computeCycles": 0.0,
    }
    if not stats_file.exists():
        return metrics

    text = stats_file.read_text(encoding="utf-8", errors="ignore")
    chunks = text.split(CHUNK_SEPARATOR)
    target = ""
    for chunk in chunks:
        match = re.search(r"totalComputeCycles\s+(\d+)", chunk)
        if match and int(match.group(1)) > 0:
            target = chunk
            break
    if not target and chunks:
        target = chunks[-1]

    for key, pattern in {
        "simSeconds": r"simSeconds\s+([0-9.]+)",
        "dmaRead": r"totalDmaBytesRead\s+(\d+)",
        "dmaWrite": r"totalDmaBytesWritten\s+(\d+)",
        "computeCycles": r"totalComputeCycles\s+(\d+)",
    }.items():
        match = re.search(pattern, target)
        if match:
            metrics[key] = float(match.group(1))
    return metrics


def patch_metrics_from_debug_log(
    metrics: dict[str, float], terminal_log: Path
) -> dict[str, float]:
    if not terminal_log.exists():
        return metrics

    text = terminal_log.read_text(encoding="utf-8", errors="ignore")

    if metrics["computeCycles"] <= 0:
        match = re.search(r"computeCycles=(\d+)", text)
        if match:
            metrics["computeCycles"] = float(match.group(1))

    if metrics["dmaRead"] <= 0:
        match = re.search(r"dmaRead=(\d+)", text)
        if match:
            metrics["dmaRead"] = float(match.group(1))

    if metrics["dmaWrite"] <= 0:
        match = re.search(r"dmaWrite=(\d+)", text)
        if match:
            metrics["dmaWrite"] = float(match.group(1))

    return metrics


def extract_phase_timings(serial_log: Path) -> dict[str, float]:
    keys = [
        "phase1_ms",
        "phase2_total_ms",
        "phase3_ms",
        "end_to_end_ms",
        "gemm_tile_count_total",
        "phase1_poll_count",
    ]
    values = {key: 0.0 for key in keys}
    if not serial_log.exists():
        return values

    patterns = {
        key: re.compile(rf"\[Timing\] {re.escape(key)}=([0-9.]+)")
        for key in keys
    }
    with serial_log.open(encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            for key, pattern in patterns.items():
                match = pattern.search(line)
                if match:
                    values[key] = float(match.group(1))
    return values


def serial_log_has_pure_gemm(serial_log: Path) -> bool:
    if not serial_log.exists():
        return False
    text = serial_log.read_text(encoding="utf-8", errors="ignore")
    return (
        "[Timing] workload=pure_gemm" in text
        and "[Phase 2] skipped for pure_gemm" in text
        and "[Phase 3] skipped for pure_gemm" in text
        and "MatrixFlow benchmark finished" in text
    )


def render_svg(rows: list[dict[str, object]]) -> None:
    plot_rows = [
        row for row in rows if row["Run Status"] == "ok" and row["Workload OK"]
    ]
    if not plot_rows:
        return

    width = 920
    height = 560
    left = 90
    right = 40
    top = 60
    bottom = 70
    plot_w = width - left - right
    plot_h = height - top - bottom

    x_vals = [float(row["Device Link (GB/s)"]) for row in plot_rows]
    y_vals = [
        max(
            float(row["GEMM-Effective (GFLOPS)"]),
            float(row["Roofline (GFLOPS)"]),
        )
        for row in plot_rows
    ]
    x_min = min(x_vals)
    x_max = max(x_vals)
    y_max = max(y_vals) * 1.12
    if y_max <= 0:
        y_max = 1.0

    def sx(x_val: float) -> float:
        if x_max == x_min:
            return left + plot_w / 2.0
        return left + (x_val - x_min) / (x_max - x_min) * plot_w

    def sy(y_val: float) -> float:
        return top + plot_h - (y_val / y_max) * plot_h

    measured_points = " ".join(
        f"{sx(float(row['Device Link (GB/s)'])):.2f},{sy(float(row['GEMM-Effective (GFLOPS)'])):.2f}"
        for row in plot_rows
    )
    roofline_points = " ".join(
        f"{sx(float(row['Device Link (GB/s)'])):.2f},{sy(float(row['Roofline (GFLOPS)'])):.2f}"
        for row in plot_rows
    )

    x_ticks = [16, 32, 64, 128, 256]
    y_tick_count = 5
    y_ticks = [y_max * idx / y_tick_count for idx in range(y_tick_count + 1)]

    lines = [
        (
            '<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="#fcfcfb"/>',
        (
            '<text x="460" y="32" text-anchor="middle" '
            'font-size="24" font-family="sans-serif" fill="#222">'
            "Device-side HBM Pure GEMM 2048 Roofline Sweep"
            "</text>"
        ),
        (
            f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" '
            f'y2="{top + plot_h}" stroke="#222" stroke-width="2"/>'
        ),
        (
            f'<line x1="{left}" y1="{top}" x2="{left}" '
            f'y2="{top + plot_h}" stroke="#222" stroke-width="2"/>'
        ),
    ]

    for tick in x_ticks:
        x_pos = sx(float(tick))
        lines.extend(
            [
                (
                    f'<line x1="{x_pos:.2f}" y1="{top}" x2="{x_pos:.2f}" '
                    f'y2="{top + plot_h}" stroke="#e0e0e0" stroke-width="1"/>'
                ),
                (
                    f'<text x="{x_pos:.2f}" y="{top + plot_h + 28}" '
                    'text-anchor="middle" font-size="16" '
                    'font-family="sans-serif" fill="#333">'
                    f"{tick}</text>"
                ),
            ]
        )

    for tick in y_ticks:
        y_pos = sy(tick)
        lines.extend(
            [
                (
                    f'<line x1="{left}" y1="{y_pos:.2f}" x2="{left + plot_w}" '
                    f'y2="{y_pos:.2f}" stroke="#e0e0e0" stroke-width="1"/>'
                ),
                (
                    f'<text x="{left - 12}" y="{y_pos + 5:.2f}" '
                    'text-anchor="end" font-size="16" '
                    'font-family="sans-serif" fill="#333">'
                    f"{tick:.0f}</text>"
                ),
            ]
        )

    lines.extend(
        [
            (
                f'<polyline fill="none" stroke="#9aa0a6" stroke-width="4" '
                f'stroke-dasharray="10 8" points="{roofline_points}"/>'
            ),
            (
                f'<polyline fill="none" stroke="#1565c0" stroke-width="4" '
                f'points="{measured_points}"/>'
            ),
        ]
    )

    for row in plot_rows:
        x_pos = sx(float(row["Device Link (GB/s)"]))
        measured_y = sy(float(row["GEMM-Effective (GFLOPS)"]))
        roof_y = sy(float(row["Roofline (GFLOPS)"]))
        bw = int(row["Device Link (GB/s)"])
        lines.extend(
            [
                (
                    f'<circle cx="{x_pos:.2f}" cy="{roof_y:.2f}" r="5" '
                    'fill="#9aa0a6"/>'
                ),
                (
                    f'<circle cx="{x_pos:.2f}" cy="{measured_y:.2f}" r="6" '
                    'fill="#1565c0"/>'
                ),
                (
                    f'<text x="{x_pos:.2f}" y="{measured_y - 12:.2f}" '
                    'text-anchor="middle" font-size="14" '
                    'font-family="sans-serif" fill="#0d47a1">'
                    f"{bw}G</text>"
                ),
            ]
        )

    legend_x = left + plot_w - 230
    legend_y = top + 18
    lines.extend(
        [
            (
                f'<line x1="{legend_x}" y1="{legend_y}" '
                f'x2="{legend_x + 42}" y2="{legend_y}" '
                'stroke="#1565c0" stroke-width="4"/>'
            ),
            (
                f'<text x="{legend_x + 52}" y="{legend_y + 5}" '
                'font-size="15" font-family="sans-serif" fill="#222">'
                "Measured GEMM-Effective</text>"
            ),
            (
                f'<line x1="{legend_x}" y1="{legend_y + 28}" '
                f'x2="{legend_x + 42}" y2="{legend_y + 28}" '
                'stroke="#9aa0a6" stroke-width="4" stroke-dasharray="10 8"/>'
            ),
            (
                f'<text x="{legend_x + 52}" y="{legend_y + 33}" '
                'font-size="15" font-family="sans-serif" fill="#222">'
                "Empirical roofline</text>"
            ),
            (
                f'<text x="{left + plot_w / 2:.2f}" y="{height - 18}" '
                'text-anchor="middle" font-size="18" '
                'font-family="sans-serif" fill="#222">'
                "Device-side data-plane bandwidth (GB/s)</text>"
            ),
            (
                f'<text x="28" y="{top + plot_h / 2:.2f}" '
                'text-anchor="middle" font-size="18" '
                'font-family="sans-serif" fill="#222" '
                f'transform="rotate(-90 28 {top + plot_h / 2:.2f})">'
                "GEMM-Effective / Roofline (GFLOPS)</text>"
            ),
            "</svg>",
        ]
    )

    SUMMARY_SVG.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    bandwidths = parse_bandwidths(args.bandwidths)
    ensure_libm5()

    if not args.skip_compile:
        run_cmd(COMPILE_CMD)

    if not args.skip_inject:
        run_cmd(INJECT_CMD_TEMPLATE.format(matrix_size=args.matrix_size))

    gemm_flops = 2.0 * (args.matrix_size**3)
    rows: list[dict[str, object]] = []
    for device_link_gbs in bandwidths:
        run_tag = f"pure_gemm_{args.matrix_size}_pcie_hbm_{device_link_gbs}g"
        m5out_dir = OUTPUT_DIR / f"m5out_{run_tag}"
        terminal_log = OUTPUT_DIR / f"terminal_log_{run_tag}.txt"
        serial_log = m5out_dir / "board.pc.com_1.device"
        stats_file = m5out_dir / "stats.txt"

        if not args.keep_existing:
            if m5out_dir.exists():
                shutil.rmtree(m5out_dir)
            if terminal_log.exists():
                terminal_log.unlink()

        run_status = "ok"
        error_msg = ""
        reuse_existing = (
            args.keep_existing and m5out_dir.exists() and terminal_log.exists()
        )
        if not reuse_existing:
            cmd = (
                f"{GEM5_BIN} -p {GEM5_ROOT / 'src/python'} "
                "--debug-flags=MatrixFlow "
                f"-d {m5out_dir} "
                f"{GEM5_SCRIPT} "
                f"--is_asic True "
                f"--cpu_type {args.cpu_type} "
                "--no-network "
                f"--matrixflow_size {args.matrix_size} "
                f"--device-link-gbs {device_link_gbs} "
                "--allow-local-trigger "
                f"> {terminal_log} 2>&1"
            )
            try:
                run_cmd(cmd)
            except subprocess.CalledProcessError as exc:
                run_status = f"failed({exc.returncode})"
                error_msg = str(exc)

        metrics = extract_roi_metrics(stats_file)
        metrics = patch_metrics_from_debug_log(metrics, terminal_log)
        phase = extract_phase_timings(serial_log)
        gemm_active_s = extract_gemm_active_time(str(terminal_log))
        gemm_wall_s = (
            gemm_active_s
            if gemm_active_s > 0
            else (phase["phase1_ms"] / 1.0e3 if phase["phase1_ms"] else 0.0)
        )
        effective_gflops = (
            (gemm_flops / 1.0e9) / gemm_wall_s if gemm_wall_s > 0 else 0.0
        )
        compute_time_s = (
            metrics["computeCycles"] * 2.0 / (CLOCK_FREQ_GHZ * 1.0e9)
            if metrics["computeCycles"] > 0
            else 0.0
        )
        compute_share = (
            100.0 * compute_time_s / gemm_wall_s if gemm_wall_s > 0 else 0.0
        )
        total_dma_bytes = metrics["dmaRead"] + metrics["dmaWrite"]
        op_intensity = (
            gemm_flops / total_dma_bytes if total_dma_bytes > 0 else 0.0
        )
        workload_ok = (
            serial_log_has_pure_gemm(serial_log)
            and metrics["computeCycles"] > 0
            and metrics["dmaRead"] > 0
        )
        if run_status == "ok" and not workload_ok:
            run_status = "invalid_false_completion"
            if not error_msg:
                error_msg = "serial log finished, but computeCycles/dmaRead stayed zero"
            effective_gflops = 0.0
            compute_share = 0.0

        row = {
            "Label": f"pcie_hbm_{device_link_gbs}g",
            "Device Link (GB/s)": device_link_gbs,
            "CXL Mem Bus Width (B)": device_link_gbs // 2,
            "Matrix Size": args.matrix_size,
            "Run Status": run_status,
            "Workload OK": workload_ok,
            "ROI simSeconds": round(metrics["simSeconds"], 9),
            "totalComputeCycles": int(metrics["computeCycles"]),
            "DMA Read Bytes": int(metrics["dmaRead"]),
            "DMA Write Bytes": int(metrics["dmaWrite"]),
            "Total DMA Bytes": int(total_dma_bytes),
            "Operational Intensity (FLOP/B)": round(op_intensity, 6),
            "Phase-1 GEMM Time (ms)": round(phase["phase1_ms"], 6),
            "GEMM Active Time (ms)": round(gemm_wall_s * 1.0e3, 6),
            "End-to-End Time (ms)": round(phase["end_to_end_ms"], 6),
            "GEMM Compute Share (%)": round(compute_share, 6),
            "GEMM-Effective (GFLOPS)": round(effective_gflops, 6),
            "terminal_log": str(terminal_log),
            "serial_log": str(serial_log),
            "m5out_dir": str(m5out_dir),
            "Error": error_msg,
        }
        rows.append(row)

        print(
            f"[{'√' if workload_ok and run_status == 'ok' else '!'}] "
            f"pcie_hbm_{device_link_gbs}g: "
            f"phase1_ms={row['Phase-1 GEMM Time (ms)']}, "
            f"GEMM-Effective={row['GEMM-Effective (GFLOPS)']} GFLOPS, "
            f"ComputeShare={row['GEMM Compute Share (%)']}%, "
            f"status={run_status}"
        )

    valid_rows = [
        row for row in rows if row["Run Status"] == "ok" and row["Workload OK"]
    ]
    compute_peak = max(
        (float(row["GEMM-Effective (GFLOPS)"]) for row in valid_rows),
        default=0.0,
    )
    avg_intensity = (
        sum(float(row["Operational Intensity (FLOP/B)"]) for row in valid_rows)
        / len(valid_rows)
        if valid_rows
        else 0.0
    )
    for row in rows:
        link_roof = min(
            compute_peak,
            avg_intensity * float(row["Device Link (GB/s)"]),
        )
        roofline_ratio = (
            100.0 * float(row["GEMM-Effective (GFLOPS)"]) / link_roof
            if link_roof > 0
            else 0.0
        )
        row["Roofline (GFLOPS)"] = round(link_roof, 6)
        row["Roofline Attainment (%)"] = round(roofline_ratio, 6)

    OUTPUT_DIR.mkdir(exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_lines = [
        (
            "bandwidth={Device Link (GB/s)} GB/s | "
            "status={Run Status} | "
            "ok={Workload OK} | "
            "phase1_ms={Phase-1 GEMM Time (ms)} | "
            "GEMM-Effective={GEMM-Effective (GFLOPS)} GFLOPS | "
            "roofline={Roofline (GFLOPS)} GFLOPS | "
            "attainment={Roofline Attainment (%)}%"
        ).format(**row)
        for row in rows
    ]
    SUMMARY_TXT.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    render_svg(rows)

    print(f"[done] sweep 完成，结果已写入 {SUMMARY_CSV}，" f"图已写入 {SUMMARY_SVG}")


if __name__ == "__main__":
    main()
