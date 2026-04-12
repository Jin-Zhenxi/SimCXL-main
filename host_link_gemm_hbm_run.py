#!/usr/bin/env python3
"""Sweep host-device link bandwidth for host_link_gemm on PCIe+HBM."""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
from pathlib import Path

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
SUMMARY_CSV = OUTPUT_DIR / "host_link_gemm_hbm_bandwidth_sweep.csv"
SUMMARY_TXT = OUTPUT_DIR / "host_link_gemm_hbm_bandwidth_sweep.txt"
SUMMARY_SVG = OUTPUT_DIR / "host_link_gemm_hbm_bandwidth_sweep.svg"


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
        description="Sweep host-device link bandwidth for host_link_gemm."
    )
    parser.add_argument(
        "--matrix-size",
        "--matrix_size",
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
        default="16,32,64,128,256",
        help="Comma-separated host-link bandwidths in GB/s.",
    )
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--skip-inject", action="store_true")
    parser.add_argument("--keep-existing", action="store_true")
    return parser.parse_args()


def parse_bandwidths(raw: str) -> list[int]:
    valid = {16, 32, 64, 128, 256}
    values = []
    seen = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value not in valid:
            raise SystemExit("只支持 16,32,64,128,256 GB/s；" f"收到非法值 {value}。")
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


def extract_host_link_debug_stats(stats_file: Path) -> dict[str, float]:
    values = {
        "bridgeReqSendSucceed": 0.0,
        "xbarReqPayloadBytes": 0.0,
        "xbarReadReqCount": 0.0,
        "xbarWriteReqCount": 0.0,
    }
    if not stats_file.exists():
        return values

    text = stats_file.read_text(encoding="utf-8", errors="ignore")
    patterns = {
        "bridgeReqSendSucceed": r"^board\.cxl_bridge\.reqSendSucceed\s+([0-9.]+)",
        "xbarReqPayloadBytes": (
            r"^board\.cxl_xbar\.pktSize_board\.cxl_bridge\.mem_side_port::total\s+"
            r"([0-9.]+)"
        ),
        "xbarReadReqCount": r"^board\.cxl_xbar\.transDist::ReadReq\s+([0-9.]+)",
        "xbarWriteReqCount": r"^board\.cxl_xbar\.transDist::WriteReq\s+([0-9.]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            values[key] = float(match.group(1))
    return values


def extract_phase_timings(serial_log: Path) -> dict[str, float]:
    keys = [
        "phase1_ms",
        "end_to_end_ms",
        "host_link_h2d_ms",
        "host_link_d2h_ms",
        "host_link_total_ms",
        "host_link_read_bytes",
        "host_link_write_bytes",
        "host_link_total_bytes",
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


def serial_log_has_workload(serial_log: Path, workload: str) -> bool:
    if not serial_log.exists():
        return False
    text = serial_log.read_text(encoding="utf-8", errors="ignore")
    return (
        f"[Timing] workload={workload}" in text
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

    x_vals = [float(row["Host Link (GB/s)"]) for row in plot_rows]
    y_vals = [
        max(
            float(row["End-to-End Effective (GFLOPS)"]),
            float(row["Roofline (GFLOPS)"]),
        )
        for row in plot_rows
    ]
    x_min = min(x_vals)
    x_max = max(x_vals)
    y_max = max(y_vals) * 1.12 if y_vals else 1.0
    if y_max <= 0:
        y_max = 1.0

    def sx(x_val: float) -> float:
        if x_max == x_min:
            return left + plot_w / 2.0
        return left + (x_val - x_min) / (x_max - x_min) * plot_w

    def sy(y_val: float) -> float:
        return top + plot_h - (y_val / y_max) * plot_h

    measured_points = " ".join(
        f"{sx(float(row['Host Link (GB/s)'])):.2f},{sy(float(row['End-to-End Effective (GFLOPS)'])):.2f}"
        for row in plot_rows
    )
    roofline_points = " ".join(
        f"{sx(float(row['Host Link (GB/s)'])):.2f},{sy(float(row['Roofline (GFLOPS)'])):.2f}"
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
            "HBM+PCIe Host-Link GEMM Roofline Sweep"
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
        x_pos = sx(float(row["Host Link (GB/s)"]))
        measured_y = sy(float(row["End-to-End Effective (GFLOPS)"]))
        roof_y = sy(float(row["Roofline (GFLOPS)"]))
        bw = int(row["Host Link (GB/s)"])
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
                "Measured end-to-end effective</text>"
            ),
            (
                f'<line x1="{legend_x}" y1="{legend_y + 28}" '
                f'x2="{legend_x + 42}" y2="{legend_y + 28}" '
                'stroke="#9aa0a6" stroke-width="4" stroke-dasharray="10 8"/>'
            ),
            (
                f'<text x="{legend_x + 52}" y="{legend_y + 33}" '
                'font-size="15" font-family="sans-serif" fill="#222">'
                "Roofline bound</text>"
            ),
            (
                f'<text x="{left + plot_w / 2:.2f}" y="{height - 18}" '
                'text-anchor="middle" font-size="18" '
                'font-family="sans-serif" fill="#222">'
                "Host-device bandwidth (GB/s)</text>"
            ),
            (
                f'<text x="28" y="{top + plot_h / 2:.2f}" '
                'text-anchor="middle" font-size="18" '
                'font-family="sans-serif" fill="#222" '
                f'transform="rotate(-90 28 {top + plot_h / 2:.2f})">'
                "GFLOPS</text>"
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

    for host_link_gbs in bandwidths:
        run_tag = (
            f"host_link_gemm_{args.matrix_size}_pcie_hbm_{host_link_gbs}g"
        )
        m5out_dir = OUTPUT_DIR / f"m5out_{run_tag}"
        terminal_log = OUTPUT_DIR / f"terminal_log_{run_tag}.txt"
        serial_log = m5out_dir / "board.pc.com_1.device"
        stats_file = m5out_dir / "stats.txt"

        if not args.keep_existing:
            if m5out_dir.exists():
                shutil.rmtree(m5out_dir)
            if terminal_log.exists():
                terminal_log.unlink()

        cmd = (
            f"{GEM5_BIN} -p {GEM5_ROOT / 'src/python'} "
            f"-d {m5out_dir} "
            f"{GEM5_SCRIPT} "
            f"--is_asic True "
            f"--cpu_type {args.cpu_type} "
            "--no-network "
            f"--matrixflow_size {args.matrix_size} "
            "--matrixflow_workload host_link_gemm "
            f"--host-link-gbs {host_link_gbs} "
            "--allow-local-trigger "
            f"> {terminal_log} 2>&1"
        )

        process_return_code = 0
        run_status = "unknown"
        error_msg = ""
        try:
            run_cmd(cmd)
        except subprocess.CalledProcessError as exc:
            process_return_code = exc.returncode
            error_msg = str(exc)

        metrics = extract_roi_metrics(stats_file)
        debug = extract_host_link_debug_stats(stats_file)
        phase = extract_phase_timings(serial_log)
        workload_ok = serial_log_has_workload(serial_log, "host_link_gemm")
        run_status = (
            "ok"
            if workload_ok
            else (
                f"failed({process_return_code})"
                if process_return_code
                else "failed"
            )
        )
        phase1_s = phase["phase1_ms"] / 1.0e3 if phase["phase1_ms"] else 0.0
        end_to_end_s = (
            phase["end_to_end_ms"] / 1.0e3 if phase["end_to_end_ms"] else 0.0
        )
        link_copy_s = (
            phase["host_link_total_ms"] / 1.0e3
            if phase["host_link_total_ms"]
            else 0.0
        )
        kernel_gflops = (
            (gemm_flops / 1.0e9) / phase1_s if phase1_s > 0 else 0.0
        )
        effective_gflops = (
            (gemm_flops / 1.0e9) / end_to_end_s if end_to_end_s > 0 else 0.0
        )
        compute_time_s = (
            metrics["computeCycles"] * 2.0 / (CLOCK_FREQ_GHZ * 1.0e9)
            if metrics["computeCycles"] > 0
            else 0.0
        )
        compute_share = (
            100.0 * compute_time_s / end_to_end_s if end_to_end_s > 0 else 0.0
        )
        host_link_total_bytes = phase["host_link_total_bytes"]
        operational_intensity = (
            gemm_flops / host_link_total_bytes
            if host_link_total_bytes > 0
            else 0.0
        )
        measured_link_bw = (
            (host_link_total_bytes / 1.0e9) / link_copy_s
            if link_copy_s > 0
            else 0.0
        )
        avg_req_payload = (
            debug["xbarReqPayloadBytes"] / debug["bridgeReqSendSucceed"]
            if debug["bridgeReqSendSucceed"] > 0
            else 0.0
        )

        row = {
            "Label": f"host_link_gemm_pcie_hbm_{host_link_gbs}g",
            "Host Link (GB/s)": host_link_gbs,
            "CXL XBar Width (B)": host_link_gbs // 2,
            "Matrix Size": args.matrix_size,
            "Run Status": run_status,
            "Process Return Code": process_return_code,
            "Workload OK": workload_ok,
            "ROI simSeconds": round(metrics["simSeconds"], 9),
            "totalComputeCycles": int(metrics["computeCycles"]),
            "DMA Read Bytes": int(metrics["dmaRead"]),
            "DMA Write Bytes": int(metrics["dmaWrite"]),
            "Host Link Read Bytes": int(phase["host_link_read_bytes"]),
            "Host Link Write Bytes": int(phase["host_link_write_bytes"]),
            "Host Link Total Bytes": int(host_link_total_bytes),
            "Operational Intensity (FLOP/B)": round(operational_intensity, 6),
            "Phase-1 GEMM Time (ms)": round(phase["phase1_ms"], 6),
            "Host Link H2D Time (ms)": round(phase["host_link_h2d_ms"], 6),
            "Host Link D2H Time (ms)": round(phase["host_link_d2h_ms"], 6),
            "Host Link Total Time (ms)": round(phase["host_link_total_ms"], 6),
            "End-to-End Time (ms)": round(phase["end_to_end_ms"], 6),
            "GEMM Compute Share (%)": round(compute_share, 6),
            "GEMM-Effective (GFLOPS)": round(kernel_gflops, 6),
            "End-to-End Effective (GFLOPS)": round(effective_gflops, 6),
            "Measured Host-Link BW (GB/s)": round(measured_link_bw, 6),
            "Host Link Req Count": int(debug["bridgeReqSendSucceed"]),
            "Host Link Req Payload Bytes": int(debug["xbarReqPayloadBytes"]),
            "Host Link Avg Req Payload (B)": round(avg_req_payload, 6),
            "Host Link ReadReq Count": int(debug["xbarReadReqCount"]),
            "Host Link WriteReq Count": int(debug["xbarWriteReqCount"]),
            "terminal_log": str(terminal_log),
            "serial_log": str(serial_log),
            "m5out_dir": str(m5out_dir),
            "Error": error_msg,
        }
        rows.append(row)

        print(
            f"[{'√' if workload_ok else '!'}] "
            f"host_link_gemm_{host_link_gbs}g: "
            f"phase1_ms={row['Phase-1 GEMM Time (ms)']}, "
            f"end_to_end_ms={row['End-to-End Time (ms)']}, "
            f"gemm_effective={row['GEMM-Effective (GFLOPS)']} GFLOPS, "
            f"effective={row['End-to-End Effective (GFLOPS)']} GFLOPS, "
            f"link_bw={row['Measured Host-Link BW (GB/s)']} GB/s, "
            f"avg_req={row['Host Link Avg Req Payload (B)']} B, "
            f"status={run_status}"
        )

    valid_rows = [row for row in rows if row["Workload OK"]]
    compute_peak = max(
        (float(row["GEMM-Effective (GFLOPS)"]) for row in valid_rows),
        default=0.0,
    )
    for row in rows:
        link_roof = min(
            compute_peak,
            float(row["Operational Intensity (FLOP/B)"])
            * float(row["Host Link (GB/s)"]),
        )
        roofline_ratio = (
            100.0 * float(row["End-to-End Effective (GFLOPS)"]) / link_roof
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
            "bandwidth={Host Link (GB/s)} GB/s | "
            "status={Run Status} | "
            "ok={Workload OK} | "
            "phase1_ms={Phase-1 GEMM Time (ms)} | "
            "end_to_end_ms={End-to-End Time (ms)} | "
            "gemm_effective={GEMM-Effective (GFLOPS)} GFLOPS | "
            "effective={End-to-End Effective (GFLOPS)} GFLOPS | "
            "roofline={Roofline (GFLOPS)} GFLOPS | "
            "link_bw={Measured Host-Link BW (GB/s)} GB/s | "
            "avg_req={Host Link Avg Req Payload (B)} B"
        ).format(**row)
        for row in rows
    ]
    SUMMARY_TXT.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    render_svg(rows)

    print(f"[done] sweep 完成，结果已写入 {SUMMARY_CSV}，" f"图已写入 {SUMMARY_SVG}")


if __name__ == "__main__":
    main()
