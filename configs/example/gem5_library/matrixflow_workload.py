import os


def add_matrixflow_args(parser):
    parser.add_argument(
        "--matrixflow-workload",
        "--matrixflow_workload",
        dest="matrixflow_workload",
        choices=["script", "vit_proxy", "pure_gemm"],
        default="script",
        help="Choose script.sh, vit_proxy, or pure_gemm guest workload.",
    )
    parser.add_argument(
        "--matrixflow-size",
        "--matrixflow_size",
        dest="matrixflow_size",
        type=int,
        default=257,
        help="Matrix size / seq_len passed to trigger_gemm.",
    )
    parser.add_argument(
        "--matrixflow-binary",
        "--matrixflow_binary",
        dest="matrixflow_binary",
        default="/root/trigger_gemm",
        help="Preferred guest path to trigger_gemm.",
    )


def build_matrixflow_command(
    manual,
    script_path,
    workload,
    matrix_size,
    trigger_binary,
    allow_guest_fallback=True,
):
    if manual:
        return "sleep 999999"

    if workload == "script" and os.path.exists(script_path):
        return "m5 exit;" + open(script_path).read()

    if workload == "script":
        return (
            "m5 exit;"
            + "numactl -H;"
            + "m5 resetstats;"
            + "numactl -N 0 -m 1 /home/test_code/simple_test;"
        )

    return (
        "m5 exit;"
        + 'TMP_TRIGGER=/tmp/trigger_gemm.download;'
        + 'TRIGGER="";'
        + 'rm -f "$TMP_TRIGGER";'
        + 'if wget -T 3 -t 1 -q http://10.0.2.2:8000/trigger_gemm '
          '-O "$TMP_TRIGGER" 2>/dev/null || '
          'wget -T 3 -t 1 http://10.0.2.2:8000/trigger_gemm '
          '-O "$TMP_TRIGGER" 2>/dev/null; then '
          'if [ -s "$TMP_TRIGGER" ]; then '
          'chmod +x "$TMP_TRIGGER"; '
          'TRIGGER="$TMP_TRIGGER"; '
          'fi; '
          'fi;'
        + (
            f'[ -x "$TRIGGER" ] || TRIGGER="{trigger_binary}";'
            '[ -x "$TRIGGER" ] || '
            '[ ! -x /home/test_code/trigger_gemm ] || '
            'TRIGGER=/home/test_code/trigger_gemm;'
            '[ -x "$TRIGGER" ] || [ ! -x ./trigger_gemm ] || '
            'TRIGGER=./trigger_gemm;'
            if allow_guest_fallback
            else ""
        )
        + 'if [ -x "$TRIGGER" ]; then '
          'echo "Using trigger binary: $TRIGGER"; '
        + f'"$TRIGGER" 0x200000000 {matrix_size} {workload};'
        + 'else '
          'echo "trigger_gemm download failed and no local fallback is allowed."; '
          'fi;'
        + 'rm -f "$TMP_TRIGGER";'
        + 'm5 exit'
    )
