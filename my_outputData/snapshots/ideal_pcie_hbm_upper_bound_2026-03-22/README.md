# Idealized PCIe-HBM Upper-Bound Snapshot

Snapshot date: `2026-03-22`

This snapshot freezes the current PCIe-HBM upper-bound baseline used for
comparison against the CXL-PNM path.

Frozen assumptions:

- Topology: PCIe + device-side HBM baseline
- Host-device link: `64 GB/s`
- Packetization model:
  - `optimal_pkt_size = 256`
  - `small_pkt_size = 64`
  - `small_pkt_overhead_pct = 0`
  - `large_pkt_size = 4096`
  - `large_pkt_overhead_pct = 36`
- Phase 2 path: explicit `devm-copy`
- Host-side Non-GEMM proxy enabled
- Workload type: ViT-like / transformer-core proxy
- Not a full-model ViT runtime
- MatrixFlow array: `16x16`
- Clock: `2.4 GHz`
- Device memory: `8-channel Ideal_CXL_HBM2`

ViT-like presets:

- Base-like: `S=197, H=768, M=3072, heads=12`
- Large-like: `S=257, H=1024, M=4096, heads=16`
- Huge-like: `S=257, H=1280, M=5120, heads=16`

Included result files:

- `square_gemm_proxy_snapshot.csv`
- `roofline_data.csv`
- `summary_raw.txt`

Note:

- The current ViT-like CSV captures ROI / DMA / GEMM metrics correctly.
- Per-phase guest timing prints were not yet captured into the CSV at this
  snapshot point and therefore appear as zeros in the ViT-like table.
