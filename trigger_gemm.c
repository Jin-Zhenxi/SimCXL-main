#define _FILE_OFFSET_BITS 64
#include <fcntl.h>
#include <gem5/m5ops.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <x86intrin.h>

struct Descriptor {
    uint64_t addrA;
    uint64_t addrB;
    uint64_t addrC;
    uint64_t flag_addr;
    uint32_t size;
};

_Static_assert(sizeof(struct Descriptor) == 40,
               "Descriptor layout must match MatrixFlowEngine");

enum Phase2Mode {
    MODE_REMOTE_SCALAR = 0,
    MODE_STAGED_BLOCK = 1,
};

enum WorkloadMode {
    WORKLOAD_VIT_PROXY = 0,
    WORKLOAD_PURE_GEMM = 1,
};

struct VitProxyConfig {
    const char *preset_name;
    uint32_t seq_len;
    uint32_t hidden_dim;
    uint32_t mlp_dim;
    uint32_t num_heads;
    const char *phase2_mode_name;
    uint32_t staged_block_bytes;
};

struct StageIoStats {
    uint64_t read_bytes;
    uint64_t write_bytes;
    uint64_t read_accesses;
    uint64_t write_accesses;
};

static unsigned long long
align_up_ull(unsigned long long value, unsigned long long align)
{
    return align == 0 ? value : ((value + align - 1) / align) * align;
}

static unsigned long long
now_sim_ns(void)
{
    return (unsigned long long)m5_rpns();
}

static float
approx_gelu(float x)
{
    const float alpha = 0.7978845608f;
    const float beta = 0.044715f;
    float inner = alpha * (x + beta * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

static float
u32_to_f32(uint32_t value)
{
    return (float)value * 0.001f;
}

static uint32_t
f32_to_u32(float value)
{
    if (value < 0.0f) {
        value = 0.0f;
    }
    return (uint32_t)(value * 1024.0f);
}

static void
record_read(struct StageIoStats *stats, size_t bytes)
{
    stats->read_bytes += bytes;
    stats->read_accesses += 1;
}

static void
record_write(struct StageIoStats *stats, size_t bytes)
{
    stats->write_bytes += bytes;
    stats->write_accesses += 1;
}

static void
stage_read_u32_to_f32(const volatile uint32_t *src, size_t count, float *dst,
                      struct StageIoStats *stats)
{
    size_t i;
    record_read(stats, count * sizeof(uint32_t));
    for (i = 0; i < count; ++i) {
        dst[i] = u32_to_f32(src[i]);
    }
}

static void
stage_write_f32_to_u32(volatile uint32_t *dst, size_t count, const float *src,
                       struct StageIoStats *stats)
{
    size_t i;
    record_write(stats, count * sizeof(uint32_t));
    for (i = 0; i < count; ++i) {
        dst[i] = f32_to_u32(src[i]);
    }
}

static void
softmax_row(float *row, uint32_t cols)
{
    uint32_t c;
    float max_val = row[0];
    float sum = 0.0f;

    for (c = 1; c < cols; ++c) {
        if (row[c] > max_val) {
            max_val = row[c];
        }
    }

    for (c = 0; c < cols; ++c) {
        row[c] = expf(row[c] - max_val);
        sum += row[c];
    }

    if (sum > 0.0f) {
        float inv_sum = 1.0f / sum;
        for (c = 0; c < cols; ++c) {
            row[c] *= inv_sum;
        }
    }
}

static void
layernorm_row(float *row, uint32_t cols)
{
    uint32_t c;
    float mean = 0.0f;
    float var = 0.0f;

    for (c = 0; c < cols; ++c) {
        mean += row[c];
    }
    mean /= (float)cols;

    for (c = 0; c < cols; ++c) {
        float delta = row[c] - mean;
        var += delta * delta;
    }
    var /= (float)cols;

    {
        float inv_std = 1.0f / sqrtf(var + 1e-5f);
        for (c = 0; c < cols; ++c) {
            row[c] = (row[c] - mean) * inv_std;
        }
    }
}

static void
flush_remote_range(volatile uint32_t *base, size_t bytes)
{
    uintptr_t p = (uintptr_t)base;
    uintptr_t end = p + bytes;
    for (; p < end; p += 64) {
        _mm_clflush((const void *)p);
    }
    asm volatile("mfence" ::: "memory");
}

static uint64_t
calc_tile_count(uint32_t n, uint32_t tile_dim)
{
    uint64_t per_dim = (n + tile_dim - 1U) / tile_dim;
    return per_dim * per_dim * per_dim;
}

static enum WorkloadMode
parse_workload_mode(const char *name)
{
    if (name != NULL && strcmp(name, "pure_gemm") == 0) {
        return WORKLOAD_PURE_GEMM;
    }
    return WORKLOAD_VIT_PROXY;
}

static void
phase2_softmax_staged(volatile uint32_t *score_buf, uint32_t seq_len,
                      float *row_buf, struct StageIoStats *stats)
{
    uint32_t r;
    for (r = 0; r < seq_len; ++r) {
        volatile uint32_t *row_src = score_buf + (size_t)r * seq_len;
        stage_read_u32_to_f32(row_src, seq_len, row_buf, stats);
        softmax_row(row_buf, seq_len);
        stage_write_f32_to_u32(row_src, seq_len, row_buf, stats);
    }
}

static void
phase2_expand_staged(volatile uint32_t *score_buf, volatile uint32_t *hidden_buf,
                     volatile uint32_t *residual_buf, uint32_t seq_len,
                     uint32_t hidden_dim, float *score_row, float *hidden_row,
                     struct StageIoStats *score_stats,
                     struct StageIoStats *hidden_stats,
                     struct StageIoStats *residual_stats)
{
    uint32_t r;
    for (r = 0; r < seq_len; ++r) {
        uint32_t c;
        volatile uint32_t *score_row_src = score_buf + (size_t)r * seq_len;
        volatile uint32_t *hidden_row_dst = hidden_buf + (size_t)r * hidden_dim;
        volatile uint32_t *residual_row_dst =
            residual_buf + (size_t)r * hidden_dim;

        stage_read_u32_to_f32(score_row_src, seq_len, score_row, score_stats);
        for (c = 0; c < hidden_dim; ++c) {
            hidden_row[c] = score_row[c % seq_len] +
                (float)((c % 17U) + 1U) * 0.0005f;
        }
        stage_write_f32_to_u32(hidden_row_dst, hidden_dim, hidden_row,
                               hidden_stats);
        stage_write_f32_to_u32(residual_row_dst, hidden_dim, hidden_row,
                               residual_stats);
    }
}

static void
phase2_layernorm_staged(volatile uint32_t *hidden_buf, uint32_t seq_len,
                        uint32_t hidden_dim, float *hidden_row,
                        struct StageIoStats *hidden_stats)
{
    uint32_t r;
    for (r = 0; r < seq_len; ++r) {
        volatile uint32_t *hidden_row_ptr =
            hidden_buf + (size_t)r * hidden_dim;
        stage_read_u32_to_f32(hidden_row_ptr, hidden_dim, hidden_row,
                              hidden_stats);
        layernorm_row(hidden_row, hidden_dim);
        stage_write_f32_to_u32(hidden_row_ptr, hidden_dim, hidden_row,
                               hidden_stats);
    }
}

static void
phase2_project_staged(volatile uint32_t *hidden_buf, volatile uint32_t *mlp_buf,
                      uint32_t seq_len, uint32_t hidden_dim, uint32_t mlp_dim,
                      uint32_t block_elems, float *hidden_row, float *mlp_block,
                      struct StageIoStats *hidden_stats,
                      struct StageIoStats *mlp_stats)
{
    uint32_t r;
    for (r = 0; r < seq_len; ++r) {
        uint32_t base;
        volatile uint32_t *hidden_row_ptr =
            hidden_buf + (size_t)r * hidden_dim;
        volatile uint32_t *mlp_row_ptr =
            mlp_buf + (size_t)r * mlp_dim;

        stage_read_u32_to_f32(hidden_row_ptr, hidden_dim, hidden_row,
                              hidden_stats);
        for (base = 0; base < mlp_dim; base += block_elems) {
            uint32_t c;
            uint32_t chunk = block_elems;
            if (base + chunk > mlp_dim) {
                chunk = mlp_dim - base;
            }
            for (c = 0; c < chunk; ++c) {
                uint32_t idx = base + c;
                float x = hidden_row[idx % hidden_dim];
                float y = hidden_row[(idx * 7U + 3U) % hidden_dim];
                mlp_block[c] = 0.7f * x + 0.3f * y;
            }
            stage_write_f32_to_u32(mlp_row_ptr + base, chunk, mlp_block,
                                   mlp_stats);
        }
    }
}

static void
phase2_gelu_staged(volatile uint32_t *mlp_buf, uint32_t seq_len,
                   uint32_t mlp_dim, uint32_t block_elems, float *mlp_block,
                   struct StageIoStats *mlp_stats)
{
    uint32_t r;
    for (r = 0; r < seq_len; ++r) {
        uint32_t base;
        volatile uint32_t *mlp_row_ptr = mlp_buf + (size_t)r * mlp_dim;
        for (base = 0; base < mlp_dim; base += block_elems) {
            uint32_t c;
            uint32_t chunk = block_elems;
            if (base + chunk > mlp_dim) {
                chunk = mlp_dim - base;
            }
            stage_read_u32_to_f32(mlp_row_ptr + base, chunk, mlp_block,
                                  mlp_stats);
            for (c = 0; c < chunk; ++c) {
                mlp_block[c] = approx_gelu(mlp_block[c]);
            }
            stage_write_f32_to_u32(mlp_row_ptr + base, chunk, mlp_block,
                                   mlp_stats);
        }
    }
}

static void
phase2_residual_staged(volatile uint32_t *hidden_buf,
                       volatile uint32_t *residual_buf, uint32_t seq_len,
                       uint32_t hidden_dim, uint32_t block_elems,
                       float *hidden_block, float *residual_block,
                       struct StageIoStats *hidden_stats,
                       struct StageIoStats *residual_stats)
{
    uint32_t r;
    for (r = 0; r < seq_len; ++r) {
        uint32_t base;
        volatile uint32_t *hidden_row_ptr =
            hidden_buf + (size_t)r * hidden_dim;
        volatile uint32_t *residual_row_ptr =
            residual_buf + (size_t)r * hidden_dim;
        for (base = 0; base < hidden_dim; base += block_elems) {
            uint32_t c;
            uint32_t chunk = block_elems;
            if (base + chunk > hidden_dim) {
                chunk = hidden_dim - base;
            }
            stage_read_u32_to_f32(hidden_row_ptr + base, chunk, hidden_block,
                                  hidden_stats);
            stage_read_u32_to_f32(residual_row_ptr + base, chunk, residual_block,
                                  residual_stats);
            for (c = 0; c < chunk; ++c) {
                hidden_block[c] += residual_block[c];
            }
            stage_write_f32_to_u32(hidden_row_ptr + base, chunk, hidden_block,
                                   hidden_stats);
        }
    }
}

static void
phase2_compact_staged(volatile uint32_t *hidden_buf, volatile uint32_t *score_buf,
                      uint32_t seq_len, uint32_t hidden_dim, float *hidden_row,
                      float *score_row, struct StageIoStats *hidden_stats,
                      struct StageIoStats *score_stats)
{
    uint32_t r;
    for (r = 0; r < seq_len; ++r) {
        uint32_t c;
        volatile uint32_t *hidden_row_ptr =
            hidden_buf + (size_t)r * hidden_dim;
        volatile uint32_t *score_row_ptr = score_buf + (size_t)r * seq_len;

        stage_read_u32_to_f32(hidden_row_ptr, hidden_dim, hidden_row,
                              hidden_stats);
        for (c = 0; c < seq_len; ++c) {
            score_row[c] = hidden_row[c % hidden_dim];
        }
        stage_write_f32_to_u32(score_row_ptr, seq_len, score_row, score_stats);
    }
}

static void
phase2_softmax_scalar(volatile uint32_t *score_buf, uint32_t seq_len,
                      float *row_buf, struct StageIoStats *score_stats)
{
    uint32_t r, c;
    for (r = 0; r < seq_len; ++r) {
        for (c = 0; c < seq_len; ++c) {
            score_stats->read_bytes += sizeof(uint32_t);
            score_stats->read_accesses += 1;
            row_buf[c] = u32_to_f32(score_buf[(size_t)r * seq_len + c]);
        }
        softmax_row(row_buf, seq_len);
        for (c = 0; c < seq_len; ++c) {
            score_stats->write_bytes += sizeof(uint32_t);
            score_stats->write_accesses += 1;
            score_buf[(size_t)r * seq_len + c] = f32_to_u32(row_buf[c]);
        }
    }
}

static void
phase2_expand_scalar(volatile uint32_t *score_buf, volatile uint32_t *hidden_buf,
                     volatile uint32_t *residual_buf, uint32_t seq_len,
                     uint32_t hidden_dim, struct StageIoStats *score_stats,
                     struct StageIoStats *hidden_stats,
                     struct StageIoStats *residual_stats)
{
    uint32_t r, c;
    for (r = 0; r < seq_len; ++r) {
        for (c = 0; c < hidden_dim; ++c) {
            float base;
            float value;
            size_t score_idx = (size_t)r * seq_len + (c % seq_len);
            size_t hidden_idx = (size_t)r * hidden_dim + c;
            score_stats->read_bytes += sizeof(uint32_t);
            score_stats->read_accesses += 1;
            base = u32_to_f32(score_buf[score_idx]);
            value = base + (float)((c % 17U) + 1U) * 0.0005f;
            hidden_stats->write_bytes += sizeof(uint32_t);
            hidden_stats->write_accesses += 1;
            residual_stats->write_bytes += sizeof(uint32_t);
            residual_stats->write_accesses += 1;
            hidden_buf[hidden_idx] = f32_to_u32(value);
            residual_buf[hidden_idx] = f32_to_u32(value);
        }
    }
}

static void
phase2_layernorm_scalar(volatile uint32_t *hidden_buf, uint32_t seq_len,
                        uint32_t hidden_dim, float *hidden_row,
                        struct StageIoStats *hidden_stats)
{
    uint32_t r, c;
    for (r = 0; r < seq_len; ++r) {
        for (c = 0; c < hidden_dim; ++c) {
            size_t idx = (size_t)r * hidden_dim + c;
            hidden_stats->read_bytes += sizeof(uint32_t);
            hidden_stats->read_accesses += 1;
            hidden_row[c] = u32_to_f32(hidden_buf[idx]);
        }
        layernorm_row(hidden_row, hidden_dim);
        for (c = 0; c < hidden_dim; ++c) {
            size_t idx = (size_t)r * hidden_dim + c;
            hidden_stats->write_bytes += sizeof(uint32_t);
            hidden_stats->write_accesses += 1;
            hidden_buf[idx] = f32_to_u32(hidden_row[c]);
        }
    }
}

static void
phase2_project_scalar(volatile uint32_t *hidden_buf, volatile uint32_t *mlp_buf,
                      uint32_t seq_len, uint32_t hidden_dim, uint32_t mlp_dim,
                      struct StageIoStats *hidden_stats,
                      struct StageIoStats *mlp_stats)
{
    uint32_t r, c;
    for (r = 0; r < seq_len; ++r) {
        for (c = 0; c < mlp_dim; ++c) {
            size_t src0 = (size_t)r * hidden_dim + (c % hidden_dim);
            size_t src1 = (size_t)r * hidden_dim + ((c * 7U + 3U) % hidden_dim);
            size_t dst = (size_t)r * mlp_dim + c;
            float x;
            float y;
            hidden_stats->read_bytes += 2 * sizeof(uint32_t);
            hidden_stats->read_accesses += 2;
            x = u32_to_f32(hidden_buf[src0]);
            y = u32_to_f32(hidden_buf[src1]);
            mlp_stats->write_bytes += sizeof(uint32_t);
            mlp_stats->write_accesses += 1;
            mlp_buf[dst] = f32_to_u32(0.7f * x + 0.3f * y);
        }
    }
}

static void
phase2_gelu_scalar(volatile uint32_t *mlp_buf, uint32_t seq_len,
                   uint32_t mlp_dim, struct StageIoStats *mlp_stats)
{
    uint32_t r, c;
    for (r = 0; r < seq_len; ++r) {
        for (c = 0; c < mlp_dim; ++c) {
            size_t idx = (size_t)r * mlp_dim + c;
            float x;
            mlp_stats->read_bytes += sizeof(uint32_t);
            mlp_stats->read_accesses += 1;
            x = u32_to_f32(mlp_buf[idx]);
            mlp_stats->write_bytes += sizeof(uint32_t);
            mlp_stats->write_accesses += 1;
            mlp_buf[idx] = f32_to_u32(approx_gelu(x));
        }
    }
}

static void
phase2_residual_scalar(volatile uint32_t *hidden_buf,
                       volatile uint32_t *residual_buf, uint32_t seq_len,
                       uint32_t hidden_dim, struct StageIoStats *hidden_stats,
                       struct StageIoStats *residual_stats)
{
    uint32_t r, c;
    for (r = 0; r < seq_len; ++r) {
        for (c = 0; c < hidden_dim; ++c) {
            size_t idx = (size_t)r * hidden_dim + c;
            float h;
            float res;
            hidden_stats->read_bytes += sizeof(uint32_t);
            hidden_stats->read_accesses += 1;
            residual_stats->read_bytes += sizeof(uint32_t);
            residual_stats->read_accesses += 1;
            h = u32_to_f32(hidden_buf[idx]);
            res = u32_to_f32(residual_buf[idx]);
            hidden_stats->write_bytes += sizeof(uint32_t);
            hidden_stats->write_accesses += 1;
            hidden_buf[idx] = f32_to_u32(h + res);
        }
    }
}

static void
phase2_compact_scalar(volatile uint32_t *hidden_buf, volatile uint32_t *score_buf,
                      uint32_t seq_len, uint32_t hidden_dim,
                      struct StageIoStats *hidden_stats,
                      struct StageIoStats *score_stats)
{
    uint32_t r, c;
    for (r = 0; r < seq_len; ++r) {
        for (c = 0; c < seq_len; ++c) {
            size_t hidden_idx = (size_t)r * hidden_dim + (c % hidden_dim);
            size_t score_idx = (size_t)r * seq_len + c;
            float value;
            hidden_stats->read_bytes += sizeof(uint32_t);
            hidden_stats->read_accesses += 1;
            value = u32_to_f32(hidden_buf[hidden_idx]);
            score_stats->write_bytes += sizeof(uint32_t);
            score_stats->write_accesses += 1;
            score_buf[score_idx] = f32_to_u32(value);
        }
    }
}

int
main(int argc, char *argv[])
{
    unsigned long long cxl_base = 0x200000000ULL;
    unsigned long long hdm_base = 0x400000000ULL;
    uint32_t default_seq_len = 257;
    uint32_t default_hidden_dim = 1024;
    uint32_t default_mlp_dim = 4096;
    uint32_t default_num_heads = 16;
    const char *default_phase2_mode = "staged_block";
    const char *default_workload = "vit_proxy";
    uint32_t default_staged_block_bytes = 1024;

    if (argc >= 2) {
        cxl_base = strtoull(argv[1], NULL, 0);
    }
    if (argc >= 3) {
        default_seq_len = (uint32_t)strtoul(argv[2], NULL, 0);
    }
    if (argc >= 4) {
        default_workload = argv[3];
    }

    {
        struct VitProxyConfig cfg = {
            .preset_name = "MatrixFlow-Benchmark",
            .seq_len = default_seq_len,
            .hidden_dim = default_hidden_dim,
            .mlp_dim = default_mlp_dim,
            .num_heads = default_num_heads,
            .phase2_mode_name = default_phase2_mode,
            .staged_block_bytes = default_staged_block_bytes,
        };
        enum WorkloadMode workload_mode = parse_workload_mode(default_workload);
        enum Phase2Mode phase2_mode =
            strcmp(cfg.phase2_mode_name, "remote_scalar") == 0
                ? MODE_REMOTE_SCALAR
                : MODE_STAGED_BLOCK;
        const int run_phase2 = workload_mode == WORKLOAD_VIT_PROXY;
        const int run_phase3 = workload_mode == WORKLOAD_VIT_PROXY;
        const char *workload_name = run_phase2 ? "vit_proxy" : "pure_gemm";
        const uint32_t descriptor_launch_count = run_phase3 ? 2U : 1U;

        const uint32_t matrix_size = cfg.seq_len;
        const unsigned long long doorbell_pa = cxl_base + 0x10000ULL;
        const unsigned long long desc_pa = hdm_base;
        const unsigned long long flag_pa = hdm_base + 0x80ULL;
        const unsigned long long matrix_span = align_up_ull(
            (unsigned long long)matrix_size * matrix_size * sizeof(uint32_t), 64ULL
        );
        const unsigned long long matrix_base =
            align_up_ull(hdm_base + 0x1000ULL, 64ULL);
        const size_t score_elems = (size_t)cfg.seq_len * (size_t)cfg.seq_len;
        const size_t score_bytes = score_elems * sizeof(uint32_t);
        const size_t hidden_elems = run_phase2
            ? (size_t)cfg.seq_len * (size_t)cfg.hidden_dim
            : 0;
        const size_t hidden_bytes = hidden_elems * sizeof(uint32_t);
        const size_t hidden_span = (size_t)align_up_ull(hidden_bytes, 64ULL);
        const size_t mlp_elems = run_phase2
            ? (size_t)cfg.seq_len * (size_t)cfg.mlp_dim
            : 0;
        const size_t mlp_bytes = mlp_elems * sizeof(uint32_t);
        const size_t mlp_span = (size_t)align_up_ull(mlp_bytes, 64ULL);
        const size_t total_data_span = run_phase2
            ? (size_t)(matrix_span * 3ULL) + hidden_span + hidden_span +
                mlp_span
            : (size_t)(matrix_span * 3ULL);
        const size_t map_size = 4096;
        const uint32_t block_elems =
            cfg.staged_block_bytes / sizeof(uint32_t) > 0
                ? cfg.staged_block_bytes / sizeof(uint32_t)
                : 1;
        const uint64_t tile_count_per_gemm = calc_tile_count(cfg.seq_len, 128);
        const uint64_t total_gemm_count = run_phase3 ? 2ULL : 1ULL;
        const uint64_t total_tile_count = tile_count_per_gemm * total_gemm_count;

        struct Descriptor desc = {
            .addrA = matrix_base,
            .addrB = matrix_base + matrix_span,
            .addrC = matrix_base + matrix_span * 2ULL,
            .flag_addr = flag_pa,
            .size = matrix_size,
        };

        int fd = -1;
        void *desc_map = MAP_FAILED;
        void *doorbell_map = MAP_FAILED;
        void *data_map = MAP_FAILED;
        volatile struct Descriptor *desc_ptr;
        volatile uint64_t *flag_ptr;
        volatile uint64_t *doorbell_ptr;
        volatile uint32_t *matrix_A_cpu_ptr;
        volatile uint32_t *matrix_B_cpu_ptr;
        volatile uint32_t *score_buf;
        volatile uint32_t *hidden_buf;
        volatile uint32_t *residual_buf;
        volatile uint32_t *mlp_buf;
        float *score_row = NULL;
        float *hidden_row = NULL;
        float *hidden_block = NULL;
        float *residual_block = NULL;
        float *mlp_block = NULL;
        struct StageIoStats score_stats = {0};
        struct StageIoStats hidden_stats = {0};
        struct StageIoStats residual_stats = {0};
        struct StageIoStats mlp_stats = {0};
        unsigned long long phase1_begin;
        unsigned long long phase1_end;
        unsigned long long phase2_begin;
        unsigned long long expand_begin;
        unsigned long long expand_end;
        unsigned long long softmax_begin;
        unsigned long long softmax_end;
        unsigned long long layernorm_begin;
        unsigned long long layernorm_end;
        unsigned long long project_begin;
        unsigned long long project_end;
        unsigned long long gelu_begin;
        unsigned long long gelu_end;
        unsigned long long residual_begin;
        unsigned long long residual_end;
        unsigned long long compact_begin;
        unsigned long long compact_end;
        unsigned long long phase2_end;
        unsigned long long phase3_begin;
        unsigned long long phase3_end;
        unsigned long long total_end;
        unsigned long long poll_count_phase1 = 0;
        unsigned long long poll_count_phase3 = 0;
        double phase2_non_gemm_ms;
        double phase2_inplace_ms;
        double phase1_ms;
        double phase3_ms;

        setbuf(stdout, NULL);
        printf("========== MatrixFlow Benchmark ==========\n");
        printf("[Config] workload=%s seq_len=%u hidden_dim=%u mlp_dim=%u num_heads=%u\n",
               workload_name, cfg.seq_len, cfg.hidden_dim, cfg.mlp_dim,
               cfg.num_heads);
        printf("[Config] matrix_size=%u x %u, CXL Type-3 + device-side DDR5 HDM\n",
               matrix_size, matrix_size);
        printf("[Config] phase2_mode=%s staged_block_bytes=%u\n",
               cfg.phase2_mode_name, cfg.staged_block_bytes);

        fd = open("/dev/mem", O_RDWR | O_SYNC);
        if (fd < 0) {
            perror("open /dev/mem");
            return -1;
        }

        desc_map = mmap(NULL, map_size, PROT_READ | PROT_WRITE,
                        MAP_SHARED, fd, (off_t)desc_pa);
        if (desc_map == MAP_FAILED) {
            perror("mmap desc");
            close(fd);
            return -1;
        }

        doorbell_map = mmap(NULL, map_size, PROT_READ | PROT_WRITE,
                            MAP_SHARED, fd, (off_t)doorbell_pa);
        if (doorbell_map == MAP_FAILED) {
            perror("mmap doorbell");
            munmap(desc_map, map_size);
            close(fd);
            return -1;
        }

        data_map = mmap(NULL, total_data_span, PROT_READ | PROT_WRITE,
                        MAP_SHARED, fd, (off_t)matrix_base);
        if (data_map == MAP_FAILED) {
            perror("mmap data window");
            munmap(doorbell_map, map_size);
            munmap(desc_map, map_size);
            close(fd);
            return -1;
        }

        desc_ptr = (volatile struct Descriptor *)desc_map;
        flag_ptr = (volatile uint64_t *)((char *)desc_map + (flag_pa - desc_pa));
        doorbell_ptr = (volatile uint64_t *)doorbell_map;
        matrix_A_cpu_ptr = (volatile uint32_t *)data_map;
        matrix_B_cpu_ptr = (volatile uint32_t *)((char *)data_map + matrix_span);
        score_buf = (volatile uint32_t *)((char *)data_map + matrix_span * 2ULL);
        hidden_buf = (volatile uint32_t *)((char *)data_map + matrix_span * 3ULL);
        residual_buf = (volatile uint32_t *)((char *)hidden_buf + hidden_span);
        mlp_buf = (volatile uint32_t *)((char *)residual_buf + hidden_span);

        if (run_phase2) {
            if (posix_memalign((void **)&score_row, 64,
                               cfg.seq_len * sizeof(float)) != 0 ||
                posix_memalign((void **)&hidden_row, 64,
                               cfg.hidden_dim * sizeof(float)) != 0 ||
                posix_memalign((void **)&hidden_block, 64,
                               block_elems * sizeof(float)) != 0 ||
                posix_memalign((void **)&residual_block, 64,
                               block_elems * sizeof(float)) != 0 ||
                posix_memalign((void **)&mlp_block, 64,
                               block_elems * sizeof(float)) != 0) {
                perror("posix_memalign scratch buffers");
                free(mlp_block);
                free(residual_block);
                free(hidden_block);
                free(hidden_row);
                free(score_row);
                munmap(data_map, total_data_span);
                munmap(doorbell_map, map_size);
                munmap(desc_map, map_size);
                close(fd);
                return -1;
            }
        }

        {
            size_t i;
            for (i = 0; i < score_elems; ++i) {
                matrix_A_cpu_ptr[i] = (uint32_t)((i * 3U + 1U) & 0xffU);
                matrix_B_cpu_ptr[i] = (uint32_t)((i * 5U + 7U) & 0xffU);
                score_buf[i] = 0;
            }
            for (i = 0; i < hidden_elems; ++i) {
                hidden_buf[i] = 0;
                residual_buf[i] = 0;
            }
            for (i = 0; i < mlp_elems; ++i) {
                mlp_buf[i] = 0;
            }
            flush_remote_range(matrix_A_cpu_ptr, total_data_span);
        }

        m5_reset_stats(0, 0);

        phase1_begin = now_sim_ns();
        printf("[Phase 1] GEMM1 on device HDM...\n");
        *flag_ptr = 0;
        __builtin_memcpy((void *)desc_ptr, &desc, sizeof(desc));
        _mm_clflush((const void *)desc_ptr);
        _mm_clflush((const void *)flag_ptr);
        asm volatile("mfence" ::: "memory");
        *doorbell_ptr = desc_pa;
        asm volatile("mfence" ::: "memory");
        while (*flag_ptr == 0) {
            poll_count_phase1 += 1;
            asm volatile("pause");
        }
        phase1_end = now_sim_ns();
        printf("[Phase 1] done\n");

        phase2_begin = phase1_end;
        softmax_begin = phase1_end;
        softmax_end = phase1_end;
        expand_begin = phase1_end;
        expand_end = phase1_end;
        layernorm_begin = phase1_end;
        layernorm_end = phase1_end;
        project_begin = phase1_end;
        project_end = phase1_end;
        gelu_begin = phase1_end;
        gelu_end = phase1_end;
        residual_begin = phase1_end;
        residual_end = phase1_end;
        compact_begin = phase1_end;
        compact_end = phase1_end;
        phase2_end = phase1_end;

        if (run_phase2) {
            phase2_begin = now_sim_ns();
            printf("[Phase 2] mode=%s over CXL HDM...\n", cfg.phase2_mode_name);

            softmax_begin = now_sim_ns();
            if (phase2_mode == MODE_STAGED_BLOCK) {
                phase2_softmax_staged(score_buf, cfg.seq_len, score_row,
                                      &score_stats);
            } else {
                phase2_softmax_scalar(score_buf, cfg.seq_len, score_row,
                                      &score_stats);
            }
            softmax_end = now_sim_ns();

            expand_begin = now_sim_ns();
            if (phase2_mode == MODE_STAGED_BLOCK) {
                phase2_expand_staged(score_buf, hidden_buf, residual_buf,
                                     cfg.seq_len, cfg.hidden_dim, score_row,
                                     hidden_row, &score_stats, &hidden_stats,
                                     &residual_stats);
            } else {
                phase2_expand_scalar(score_buf, hidden_buf, residual_buf,
                                     cfg.seq_len, cfg.hidden_dim, &score_stats,
                                     &hidden_stats, &residual_stats);
            }
            expand_end = now_sim_ns();

            layernorm_begin = now_sim_ns();
            if (phase2_mode == MODE_STAGED_BLOCK) {
                phase2_layernorm_staged(hidden_buf, cfg.seq_len,
                                        cfg.hidden_dim, hidden_row,
                                        &hidden_stats);
            } else {
                phase2_layernorm_scalar(hidden_buf, cfg.seq_len,
                                        cfg.hidden_dim, hidden_row,
                                        &hidden_stats);
            }
            layernorm_end = now_sim_ns();

            project_begin = now_sim_ns();
            if (phase2_mode == MODE_STAGED_BLOCK) {
                phase2_project_staged(hidden_buf, mlp_buf, cfg.seq_len,
                                      cfg.hidden_dim, cfg.mlp_dim, block_elems,
                                      hidden_row, mlp_block, &hidden_stats,
                                      &mlp_stats);
            } else {
                phase2_project_scalar(hidden_buf, mlp_buf, cfg.seq_len,
                                      cfg.hidden_dim, cfg.mlp_dim,
                                      &hidden_stats, &mlp_stats);
            }
            project_end = now_sim_ns();

            gelu_begin = now_sim_ns();
            if (phase2_mode == MODE_STAGED_BLOCK) {
                phase2_gelu_staged(mlp_buf, cfg.seq_len, cfg.mlp_dim,
                                   block_elems, mlp_block, &mlp_stats);
            } else {
                phase2_gelu_scalar(mlp_buf, cfg.seq_len, cfg.mlp_dim,
                                   &mlp_stats);
            }
            gelu_end = now_sim_ns();

            residual_begin = now_sim_ns();
            if (phase2_mode == MODE_STAGED_BLOCK) {
                phase2_residual_staged(hidden_buf, residual_buf, cfg.seq_len,
                                       cfg.hidden_dim, block_elems,
                                       hidden_block, residual_block,
                                       &hidden_stats, &residual_stats);
            } else {
                phase2_residual_scalar(hidden_buf, residual_buf, cfg.seq_len,
                                       cfg.hidden_dim, &hidden_stats,
                                       &residual_stats);
            }
            residual_end = now_sim_ns();

            compact_begin = now_sim_ns();
            if (phase2_mode == MODE_STAGED_BLOCK) {
                phase2_compact_staged(hidden_buf, score_buf, cfg.seq_len,
                                      cfg.hidden_dim, hidden_row, score_row,
                                      &hidden_stats, &score_stats);
            } else {
                phase2_compact_scalar(hidden_buf, score_buf, cfg.seq_len,
                                      cfg.hidden_dim, &hidden_stats,
                                      &score_stats);
            }
            compact_end = now_sim_ns();

            flush_remote_range(score_buf, score_bytes);
            phase2_end = now_sim_ns();
            printf("[Phase 2] done\n");
        } else {
            printf("[Phase 2] skipped for pure_gemm\n");
        }

        phase3_begin = phase2_end;
        phase3_end = phase2_end;

        if (run_phase3) {
            phase3_begin = now_sim_ns();
            printf("[Phase 3] GEMM2 on device HDM...\n");
            desc.addrA = matrix_base + matrix_span * 2ULL;
            desc.addrC = matrix_base;
            *flag_ptr = 0;
            __builtin_memcpy((void *)desc_ptr, &desc, sizeof(desc));
            _mm_clflush((const void *)desc_ptr);
            _mm_clflush((const void *)flag_ptr);
            asm volatile("mfence" ::: "memory");
            *doorbell_ptr = desc_pa;
            asm volatile("mfence" ::: "memory");
            while (*flag_ptr == 0) {
                poll_count_phase3 += 1;
                asm volatile("pause");
            }
            phase3_end = now_sim_ns();
            printf("[Phase 3] done\n");
        } else {
            printf("[Phase 3] skipped for pure_gemm\n");
        }

        total_end = now_sim_ns();
        m5_dump_stats(0, 0);

        phase2_non_gemm_ms = (softmax_end - softmax_begin +
                              layernorm_end - layernorm_begin +
                              gelu_end - gelu_begin +
                              residual_end - residual_begin) / 1.0e6;
        phase2_inplace_ms = (phase2_end - phase2_begin) / 1.0e6;
        phase1_ms = (phase1_end - phase1_begin) / 1.0e6;
        phase3_ms = (phase3_end - phase3_begin) / 1.0e6;

        printf("[Timing] preset=%s\n", cfg.preset_name);
        printf("[Timing] workload=%s\n", workload_name);
        printf("[Timing] phase2_mode=%s\n", cfg.phase2_mode_name);
        printf("[Timing] staged_block_bytes=%u\n", cfg.staged_block_bytes);
        printf("[Timing] gemm_tile_count_per_gemm=%llu\n",
               (unsigned long long)tile_count_per_gemm);
        printf("[Timing] gemm_tile_count_total=%llu\n",
               (unsigned long long)total_tile_count);
        printf("[Timing] descriptor_launch_count=%u\n", descriptor_launch_count);
        printf("[Timing] doorbell_launch_count=%u\n", descriptor_launch_count);
        printf("[Timing] phase1_poll_count=%llu\n", poll_count_phase1);
        printf("[Timing] phase3_poll_count=%llu\n", poll_count_phase3);
        printf("[Timing] phase1_ms=%.6f\n", phase1_ms);
        printf("[Timing] phase2_d2h_ms=0.000000\n");
        printf("[Timing] phase2_h2d_ms=0.000000\n");
        printf("[Timing] phase2_cxl_inplace_ms=%.6f\n", phase2_inplace_ms);
        printf("[Timing] phase2_expand_ms=%.6f\n",
               (expand_end - expand_begin) / 1.0e6);
        printf("[Timing] phase2_softmax_ms=%.6f\n",
               (softmax_end - softmax_begin) / 1.0e6);
        printf("[Timing] phase2_layernorm_ms=%.6f\n",
               (layernorm_end - layernorm_begin) / 1.0e6);
        printf("[Timing] phase2_project_ms=%.6f\n",
               (project_end - project_begin) / 1.0e6);
        printf("[Timing] phase2_gelu_ms=%.6f\n",
               (gelu_end - gelu_begin) / 1.0e6);
        printf("[Timing] phase2_residual_ms=%.6f\n",
               (residual_end - residual_begin) / 1.0e6);
        printf("[Timing] phase2_compact_ms=%.6f\n",
               (compact_end - compact_begin) / 1.0e6);
        printf("[Timing] phase2_non_gemm_ms=%.6f\n", phase2_non_gemm_ms);
        printf("[Timing] phase2_total_ms=%.6f\n", phase2_inplace_ms);
        printf("[Timing] phase3_ms=%.6f\n", phase3_ms);
        printf("[Timing] end_to_end_ms=%.6f\n",
               (total_end - phase1_begin) / 1.0e6);
        printf("[Timing] host_mediated_copy_bytes=0\n");
        printf("[Timing] score_read_bytes=%llu\n",
               (unsigned long long)score_stats.read_bytes);
        printf("[Timing] score_write_bytes=%llu\n",
               (unsigned long long)score_stats.write_bytes);
        printf("[Timing] score_read_accesses=%llu\n",
               (unsigned long long)score_stats.read_accesses);
        printf("[Timing] score_write_accesses=%llu\n",
               (unsigned long long)score_stats.write_accesses);
        printf("[Timing] hidden_read_bytes=%llu\n",
               (unsigned long long)hidden_stats.read_bytes);
        printf("[Timing] hidden_write_bytes=%llu\n",
               (unsigned long long)hidden_stats.write_bytes);
        printf("[Timing] hidden_read_accesses=%llu\n",
               (unsigned long long)hidden_stats.read_accesses);
        printf("[Timing] hidden_write_accesses=%llu\n",
               (unsigned long long)hidden_stats.write_accesses);
        printf("[Timing] residual_read_bytes=%llu\n",
               (unsigned long long)residual_stats.read_bytes);
        printf("[Timing] residual_write_bytes=%llu\n",
               (unsigned long long)residual_stats.write_bytes);
        printf("[Timing] residual_read_accesses=%llu\n",
               (unsigned long long)residual_stats.read_accesses);
        printf("[Timing] residual_write_accesses=%llu\n",
               (unsigned long long)residual_stats.write_accesses);
        printf("[Timing] mlp_read_bytes=%llu\n",
               (unsigned long long)mlp_stats.read_bytes);
        printf("[Timing] mlp_write_bytes=%llu\n",
               (unsigned long long)mlp_stats.write_bytes);
        printf("[Timing] mlp_read_accesses=%llu\n",
               (unsigned long long)mlp_stats.read_accesses);
        printf("[Timing] mlp_write_accesses=%llu\n",
               (unsigned long long)mlp_stats.write_accesses);
        printf("[Timing] cxl_inplace_access_bytes=%llu\n",
               (unsigned long long)(score_stats.read_bytes +
                                    score_stats.write_bytes +
                                    hidden_stats.read_bytes +
                                    hidden_stats.write_bytes +
                                    residual_stats.read_bytes +
                                    residual_stats.write_bytes +
                                    mlp_stats.read_bytes +
                                    mlp_stats.write_bytes));
        printf("[Timing] proxy_remote_footprint_bytes=%llu\n",
               (unsigned long long)(score_bytes + hidden_bytes +
                                    hidden_bytes + mlp_bytes));
        printf("[Timing] total_proxy_movement_bytes=%llu\n",
               (unsigned long long)(score_stats.read_bytes +
                                    score_stats.write_bytes +
                                    hidden_stats.read_bytes +
                                    hidden_stats.write_bytes +
                                    residual_stats.read_bytes +
                                    residual_stats.write_bytes +
                                    mlp_stats.read_bytes +
                                    mlp_stats.write_bytes));
        printf("========== MatrixFlow benchmark finished ==========\n");

        free(mlp_block);
        free(residual_block);
        free(hidden_block);
        free(hidden_row);
        free(score_row);
        munmap(data_map, total_data_span);
        munmap(doorbell_map, map_size);
        munmap(desc_map, map_size);
        close(fd);
    }

    return 0;
}
