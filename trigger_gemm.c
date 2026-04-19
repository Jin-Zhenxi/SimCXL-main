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
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t lda;
    uint32_t ldb;
    uint32_t ldc;
    uint32_t flags;
    uint64_t completion_value;
};

_Static_assert(sizeof(struct Descriptor) == 72,
               "Descriptor layout must match MatrixFlowEngine");

enum Phase2Mode {
    MODE_REMOTE_SCALAR = 0,
    MODE_STAGED_BLOCK = 1,
};

enum WorkloadMode {
    WORKLOAD_VIT_PROXY = 0,
    WORKLOAD_PURE_GEMM = 1,
};

enum PureGemmPrototypeMode {
    PURE_GEMM_PROTO_OFF = 0,
    PURE_GEMM_PROTO_TAIL_PACK_ALIGN16 = 1,
    PURE_GEMM_PROTO_PEELED_RECT_V1 = 2,
    PURE_GEMM_PROTO_PEELED_RECT_V2_RIGHT_EDGE_RECTIFIED = 3,
    PURE_GEMM_PROTO_PEELED_RECT_BATCHED_SINGLE_DOORBELL_FIRST_CUT = 4,
    PURE_GEMM_PROTO_IRREGULAR_GEMM_B_TAIL_SCRATCHPAD_OUTPUT_HOLD_FIRST_CUT = 5,
    PURE_GEMM_PROTO_IRREGULAR_GEMM_FUSED_EDGES_COMPLETION_OPTIMIZED_FIRST_CUT = 6,
    PURE_GEMM_PROTO_IRREGULAR_GEMM_FUSED_RIGHT_EDGE_CLEAN_TIMING_FIRST_CUT = 7,
    PURE_GEMM_PROTO_IRREGULAR_GEMM_NO_WAIT_FUSED_RIGHT_FIRST_CUT = 8,
    PURE_GEMM_PROTO_IRREGULAR_GEMM_NO_WAIT_FUSED_BOTTOM_FIRST_CUT = 9,
    PURE_GEMM_PROTO_IRREGULAR_GEMM_SINGLE_FUSED_DESCRIPTOR_CORNER_COLLAPSE_FIRST_CUT = 10,
    PURE_GEMM_PROTO_IRREGULAR_GEMM_FINAL_COMPLETION_CHAIN_AUTOPSY_FIRST_CUT = 11,
    PURE_GEMM_PROTO_IRREGULAR_GEMM_BOUNDARY_ONLY_HOLD_EARLY_BODY_WRITEBACK_FIRST_CUT = 12,
    PURE_GEMM_PROTO_IRREGULAR_GEMM_STATIC_OUTPUT_TILE_CLASSIFIER_BOUNDARY_HOLD_FIRST_CUT = 13,
    PURE_GEMM_PROTO_IRREGULAR_GEMM_BOUNDARY_WRITEBACK_COALESCING_FIRST_CUT = 14,
    PURE_GEMM_PROTO_IRREGULAR_GEMM_STREAMING_BODY_WRITEBACK_FIRST_CUT = 15,
    PURE_GEMM_PROTO_SMALL_FULL_RESIDENCY_PAD256 = 16,
    PURE_GEMM_PROTO_LOGICAL_ZERO_FILL_CLIPPED_EXECUTION = 17,
};

enum DescriptorFlags {
    DESC_FLAG_CHAIN_CONTINUE = 1u << 0,
    DESC_FLAG_SUPPRESS_COMPLETION = 1u << 1,
    DESC_FLAG_PEELED_SUBPROBLEM = 1u << 2,
    DESC_FLAG_IRREGULAR_B_TAIL_SCRATCHPAD_OUTPUT_HOLD = 1u << 3,
    DESC_FLAG_IRREGULAR_FUSED_EDGES_COMPLETION_OPTIMIZED = 1u << 4,
    DESC_FLAG_IRREGULAR_FUSED_RIGHT_EDGE_CLEAN_TIMING = 1u << 5,
    DESC_FLAG_IRREGULAR_NO_WAIT_FUSED_RIGHT = 1u << 6,
    DESC_FLAG_IRREGULAR_NO_WAIT_FUSED_BOTTOM = 1u << 7,
    DESC_FLAG_IRREGULAR_SINGLE_FUSED_DESCRIPTOR_CORNER_COLLAPSE = 1u << 8,
    DESC_FLAG_IRREGULAR_FINAL_COMPLETION_CHAIN_AUTOPSY = 1u << 9,
    DESC_FLAG_IRREGULAR_BOUNDARY_ONLY_HOLD_EARLY_BODY_WRITEBACK = 1u << 10,
    DESC_FLAG_IRREGULAR_STATIC_OUTPUT_TILE_CLASSIFIER_BOUNDARY_HOLD = 1u << 11,
    DESC_FLAG_IRREGULAR_BOUNDARY_WRITEBACK_COALESCING = 1u << 12,
    DESC_FLAG_IRREGULAR_STREAMING_BODY_WRITEBACK = 1u << 13,
    DESC_FLAG_IRREGULAR_SMALL_FULL_RESIDENCY_PAD256 = 1u << 14,
    DESC_FLAG_IRREGULAR_LOGICAL_ZERO_FILL_CLIPPED_EXECUTION = 1u << 15,
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

static uint32_t
align_up_u32(uint32_t value, uint32_t align)
{
    return align == 0 ? value : ((value + align - 1U) / align) * align;
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

static void
clflush_range(const void *ptr, size_t size)
{
    uintptr_t p = (uintptr_t)ptr;
    uintptr_t end = p + size;

    for (; p < end; p += 64) {
        _mm_clflush((const void *)p);
    }
}

static uint64_t
make_completion_token(uint32_t matrix_size, uint32_t phase_id)
{
    uint64_t token = now_sim_ns();
    token ^= 0xC0DE000000000000ULL;
    token ^= ((uint64_t)phase_id << 48);
    token ^= ((uint64_t)matrix_size << 16);
    token |= 1ULL;

    if (token == 0) {
        token = 0xC0DE000000000001ULL |
                ((uint64_t)phase_id << 48) |
                ((uint64_t)matrix_size << 16);
    }

    return token;
}

static uint64_t
make_completion_sentinel(uint64_t token)
{
    uint64_t sentinel = token ^ 0x5A5A5A5A5A5A5A5AULL;
    if (sentinel == token) {
        sentinel ^= 0x00FF00FF00FF00FFULL;
    }
    return sentinel;
}

static int
prime_completion_flag(volatile uint64_t *flag_ptr, uint64_t sentinel)
{
    *flag_ptr = sentinel;
    _mm_clflush((const void *)flag_ptr);
    asm volatile("mfence" ::: "memory");

    for (unsigned i = 0; i < 100000; ++i) {
        if (*flag_ptr == sentinel) {
            return 0;
        }
        asm volatile("pause");
    }

    fprintf(stderr,
            "[ERROR] completion flag prime failed: expected=%#llx observed=%#llx\n",
            (unsigned long long)sentinel,
            (unsigned long long)*flag_ptr);
    return -1;
}

static void
refresh_device_buffer(const volatile void *ptr, size_t size);

struct LaunchTimingMarkers {
    unsigned long long desc_setup_begin_ns;
    unsigned long long desc_setup_end_ns;
    unsigned long long doorbell_write_ns;
    unsigned long long token_observed_ns;
};

static int
wait_for_completion_token(volatile uint64_t *flag_ptr, uint64_t token,
                          unsigned long long *poll_count,
                          unsigned long long *backoff_count,
                          unsigned long long *token_observed_ns)
{
    const unsigned long long max_polls = 100000000ULL;
    unsigned int pause_budget = 1;

    while (1) {
        refresh_device_buffer((const void *)flag_ptr, sizeof(*flag_ptr));
        *poll_count += 1;
        if (*flag_ptr == token) {
            if (token_observed_ns != NULL) {
                *token_observed_ns = now_sim_ns();
            }
            return 0;
        }
        if (*poll_count >= max_polls) {
            fprintf(stderr,
                    "[ERROR] completion timeout: expected=%#llx observed=%#llx polls=%llu\n",
                    (unsigned long long)token,
                    (unsigned long long)*flag_ptr,
                    (unsigned long long)*poll_count);
            return -1;
        }
        for (unsigned int spin = 0; spin < pause_budget; ++spin) {
            asm volatile("pause");
        }
        if (pause_budget < 4096U) {
            pause_budget <<= 1;
            if (backoff_count != NULL) {
                *backoff_count += 1ULL;
            }
        }
    }
}

static void
fill_u32_buffer(volatile uint32_t *ptr, size_t elems, uint32_t value)
{
    size_t i;
    for (i = 0; i < elems; ++i) {
        ptr[i] = value;
    }
}

static void
refresh_device_buffer(const volatile void *ptr, size_t size)
{
    clflush_range((const void *)ptr, size);
    asm volatile("mfence" ::: "memory");
}

static int
validate_expected_result(const volatile uint32_t *matrix_C_cpu_ptr,
                         size_t matrix_elems, uint32_t expected_value)
{
    const size_t probe_count = matrix_elems < 16 ? matrix_elems : 16;
    size_t i;

    for (i = 0; i < probe_count; ++i) {
        if (matrix_C_cpu_ptr[i] != expected_value) {
            fprintf(stderr,
                    "[ERROR] invalid result detected: C[%zu]=%u expected=%u\n",
                    i, matrix_C_cpu_ptr[i], expected_value);
            return -1;
        }
    }

    return 0;
}

static int
wait_for_expected_result(const volatile uint32_t *matrix_C_cpu_ptr,
                         size_t matrix_elems, uint32_t expected_value,
                         unsigned long long *poll_count)
{
    const size_t probe_count = matrix_elems < 16 ? matrix_elems : 16;
    const unsigned long long max_polls = 100000000ULL;

    while (*poll_count < max_polls) {
        int all_match = 1;
        size_t i;

        refresh_device_buffer(matrix_C_cpu_ptr, probe_count * sizeof(uint32_t));
        for (i = 0; i < probe_count; ++i) {
            if (matrix_C_cpu_ptr[i] != expected_value) {
                all_match = 0;
                break;
            }
        }

        if (all_match) {
            return 0;
        }

        *poll_count += 1;
        asm volatile("pause");
    }

    fprintf(stderr,
            "[ERROR] result timeout: sampled C never reached expected=%u, polls=%llu "
            "first_word=%u\n",
            expected_value,
            (unsigned long long)*poll_count,
            matrix_C_cpu_ptr[0]);
    return -1;
}

static enum WorkloadMode
parse_workload_mode(const char *name)
{
    if (name != NULL && strcmp(name, "pure_gemm") == 0) {
        return WORKLOAD_PURE_GEMM;
    }
    return WORKLOAD_VIT_PROXY;
}

static enum PureGemmPrototypeMode
parse_pure_gemm_prototype_mode(const char *name)
{
    if (name != NULL && strcmp(name, "pack_tail_align16") == 0) {
        return PURE_GEMM_PROTO_TAIL_PACK_ALIGN16;
    }
    if (name != NULL && strcmp(name, "peeled_rect_v1") == 0) {
        return PURE_GEMM_PROTO_PEELED_RECT_V1;
    }
    if (name != NULL &&
        strcmp(name, "peeled_rect_v2_right_edge_rectified") == 0) {
        return PURE_GEMM_PROTO_PEELED_RECT_V2_RIGHT_EDGE_RECTIFIED;
    }
    if (name != NULL &&
        strcmp(name, "peeled_rect_batched_single_doorbell_first_cut") == 0) {
        return PURE_GEMM_PROTO_PEELED_RECT_BATCHED_SINGLE_DOORBELL_FIRST_CUT;
    }
    if (name != NULL &&
        strcmp(name,
               "irregular_gemm_b_tail_scratchpad_output_hold_first_cut") == 0) {
        return
            PURE_GEMM_PROTO_IRREGULAR_GEMM_B_TAIL_SCRATCHPAD_OUTPUT_HOLD_FIRST_CUT;
    }
    if (name != NULL &&
        strcmp(name,
               "irregular_gemm_fused_edges_completion_optimized_first_cut") == 0) {
        return
            PURE_GEMM_PROTO_IRREGULAR_GEMM_FUSED_EDGES_COMPLETION_OPTIMIZED_FIRST_CUT;
    }
    if (name != NULL &&
        strcmp(name,
               "irregular_gemm_fused_right_edge_clean_timing_first_cut") == 0) {
        return
            PURE_GEMM_PROTO_IRREGULAR_GEMM_FUSED_RIGHT_EDGE_CLEAN_TIMING_FIRST_CUT;
    }
    if (name != NULL &&
        strcmp(name, "irregular_gemm_no_wait_fused_right_first_cut") == 0) {
        return
            PURE_GEMM_PROTO_IRREGULAR_GEMM_NO_WAIT_FUSED_RIGHT_FIRST_CUT;
    }
    if (name != NULL &&
        strcmp(name, "irregular_gemm_no_wait_fused_bottom_first_cut") == 0) {
        return
            PURE_GEMM_PROTO_IRREGULAR_GEMM_NO_WAIT_FUSED_BOTTOM_FIRST_CUT;
    }
    if (name != NULL &&
        strcmp(name,
               "irregular_gemm_single_fused_descriptor_corner_collapse_first_cut") == 0) {
        return
            PURE_GEMM_PROTO_IRREGULAR_GEMM_SINGLE_FUSED_DESCRIPTOR_CORNER_COLLAPSE_FIRST_CUT;
    }
    if (name != NULL &&
        strcmp(name,
               "irregular_gemm_final_completion_chain_autopsy_first_cut") == 0) {
        return
            PURE_GEMM_PROTO_IRREGULAR_GEMM_FINAL_COMPLETION_CHAIN_AUTOPSY_FIRST_CUT;
    }
    if (name != NULL &&
        strcmp(name,
               "irregular_gemm_boundary_only_hold_early_body_writeback_first_cut") == 0) {
        return
            PURE_GEMM_PROTO_IRREGULAR_GEMM_BOUNDARY_ONLY_HOLD_EARLY_BODY_WRITEBACK_FIRST_CUT;
    }
    if (name != NULL &&
        strcmp(name,
               "irregular_gemm_static_output_tile_classifier_boundary_hold_first_cut") == 0) {
        return
            PURE_GEMM_PROTO_IRREGULAR_GEMM_STATIC_OUTPUT_TILE_CLASSIFIER_BOUNDARY_HOLD_FIRST_CUT;
    }
    if (name != NULL &&
        strcmp(name,
               "irregular_gemm_boundary_writeback_coalescing_first_cut") == 0) {
        return
            PURE_GEMM_PROTO_IRREGULAR_GEMM_BOUNDARY_WRITEBACK_COALESCING_FIRST_CUT;
    }
    if (name != NULL &&
        strcmp(name,
               "irregular_gemm_streaming_body_writeback_first_cut") == 0) {
        return
            PURE_GEMM_PROTO_IRREGULAR_GEMM_STREAMING_BODY_WRITEBACK_FIRST_CUT;
    }
    if (name != NULL &&
        strcmp(name,
               "irregular_gemm_small_full_residency_pad256_first_cut") == 0) {
        return
            PURE_GEMM_PROTO_SMALL_FULL_RESIDENCY_PAD256;
    }
    if (name != NULL &&
        strcmp(
            name,
            "irregular_gemm_logical_zero_fill_clipped_execution_first_cut") ==
            0) {
        return
            PURE_GEMM_PROTO_LOGICAL_ZERO_FILL_CLIPPED_EXECUTION;
    }
    if (name != NULL &&
        strcmp(name,
               "irregular_gemm_residual_device_overhead_autopsy_first_cut") == 0) {
        return
            PURE_GEMM_PROTO_IRREGULAR_GEMM_FUSED_RIGHT_EDGE_CLEAN_TIMING_FIRST_CUT;
    }
    return PURE_GEMM_PROTO_OFF;
}

static void
fill_packed_square_u32(volatile uint32_t *dst, uint32_t logical_dim,
                       uint32_t packed_dim, uint32_t value)
{
    uint32_t r;
    uint32_t c;

    fill_u32_buffer(dst, (size_t)packed_dim * packed_dim, 0);
    for (r = 0; r < logical_dim; ++r) {
        volatile uint32_t *row_ptr = dst + (size_t)r * packed_dim;
        for (c = 0; c < logical_dim; ++c) {
            row_ptr[c] = value;
        }
    }
}

static int
validate_expected_square_result_stride(const volatile uint32_t *matrix_C_cpu_ptr,
                                       uint32_t logical_dim,
                                       uint32_t packed_dim,
                                       uint32_t expected_value)
{
    const uint32_t probe_span = logical_dim < 2 ? logical_dim : 2;
    uint32_t row_starts[2] = {0, logical_dim > probe_span ? logical_dim - probe_span : 0};
    uint32_t col_starts[2] = {0, logical_dim > probe_span ? logical_dim - probe_span : 0};

    for (uint32_t rs = 0; rs < 2; ++rs) {
        for (uint32_t cs = 0; cs < 2; ++cs) {
            for (uint32_t r = row_starts[rs];
                 r < row_starts[rs] + probe_span && r < logical_dim; ++r) {
                for (uint32_t c = col_starts[cs];
                     c < col_starts[cs] + probe_span && c < logical_dim; ++c) {
                    const uint32_t observed =
                        matrix_C_cpu_ptr[(size_t)r * packed_dim + c];
                    if (observed != expected_value) {
                        fprintf(stderr,
                                "[ERROR] invalid packed result: C[%u,%u]=%u expected=%u "
                                "(logical=%u packed=%u)\n",
                                r, c, observed, expected_value,
                                logical_dim, packed_dim);
                        return -1;
                    }
                }
            }
        }
    }

    return 0;
}

static int
wait_for_expected_rect_result_stride(const volatile uint32_t *matrix_C_cpu_ptr,
                                     uint32_t row_base,
                                     uint32_t col_base,
                                     uint32_t rect_m,
                                     uint32_t rect_n,
                                     uint32_t packed_dim,
                                     uint32_t expected_value,
                                     unsigned long long *poll_count)
{
    const unsigned long long max_polls = 100000000ULL;
    const uint32_t probe_rows = rect_m < 2 ? rect_m : 2;
    const uint32_t probe_cols = rect_n < 2 ? rect_n : 2;
    uint32_t row_starts[2] = {
        row_base,
        row_base + (rect_m > probe_rows ? rect_m - probe_rows : 0),
    };
    uint32_t col_starts[2] = {
        col_base,
        col_base + (rect_n > probe_cols ? rect_n - probe_cols : 0),
    };

    while (*poll_count < max_polls) {
        int all_match = 1;
        refresh_device_buffer(matrix_C_cpu_ptr,
                              (size_t)packed_dim * packed_dim * sizeof(uint32_t));
        for (uint32_t rs = 0; rs < 2 && all_match; ++rs) {
            for (uint32_t cs = 0; cs < 2 && all_match; ++cs) {
                for (uint32_t r = row_starts[rs];
                     r < row_starts[rs] + probe_rows && r < row_base + rect_m;
                     ++r) {
                    for (uint32_t c = col_starts[cs];
                         c < col_starts[cs] + probe_cols &&
                         c < col_base + rect_n;
                         ++c) {
                        const uint32_t observed =
                            matrix_C_cpu_ptr[(size_t)r * packed_dim + c];
                        if (observed != expected_value) {
                            all_match = 0;
                            break;
                        }
                    }
                    if (!all_match) {
                        break;
                    }
                }
            }
        }

        if (all_match) {
            return 0;
        }

        *poll_count += 1;
        asm volatile("pause");
    }

    fprintf(stderr,
            "[ERROR] rect result timeout: row_base=%u col_base=%u rect=(%u,%u) "
            "expected=%u polls=%llu\n",
            row_base, col_base, rect_m, rect_n, expected_value,
            (unsigned long long)*poll_count);
    return -1;
}

static uint64_t
calc_tile_count(uint32_t n, uint32_t tile_dim)
{
    uint64_t per_dim = (n + tile_dim - 1U) / tile_dim;
    return per_dim * per_dim * per_dim;
}

static uint64_t
calc_rect_tile_count(uint32_t m, uint32_t n, uint32_t k, uint32_t tile_dim)
{
    const uint64_t tiles_m = (m + tile_dim - 1U) / tile_dim;
    const uint64_t tiles_n = (n + tile_dim - 1U) / tile_dim;
    const uint64_t tiles_k = (k + tile_dim - 1U) / tile_dim;
    return tiles_m * tiles_n * tiles_k;
}

static unsigned long long
matrix_elem_offset_bytes(uint32_t row, uint32_t col, uint32_t ld)
{
    return ((unsigned long long)row * ld + col) * sizeof(uint32_t);
}

static int
launch_descriptor(volatile struct Descriptor *desc_ptr,
                  volatile uint64_t *flag_ptr,
                  volatile uint64_t *doorbell_ptr,
                  unsigned long long desc_pa,
                  const struct Descriptor *desc,
                  uint64_t token,
                  unsigned long long *poll_count,
                  unsigned long long *backoff_count,
                  struct LaunchTimingMarkers *timing,
                  const char *tag)
{
    const uint64_t sentinel = make_completion_sentinel(token);
    struct Descriptor local = *desc;
    local.completion_value = token;

    printf("%s launch: dims=(%u,%u,%u) lda/ldb/ldc=(%u,%u,%u)\n",
           tag, local.m, local.n, local.k,
           local.lda, local.ldb, local.ldc);
    if (prime_completion_flag(flag_ptr, sentinel) != 0) {
        return -1;
    }
    if (timing != NULL) {
        timing->desc_setup_begin_ns = now_sim_ns();
    }
    __builtin_memcpy((void *)desc_ptr, &local, sizeof(local));
    clflush_range((const void *)desc_ptr, sizeof(local));
    asm volatile("mfence" ::: "memory");
    if (timing != NULL) {
        timing->desc_setup_end_ns = now_sim_ns();
    }
    *doorbell_ptr = desc_pa;
    asm volatile("mfence" ::: "memory");
    if (timing != NULL) {
        timing->doorbell_write_ns = now_sim_ns();
    }
    if (wait_for_completion_token(flag_ptr, token, poll_count, backoff_count,
                                  timing != NULL ? &timing->token_observed_ns
                                                 : NULL) != 0) {
        return -1;
    }
    printf("%s done: completion=%#llx polls=%llu\n",
           tag,
           (unsigned long long)*flag_ptr,
           (unsigned long long)*poll_count);
    fflush(stdout);
    return 0;
}

static int
launch_pure_gemm_rect_subproblem(volatile struct Descriptor *desc_ptr,
                                 volatile uint64_t *flag_ptr,
                                 volatile uint64_t *doorbell_ptr,
                                 unsigned long long desc_pa,
                                 const struct Descriptor *desc,
                                 uint64_t token,
                                 volatile uint32_t *matrix_C_cpu_ptr,
                                 uint32_t row_base,
                                 uint32_t col_base,
                                 uint32_t expected_value,
                                 unsigned long long *poll_count,
                                 const char *tag)
{
    const uint64_t sentinel = make_completion_sentinel(token);
    struct Descriptor local = *desc;
    local.completion_value = token;

    printf("%s launch: dims=(%u,%u,%u) lda/ldb/ldc=(%u,%u,%u)\n",
           tag, local.m, local.n, local.k,
           local.lda, local.ldb, local.ldc);
    if (prime_completion_flag(flag_ptr, sentinel) != 0) {
        return -1;
    }
    __builtin_memcpy((void *)desc_ptr, &local, sizeof(local));
    clflush_range((const void *)desc_ptr, sizeof(local));
    asm volatile("mfence" ::: "memory");
    *doorbell_ptr = desc_pa;
    asm volatile("mfence" ::: "memory");
    if (wait_for_expected_rect_result_stride(matrix_C_cpu_ptr,
                                             row_base, col_base,
                                             local.m, local.n, local.ldc,
                                             expected_value, poll_count) != 0) {
        return -1;
    }
    refresh_device_buffer((const void *)flag_ptr, sizeof(*flag_ptr));
    printf("%s done: completion=%#llx polls=%llu\n",
           tag,
           (unsigned long long)*flag_ptr,
           (unsigned long long)*poll_count);
    fflush(stdout);
    return 0;
}

static int
launch_pure_gemm_rect_batch(
    volatile struct Descriptor *desc_array_ptr,
    volatile uint64_t *flag_ptr,
    volatile uint64_t *doorbell_ptr,
    unsigned long long desc_pa,
    const struct Descriptor *desc_list,
    uint32_t desc_count,
    uint64_t final_token,
    unsigned long long *poll_count,
    unsigned long long *backoff_count,
    struct LaunchTimingMarkers *timing,
    const char *tag)
{
    const uint64_t sentinel = make_completion_sentinel(final_token);

    printf("%s launch: batch_subproblems=%u mode=single_doorbell\n",
           tag, desc_count);
    if (prime_completion_flag(flag_ptr, sentinel) != 0) {
        return -1;
    }

    if (timing != NULL) {
        timing->desc_setup_begin_ns = now_sim_ns();
    }
    for (uint32_t i = 0; i < desc_count; ++i) {
        struct Descriptor local = desc_list[i];
        local.flags |= DESC_FLAG_PEELED_SUBPROBLEM;
        if (desc_list[i].flags &
            DESC_FLAG_IRREGULAR_B_TAIL_SCRATCHPAD_OUTPUT_HOLD) {
            local.flags |= DESC_FLAG_IRREGULAR_B_TAIL_SCRATCHPAD_OUTPUT_HOLD;
        }
        if (desc_list[i].flags &
            DESC_FLAG_IRREGULAR_FUSED_EDGES_COMPLETION_OPTIMIZED) {
            local.flags |= DESC_FLAG_IRREGULAR_FUSED_EDGES_COMPLETION_OPTIMIZED;
        }
        if (desc_list[i].flags &
            DESC_FLAG_IRREGULAR_FUSED_RIGHT_EDGE_CLEAN_TIMING) {
            local.flags |= DESC_FLAG_IRREGULAR_FUSED_RIGHT_EDGE_CLEAN_TIMING;
        }
        if (desc_list[i].flags &
            DESC_FLAG_IRREGULAR_NO_WAIT_FUSED_RIGHT) {
            local.flags |= DESC_FLAG_IRREGULAR_NO_WAIT_FUSED_RIGHT;
        }
        if (desc_list[i].flags &
            DESC_FLAG_IRREGULAR_NO_WAIT_FUSED_BOTTOM) {
            local.flags |= DESC_FLAG_IRREGULAR_NO_WAIT_FUSED_BOTTOM;
        }
        if (desc_list[i].flags &
            DESC_FLAG_IRREGULAR_SINGLE_FUSED_DESCRIPTOR_CORNER_COLLAPSE) {
            local.flags |=
                DESC_FLAG_IRREGULAR_SINGLE_FUSED_DESCRIPTOR_CORNER_COLLAPSE;
        }
        if (desc_list[i].flags &
            DESC_FLAG_IRREGULAR_FINAL_COMPLETION_CHAIN_AUTOPSY) {
            local.flags |= DESC_FLAG_IRREGULAR_FINAL_COMPLETION_CHAIN_AUTOPSY;
        }
        if (desc_list[i].flags &
            DESC_FLAG_IRREGULAR_BOUNDARY_ONLY_HOLD_EARLY_BODY_WRITEBACK) {
            local.flags |=
                DESC_FLAG_IRREGULAR_BOUNDARY_ONLY_HOLD_EARLY_BODY_WRITEBACK;
        }
        if (desc_list[i].flags &
            DESC_FLAG_IRREGULAR_STATIC_OUTPUT_TILE_CLASSIFIER_BOUNDARY_HOLD) {
            local.flags |=
                DESC_FLAG_IRREGULAR_STATIC_OUTPUT_TILE_CLASSIFIER_BOUNDARY_HOLD;
        }
        if (desc_list[i].flags &
            DESC_FLAG_IRREGULAR_BOUNDARY_WRITEBACK_COALESCING) {
            local.flags |= DESC_FLAG_IRREGULAR_BOUNDARY_WRITEBACK_COALESCING;
        }
        if (desc_list[i].flags &
            DESC_FLAG_IRREGULAR_STREAMING_BODY_WRITEBACK) {
            local.flags |= DESC_FLAG_IRREGULAR_STREAMING_BODY_WRITEBACK;
        }
        if (i + 1U < desc_count) {
            local.flags |= DESC_FLAG_CHAIN_CONTINUE |
                           DESC_FLAG_SUPPRESS_COMPLETION;
            if ((local.flags &
                 (DESC_FLAG_IRREGULAR_SINGLE_FUSED_DESCRIPTOR_CORNER_COLLAPSE |
                  DESC_FLAG_IRREGULAR_FINAL_COMPLETION_CHAIN_AUTOPSY |
                  DESC_FLAG_IRREGULAR_BOUNDARY_ONLY_HOLD_EARLY_BODY_WRITEBACK |
                  DESC_FLAG_IRREGULAR_STATIC_OUTPUT_TILE_CLASSIFIER_BOUNDARY_HOLD |
                  DESC_FLAG_IRREGULAR_BOUNDARY_WRITEBACK_COALESCING |
                  DESC_FLAG_IRREGULAR_STREAMING_BODY_WRITEBACK)) &&
                i == 0) {
                local.completion_value = final_token;
            } else {
                local.completion_value = 0;
            }
        } else {
            local.flags &= ~(DESC_FLAG_CHAIN_CONTINUE |
                             DESC_FLAG_SUPPRESS_COMPLETION);
            local.completion_value = final_token;
        }
        printf("%s descriptor[%u]: dims=(%u,%u,%u) flags=%#x completion=%#llx\n",
               tag, i, local.m, local.n, local.k, local.flags,
               (unsigned long long)local.completion_value);
        __builtin_memcpy((void *)(desc_array_ptr + i), &local, sizeof(local));
    }

    clflush_range((const void *)desc_array_ptr,
                  sizeof(struct Descriptor) * desc_count);
    asm volatile("mfence" ::: "memory");
    if (timing != NULL) {
        timing->desc_setup_end_ns = now_sim_ns();
    }
    *doorbell_ptr = desc_pa;
    asm volatile("mfence" ::: "memory");
    if (timing != NULL) {
        timing->doorbell_write_ns = now_sim_ns();
    }
    if (wait_for_completion_token(flag_ptr, final_token, poll_count, backoff_count,
                                  timing != NULL ? &timing->token_observed_ns
                                                 : NULL) != 0) {
        return -1;
    }
    printf("%s done: completion=%#llx polls=%llu\n",
           tag,
           (unsigned long long)*flag_ptr,
           (unsigned long long)*poll_count);
    fflush(stdout);
    return 0;
}

static void
pack_b_tail_block(volatile uint32_t *dst,
                  const volatile uint32_t *matrix_B_cpu_ptr,
                  uint32_t logical_dim,
                  uint32_t main_dim,
                  uint32_t tail_dim)
{
    for (uint32_t r = 0; r < logical_dim; ++r) {
        for (uint32_t c = 0; c < tail_dim; ++c) {
            dst[(size_t)r * tail_dim + c] =
                matrix_B_cpu_ptr[(size_t)r * logical_dim + (main_dim + c)];
        }
    }
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
    uint32_t default_hidden_dim = 1280;
    uint32_t default_mlp_dim = 5120;
    uint32_t default_num_heads = 16;
    const char *default_phase2_mode = "staged_block";
    uint32_t default_staged_block_bytes = 1024;
    const char *default_workload = "vit_proxy";
    uint32_t configured_device_link_gbs = 0;
    enum PureGemmPrototypeMode pure_gemm_proto_mode = PURE_GEMM_PROTO_OFF;

    if (argc >= 2) {
        cxl_base = strtoull(argv[1], NULL, 0);
    }
    if (argc >= 3) {
        default_seq_len = (uint32_t)strtoul(argv[2], NULL, 0);
    }
    if (argc >= 4) {
        default_workload = argv[3];
    }
    if (argc >= 5) {
        configured_device_link_gbs = (uint32_t)strtoul(argv[4], NULL, 0);
    }
    if (argc >= 6) {
        pure_gemm_proto_mode = parse_pure_gemm_prototype_mode(argv[5]);
    }

    {
        struct VitProxyConfig cfg = {
            .preset_name = "ViT-Huge-like",
            .seq_len = default_seq_len,
            .hidden_dim = default_hidden_dim,
            .mlp_dim = default_mlp_dim,
            .num_heads = default_num_heads,
            .phase2_mode_name = default_phase2_mode,
            .staged_block_bytes = default_staged_block_bytes,
        };
        enum Phase2Mode phase2_mode =
            strcmp(cfg.phase2_mode_name, "remote_scalar") == 0
                ? MODE_REMOTE_SCALAR
                : MODE_STAGED_BLOCK;
        enum WorkloadMode workload_mode = parse_workload_mode(default_workload);
        const int run_phase2 = workload_mode == WORKLOAD_VIT_PROXY;
        const int run_phase3 = workload_mode == WORKLOAD_VIT_PROXY;
        const char *workload_name =
            workload_mode == WORKLOAD_PURE_GEMM ? "pure_gemm" : "vit_proxy";
        const uint32_t logical_matrix_size = cfg.seq_len;
        const uint32_t peeled_block_size = 256U;
        const uint32_t peeled_main_dim =
            (logical_matrix_size / peeled_block_size) * peeled_block_size;
        const uint32_t peeled_tail_dim =
            logical_matrix_size - peeled_main_dim;
        const int small_full_residency_active =
            pure_gemm_proto_mode ==
                PURE_GEMM_PROTO_SMALL_FULL_RESIDENCY_PAD256 &&
            logical_matrix_size > 0 && logical_matrix_size < 256U;
        const int logical_zero_fill_active =
            pure_gemm_proto_mode ==
                PURE_GEMM_PROTO_LOGICAL_ZERO_FILL_CLIPPED_EXECUTION &&
            logical_matrix_size > 0 && logical_matrix_size < 256U;
        const int irregular_rect_active =
            ((pure_gemm_proto_mode ==
                  PURE_GEMM_PROTO_IRREGULAR_GEMM_B_TAIL_SCRATCHPAD_OUTPUT_HOLD_FIRST_CUT) ||
             (pure_gemm_proto_mode ==
                  PURE_GEMM_PROTO_IRREGULAR_GEMM_FUSED_EDGES_COMPLETION_OPTIMIZED_FIRST_CUT) ||
             (pure_gemm_proto_mode ==
                  PURE_GEMM_PROTO_IRREGULAR_GEMM_FUSED_RIGHT_EDGE_CLEAN_TIMING_FIRST_CUT) ||
             (pure_gemm_proto_mode ==
                  PURE_GEMM_PROTO_IRREGULAR_GEMM_NO_WAIT_FUSED_RIGHT_FIRST_CUT) ||
             (pure_gemm_proto_mode ==
                  PURE_GEMM_PROTO_IRREGULAR_GEMM_NO_WAIT_FUSED_BOTTOM_FIRST_CUT) ||
             (pure_gemm_proto_mode ==
                  PURE_GEMM_PROTO_IRREGULAR_GEMM_SINGLE_FUSED_DESCRIPTOR_CORNER_COLLAPSE_FIRST_CUT) ||
             (pure_gemm_proto_mode ==
                  PURE_GEMM_PROTO_IRREGULAR_GEMM_FINAL_COMPLETION_CHAIN_AUTOPSY_FIRST_CUT) ||
             (pure_gemm_proto_mode ==
                  PURE_GEMM_PROTO_IRREGULAR_GEMM_BOUNDARY_ONLY_HOLD_EARLY_BODY_WRITEBACK_FIRST_CUT) ||
             (pure_gemm_proto_mode ==
                  PURE_GEMM_PROTO_IRREGULAR_GEMM_STATIC_OUTPUT_TILE_CLASSIFIER_BOUNDARY_HOLD_FIRST_CUT) ||
             (pure_gemm_proto_mode ==
                  PURE_GEMM_PROTO_IRREGULAR_GEMM_BOUNDARY_WRITEBACK_COALESCING_FIRST_CUT) ||
             (pure_gemm_proto_mode ==
                  PURE_GEMM_PROTO_IRREGULAR_GEMM_STREAMING_BODY_WRITEBACK_FIRST_CUT)) &&
            peeled_main_dim > 0 &&
            peeled_tail_dim > 0;
        const int peeled_rect_active =
            (((workload_mode == WORKLOAD_PURE_GEMM) &&
              ((pure_gemm_proto_mode == PURE_GEMM_PROTO_PEELED_RECT_V1) ||
               (pure_gemm_proto_mode ==
                PURE_GEMM_PROTO_PEELED_RECT_V2_RIGHT_EDGE_RECTIFIED) ||
               (pure_gemm_proto_mode ==
                PURE_GEMM_PROTO_PEELED_RECT_BATCHED_SINGLE_DOORBELL_FIRST_CUT))) ||
             irregular_rect_active);
        const int peeled_rect_batched_active =
            (((workload_mode == WORKLOAD_PURE_GEMM) &&
              (pure_gemm_proto_mode ==
               PURE_GEMM_PROTO_PEELED_RECT_BATCHED_SINGLE_DOORBELL_FIRST_CUT)) ||
             irregular_rect_active);
        const uint32_t matrix_size =
            workload_mode == WORKLOAD_PURE_GEMM &&
                    pure_gemm_proto_mode == PURE_GEMM_PROTO_TAIL_PACK_ALIGN16 &&
                    (logical_matrix_size % 16U) != 0
                ? align_up_u32(logical_matrix_size, 16U)
                : logical_matrix_size;
        const int tail_pack_active =
            workload_mode == WORKLOAD_PURE_GEMM &&
            pure_gemm_proto_mode == PURE_GEMM_PROTO_TAIL_PACK_ALIGN16 &&
            matrix_size != logical_matrix_size;
        const unsigned long long doorbell_pa = cxl_base + 0x10000ULL;
        const unsigned long long desc_pa = hdm_base;
        const unsigned long long flag_pa = hdm_base + 0x1000ULL;
        const unsigned long long matrix_span = align_up_ull(
            (unsigned long long)matrix_size * matrix_size * sizeof(uint32_t), 64ULL
        );
        const unsigned long long matrix_base =
            align_up_ull(hdm_base + 0x2000ULL, 64ULL);
        const size_t matrix_elems =
            (size_t)matrix_size * (size_t)matrix_size;
        const size_t matrix_bytes = matrix_elems * sizeof(uint32_t);
        const size_t score_elems = (size_t)cfg.seq_len * (size_t)cfg.seq_len;
        const size_t score_bytes = score_elems * sizeof(uint32_t);
        const size_t hidden_elems = (size_t)cfg.seq_len * (size_t)cfg.hidden_dim;
        const size_t hidden_bytes = hidden_elems * sizeof(uint32_t);
        const size_t hidden_span = (size_t)align_up_ull(hidden_bytes, 64ULL);
        const size_t mlp_elems = (size_t)cfg.seq_len * (size_t)cfg.mlp_dim;
        const size_t mlp_bytes = mlp_elems * sizeof(uint32_t);
        const size_t mlp_span = (size_t)align_up_ull(mlp_bytes, 64ULL);
        const size_t total_data_span =
            (size_t)(matrix_span * 3ULL) + hidden_span + hidden_span + mlp_span;
        const size_t ctl_map_size = 4096;
        const size_t flag_map_size = 4096;
        const size_t pio_map_size = 4096;
        const uint32_t block_elems =
            cfg.staged_block_bytes / sizeof(uint32_t) > 0
                ? cfg.staged_block_bytes / sizeof(uint32_t)
                : 1;
        const uint64_t tile_count_per_gemm =
            small_full_residency_active
                ? 1ULL
                : (logical_zero_fill_active
                ? calc_tile_count(cfg.seq_len, 128U)
                : (peeled_rect_active
                ? (calc_rect_tile_count(peeled_main_dim, peeled_main_dim,
                                        logical_matrix_size, 128U) +
                   calc_rect_tile_count(peeled_main_dim, peeled_tail_dim,
                                        logical_matrix_size, 128U) +
                   calc_rect_tile_count(peeled_tail_dim, peeled_main_dim,
                                        logical_matrix_size, 128U) +
                   calc_rect_tile_count(peeled_tail_dim, peeled_tail_dim,
                                        logical_matrix_size, 128U))
                : calc_tile_count(cfg.seq_len, 128U)));
        const uint64_t total_tile_count =
            tile_count_per_gemm * (run_phase3 ? 2ULL : 1ULL);
        const uint32_t descriptor_launch_count =
            peeled_rect_active ? 4U : (run_phase3 ? 2U : 1U);
        const uint32_t doorbell_launch_count =
            peeled_rect_batched_active ? 1U : descriptor_launch_count;
        const uint32_t completion_wait_count =
            peeled_rect_batched_active ? 1U : descriptor_launch_count;
        const uint32_t batch_launch_count =
            peeled_rect_batched_active ? 1U : 0U;
        const uint32_t batch_internal_step_count =
            peeled_rect_batched_active ? 4U : 0U;

        struct Descriptor desc = {
            .addrA = matrix_base,
            .addrB = matrix_base + matrix_span,
            .addrC = matrix_base + matrix_span * 2ULL,
            .flag_addr = flag_pa,
            .m = matrix_size,
            .n = matrix_size,
            .k = matrix_size,
            .lda = matrix_size,
            .ldb = matrix_size,
            .ldc = matrix_size,
            .flags = 0,
            .completion_value = 0,
        };

        int fd = -1;
        void *desc_map = MAP_FAILED;
        void *flag_map = MAP_FAILED;
        void *doorbell_map = MAP_FAILED;
        void *data_map = MAP_FAILED;
        volatile struct Descriptor *desc_ptr;
        volatile struct Descriptor *desc_array_ptr;
        volatile uint64_t *flag_ptr;
        volatile uint64_t *doorbell_ptr;
        volatile uint32_t *matrix_A_cpu_ptr;
        volatile uint32_t *matrix_B_cpu_ptr;
        volatile uint32_t *matrix_C_cpu_ptr;
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
        struct LaunchTimingMarkers phase1_launch_timing = {0};
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
        unsigned long long adaptive_backoff_count_phase1 = 0;
        unsigned long long adaptive_backoff_count_phase3 = 0;
        double phase2_non_gemm_ms;
        double phase2_inplace_ms;
        double phase1_ms;
        double clean_phase1_device_us;
        double phase3_ms;

        setbuf(stdout, NULL);
        printf("========== MatrixFlow Benchmark ==========\n");
        printf("[Config] workload=%s preset=%s seq_len=%u hidden_dim=%u mlp_dim=%u num_heads=%u\n",
               workload_name, cfg.preset_name, cfg.seq_len, cfg.hidden_dim,
               cfg.mlp_dim, cfg.num_heads);
        if (configured_device_link_gbs > 0) {
            printf("[Config] matrix_size=%u x %u, configured_device_link=%uGB/s\n",
                   logical_matrix_size, logical_matrix_size,
                   configured_device_link_gbs);
        } else {
            printf("[Config] matrix_size=%u x %u, configured_device_link=unknown\n",
                   logical_matrix_size, logical_matrix_size);
        }
        if (tail_pack_active) {
            printf("[Config] pure_gemm_prototype=pack_tail_align16 packed_size=%u x %u\n",
                   matrix_size, matrix_size);
        } else if (small_full_residency_active) {
            printf("[Config] pure_gemm_prototype="
                   "irregular_gemm_small_full_residency_pad256_first_cut "
                   "effective_dim=%u padded_dim=256\n",
                   logical_matrix_size);
        } else if (logical_zero_fill_active) {
            printf("[Config] pure_gemm_prototype="
                   "irregular_gemm_logical_zero_fill_"
                   "clipped_execution_first_cut "
                   "valid_dim=%u tile_dim=128\n",
                   logical_matrix_size);
        } else if (peeled_rect_active) {
            printf("[Config] pure_gemm_prototype=%s main_dim=%u tail_dim=%u\n",
                   pure_gemm_proto_mode ==
                           PURE_GEMM_PROTO_PEELED_RECT_BATCHED_SINGLE_DOORBELL_FIRST_CUT
                       ? "peeled_rect_batched_single_doorbell_first_cut"
                       : (pure_gemm_proto_mode ==
                                  PURE_GEMM_PROTO_IRREGULAR_GEMM_B_TAIL_SCRATCHPAD_OUTPUT_HOLD_FIRST_CUT
                              ? "irregular_gemm_b_tail_scratchpad_output_hold_first_cut"
                       : (pure_gemm_proto_mode ==
                                  PURE_GEMM_PROTO_IRREGULAR_GEMM_FUSED_EDGES_COMPLETION_OPTIMIZED_FIRST_CUT
                              ? "irregular_gemm_fused_edges_completion_optimized_first_cut"
                       : (pure_gemm_proto_mode ==
                       PURE_GEMM_PROTO_IRREGULAR_GEMM_FUSED_RIGHT_EDGE_CLEAN_TIMING_FIRST_CUT
                              ? "irregular_gemm_fused_right_edge_clean_timing_first_cut"
                       : (pure_gemm_proto_mode ==
                                  PURE_GEMM_PROTO_IRREGULAR_GEMM_NO_WAIT_FUSED_RIGHT_FIRST_CUT
                              ? "irregular_gemm_no_wait_fused_right_first_cut"
                       : (pure_gemm_proto_mode ==
                                  PURE_GEMM_PROTO_IRREGULAR_GEMM_NO_WAIT_FUSED_BOTTOM_FIRST_CUT
                              ? "irregular_gemm_no_wait_fused_bottom_first_cut"
                       : (pure_gemm_proto_mode ==
                                  PURE_GEMM_PROTO_IRREGULAR_GEMM_SINGLE_FUSED_DESCRIPTOR_CORNER_COLLAPSE_FIRST_CUT
                              ? "irregular_gemm_single_fused_descriptor_corner_collapse_first_cut"
                       : (pure_gemm_proto_mode ==
                                  PURE_GEMM_PROTO_IRREGULAR_GEMM_FINAL_COMPLETION_CHAIN_AUTOPSY_FIRST_CUT
                              ? "irregular_gemm_final_completion_chain_autopsy_first_cut"
                       : (pure_gemm_proto_mode ==
                                  PURE_GEMM_PROTO_IRREGULAR_GEMM_BOUNDARY_ONLY_HOLD_EARLY_BODY_WRITEBACK_FIRST_CUT
                              ? "irregular_gemm_boundary_only_hold_early_body_writeback_first_cut"
                       : (pure_gemm_proto_mode ==
                                  PURE_GEMM_PROTO_IRREGULAR_GEMM_STATIC_OUTPUT_TILE_CLASSIFIER_BOUNDARY_HOLD_FIRST_CUT
                              ? "irregular_gemm_static_output_tile_classifier_boundary_hold_first_cut"
                       : (pure_gemm_proto_mode ==
                                  PURE_GEMM_PROTO_IRREGULAR_GEMM_BOUNDARY_WRITEBACK_COALESCING_FIRST_CUT
                              ? "irregular_gemm_boundary_writeback_coalescing_first_cut"
                       : (pure_gemm_proto_mode ==
                                  PURE_GEMM_PROTO_IRREGULAR_GEMM_STREAMING_BODY_WRITEBACK_FIRST_CUT
                              ? "irregular_gemm_streaming_body_writeback_first_cut"
                       : (pure_gemm_proto_mode ==
                                  PURE_GEMM_PROTO_PEELED_RECT_V2_RIGHT_EDGE_RECTIFIED
                              ? "peeled_rect_v2_right_edge_rectified"
                              : "peeled_rect_v1")))))))))))),
                   peeled_main_dim, peeled_tail_dim);
        } else {
            printf("[Config] pure_gemm_prototype=off packed_size=%u x %u\n",
                   matrix_size, matrix_size);
        }
        printf("[Config] phase2_mode=%s staged_block_bytes=%u\n",
               cfg.phase2_mode_name, cfg.staged_block_bytes);

        fd = open("/dev/mem", O_RDWR | O_SYNC);
        if (fd < 0) {
            perror("open /dev/mem");
            return -1;
        }

        desc_map = mmap(NULL, ctl_map_size, PROT_READ | PROT_WRITE,
                        MAP_SHARED, fd, (off_t)desc_pa);
        if (desc_map == MAP_FAILED) {
            perror("mmap desc");
            close(fd);
            return -1;
        }

        doorbell_map = mmap(NULL, pio_map_size, PROT_READ | PROT_WRITE,
                            MAP_SHARED, fd, (off_t)doorbell_pa);
        if (doorbell_map == MAP_FAILED) {
            perror("mmap doorbell");
            munmap(desc_map, ctl_map_size);
            close(fd);
            return -1;
        }

        flag_map = mmap(NULL, flag_map_size, PROT_READ | PROT_WRITE,
                        MAP_SHARED, fd, (off_t)flag_pa);
        if (flag_map == MAP_FAILED) {
            perror("mmap flag");
            munmap(doorbell_map, pio_map_size);
            munmap(desc_map, ctl_map_size);
            close(fd);
            return -1;
        }

        data_map = mmap(NULL, total_data_span, PROT_READ | PROT_WRITE,
                        MAP_SHARED, fd, (off_t)matrix_base);
        if (data_map == MAP_FAILED) {
            perror("mmap data window");
            munmap(flag_map, flag_map_size);
            munmap(doorbell_map, pio_map_size);
            munmap(desc_map, ctl_map_size);
            close(fd);
            return -1;
        }

        desc_ptr = (volatile struct Descriptor *)desc_map;
        desc_array_ptr = (volatile struct Descriptor *)desc_map;
        flag_ptr = (volatile uint64_t *)flag_map;
        doorbell_ptr = (volatile uint64_t *)doorbell_map;
        matrix_A_cpu_ptr = (volatile uint32_t *)data_map;
        matrix_B_cpu_ptr = (volatile uint32_t *)((char *)data_map + matrix_span);
        matrix_C_cpu_ptr =
            (volatile uint32_t *)((char *)data_map + matrix_span * 2ULL);
        score_buf = matrix_C_cpu_ptr;
        hidden_buf = (volatile uint32_t *)((char *)data_map + matrix_span * 3ULL);
        residual_buf = (volatile uint32_t *)((char *)hidden_buf + hidden_span);
        mlp_buf = (volatile uint32_t *)((char *)residual_buf + hidden_span);

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
            munmap(flag_map, flag_map_size);
            munmap(doorbell_map, pio_map_size);
            munmap(desc_map, ctl_map_size);
            close(fd);
            return -1;
        }

        if (!run_phase2) {
            const uint32_t pure_gemm_a_value = 1;
            const uint32_t pure_gemm_b_value = 2;
            const uint32_t pure_gemm_c_init = 0;

            if (tail_pack_active) {
                fill_packed_square_u32(matrix_A_cpu_ptr, logical_matrix_size,
                                       matrix_size, pure_gemm_a_value);
                fill_packed_square_u32(matrix_B_cpu_ptr, logical_matrix_size,
                                       matrix_size, pure_gemm_b_value);
                fill_u32_buffer(matrix_C_cpu_ptr, matrix_elems, pure_gemm_c_init);
                clflush_range((const void *)matrix_A_cpu_ptr, matrix_bytes);
                clflush_range((const void *)matrix_B_cpu_ptr, matrix_bytes);
                clflush_range((const void *)matrix_C_cpu_ptr, matrix_bytes);
            } else {
                fill_u32_buffer(matrix_A_cpu_ptr, matrix_elems, pure_gemm_a_value);
                fill_u32_buffer(matrix_B_cpu_ptr, matrix_elems, pure_gemm_b_value);
                fill_u32_buffer(matrix_C_cpu_ptr, matrix_elems, pure_gemm_c_init);
                if (pure_gemm_proto_mode ==
                        PURE_GEMM_PROTO_IRREGULAR_GEMM_NO_WAIT_FUSED_RIGHT_FIRST_CUT ||
                    pure_gemm_proto_mode ==
                        PURE_GEMM_PROTO_IRREGULAR_GEMM_NO_WAIT_FUSED_BOTTOM_FIRST_CUT ||
                    pure_gemm_proto_mode ==
                        PURE_GEMM_PROTO_IRREGULAR_GEMM_SINGLE_FUSED_DESCRIPTOR_CORNER_COLLAPSE_FIRST_CUT ||
                    pure_gemm_proto_mode ==
                        PURE_GEMM_PROTO_IRREGULAR_GEMM_FINAL_COMPLETION_CHAIN_AUTOPSY_FIRST_CUT ||
                    pure_gemm_proto_mode ==
                        PURE_GEMM_PROTO_IRREGULAR_GEMM_BOUNDARY_ONLY_HOLD_EARLY_BODY_WRITEBACK_FIRST_CUT ||
                    pure_gemm_proto_mode ==
                        PURE_GEMM_PROTO_IRREGULAR_GEMM_STATIC_OUTPUT_TILE_CLASSIFIER_BOUNDARY_HOLD_FIRST_CUT ||
                    pure_gemm_proto_mode ==
                        PURE_GEMM_PROTO_IRREGULAR_GEMM_BOUNDARY_WRITEBACK_COALESCING_FIRST_CUT ||
                    pure_gemm_proto_mode ==
                        PURE_GEMM_PROTO_IRREGULAR_GEMM_STREAMING_BODY_WRITEBACK_FIRST_CUT) {
                    pack_b_tail_block(hidden_buf, matrix_B_cpu_ptr,
                                      logical_matrix_size, peeled_main_dim,
                                      peeled_tail_dim);
                    clflush_range((const void *)hidden_buf,
                                  (size_t)logical_matrix_size *
                                      peeled_tail_dim * sizeof(uint32_t));
                }
                clflush_range((const void *)matrix_A_cpu_ptr, matrix_bytes);
                clflush_range((const void *)matrix_B_cpu_ptr, matrix_bytes);
                clflush_range((const void *)matrix_C_cpu_ptr, matrix_bytes);
            }
            asm volatile("mfence" ::: "memory");
        } else {
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
            if (pure_gemm_proto_mode ==
                    PURE_GEMM_PROTO_IRREGULAR_GEMM_NO_WAIT_FUSED_RIGHT_FIRST_CUT ||
                pure_gemm_proto_mode ==
                    PURE_GEMM_PROTO_IRREGULAR_GEMM_NO_WAIT_FUSED_BOTTOM_FIRST_CUT ||
                pure_gemm_proto_mode ==
                    PURE_GEMM_PROTO_IRREGULAR_GEMM_SINGLE_FUSED_DESCRIPTOR_CORNER_COLLAPSE_FIRST_CUT ||
                pure_gemm_proto_mode ==
                    PURE_GEMM_PROTO_IRREGULAR_GEMM_FINAL_COMPLETION_CHAIN_AUTOPSY_FIRST_CUT ||
                pure_gemm_proto_mode ==
                    PURE_GEMM_PROTO_IRREGULAR_GEMM_BOUNDARY_ONLY_HOLD_EARLY_BODY_WRITEBACK_FIRST_CUT ||
                pure_gemm_proto_mode ==
                    PURE_GEMM_PROTO_IRREGULAR_GEMM_STATIC_OUTPUT_TILE_CLASSIFIER_BOUNDARY_HOLD_FIRST_CUT ||
                pure_gemm_proto_mode ==
                    PURE_GEMM_PROTO_IRREGULAR_GEMM_BOUNDARY_WRITEBACK_COALESCING_FIRST_CUT ||
                pure_gemm_proto_mode ==
                    PURE_GEMM_PROTO_IRREGULAR_GEMM_STREAMING_BODY_WRITEBACK_FIRST_CUT) {
                pack_b_tail_block(hidden_buf, matrix_B_cpu_ptr,
                                  logical_matrix_size, peeled_main_dim,
                                  peeled_tail_dim);
            }
            flush_remote_range(matrix_A_cpu_ptr, total_data_span);
        }

        m5_reset_stats(0, 0);

        phase1_begin = now_sim_ns();
        printf("[Phase 1] GEMM1 on device HDM...\n");
        if (peeled_rect_active) {
            struct Descriptor sub_desc = desc;
            uint32_t launch_idx = 0;
            const uint32_t pure_gemm_expected_c = logical_matrix_size * 2U;
            struct Descriptor batch_descs[4];

            sub_desc.k = logical_matrix_size;
            sub_desc.lda = logical_matrix_size;
            sub_desc.ldb = logical_matrix_size;
            sub_desc.ldc = logical_matrix_size;

            sub_desc.addrA = matrix_base;
            sub_desc.addrB = matrix_base + matrix_span;
            sub_desc.addrC = matrix_base + matrix_span * 2ULL;
            sub_desc.m = peeled_main_dim;
            sub_desc.n = peeled_main_dim;
            batch_descs[0] = sub_desc;

            sub_desc.addrA = matrix_base;
            sub_desc.addrB = matrix_base + matrix_span +
                matrix_elem_offset_bytes(0, peeled_main_dim,
                                         logical_matrix_size);
            sub_desc.addrC = matrix_base + matrix_span * 2ULL +
                matrix_elem_offset_bytes(0, peeled_main_dim,
                                         logical_matrix_size);
            sub_desc.m = peeled_main_dim;
            sub_desc.n = peeled_tail_dim;
            batch_descs[1] = sub_desc;

            sub_desc.addrA = matrix_base +
                matrix_elem_offset_bytes(peeled_main_dim, 0,
                                         logical_matrix_size);
            sub_desc.addrB = matrix_base + matrix_span;
            sub_desc.addrC = matrix_base + matrix_span * 2ULL +
                matrix_elem_offset_bytes(peeled_main_dim, 0,
                                         logical_matrix_size);
            sub_desc.m = peeled_tail_dim;
            sub_desc.n = peeled_main_dim;
            batch_descs[2] = sub_desc;

            sub_desc.addrA = matrix_base +
                matrix_elem_offset_bytes(peeled_main_dim, 0,
                                         logical_matrix_size);
            sub_desc.addrB = matrix_base + matrix_span +
                matrix_elem_offset_bytes(0, peeled_main_dim,
                                         logical_matrix_size);
            sub_desc.addrC = matrix_base + matrix_span * 2ULL +
                matrix_elem_offset_bytes(peeled_main_dim, peeled_main_dim,
                                         logical_matrix_size);
            sub_desc.m = peeled_tail_dim;
            sub_desc.n = peeled_tail_dim;
            batch_descs[3] = sub_desc;

            if (pure_gemm_proto_mode ==
                PURE_GEMM_PROTO_IRREGULAR_GEMM_B_TAIL_SCRATCHPAD_OUTPUT_HOLD_FIRST_CUT) {
                for (uint32_t batch_i = 0; batch_i < 4; ++batch_i) {
                    batch_descs[batch_i].flags |=
                        DESC_FLAG_IRREGULAR_B_TAIL_SCRATCHPAD_OUTPUT_HOLD;
                }
            } else if (pure_gemm_proto_mode ==
                       PURE_GEMM_PROTO_IRREGULAR_GEMM_FUSED_EDGES_COMPLETION_OPTIMIZED_FIRST_CUT) {
                for (uint32_t batch_i = 0; batch_i < 4; ++batch_i) {
                    batch_descs[batch_i].flags |=
                        DESC_FLAG_IRREGULAR_B_TAIL_SCRATCHPAD_OUTPUT_HOLD |
                        DESC_FLAG_IRREGULAR_FUSED_EDGES_COMPLETION_OPTIMIZED;
                }
            } else if (pure_gemm_proto_mode ==
                       PURE_GEMM_PROTO_IRREGULAR_GEMM_FUSED_RIGHT_EDGE_CLEAN_TIMING_FIRST_CUT) {
                for (uint32_t batch_i = 0; batch_i < 4; ++batch_i) {
                    batch_descs[batch_i].flags |=
                        DESC_FLAG_IRREGULAR_B_TAIL_SCRATCHPAD_OUTPUT_HOLD |
                        DESC_FLAG_IRREGULAR_FUSED_RIGHT_EDGE_CLEAN_TIMING;
                }
            } else if (pure_gemm_proto_mode ==
                       PURE_GEMM_PROTO_IRREGULAR_GEMM_NO_WAIT_FUSED_RIGHT_FIRST_CUT) {
                for (uint32_t batch_i = 0; batch_i < 4; ++batch_i) {
                    batch_descs[batch_i].flags |=
                        DESC_FLAG_IRREGULAR_B_TAIL_SCRATCHPAD_OUTPUT_HOLD |
                        DESC_FLAG_IRREGULAR_FUSED_RIGHT_EDGE_CLEAN_TIMING |
                        DESC_FLAG_IRREGULAR_NO_WAIT_FUSED_RIGHT;
                }
            } else if (pure_gemm_proto_mode ==
                       PURE_GEMM_PROTO_IRREGULAR_GEMM_NO_WAIT_FUSED_BOTTOM_FIRST_CUT) {
                for (uint32_t batch_i = 0; batch_i < 4; ++batch_i) {
                    batch_descs[batch_i].flags |=
                        DESC_FLAG_IRREGULAR_B_TAIL_SCRATCHPAD_OUTPUT_HOLD |
                        DESC_FLAG_IRREGULAR_FUSED_RIGHT_EDGE_CLEAN_TIMING |
                        DESC_FLAG_IRREGULAR_NO_WAIT_FUSED_RIGHT |
                        DESC_FLAG_IRREGULAR_NO_WAIT_FUSED_BOTTOM;
                }
            } else if (pure_gemm_proto_mode ==
                       PURE_GEMM_PROTO_IRREGULAR_GEMM_SINGLE_FUSED_DESCRIPTOR_CORNER_COLLAPSE_FIRST_CUT) {
                for (uint32_t batch_i = 0; batch_i < 4; ++batch_i) {
                    batch_descs[batch_i].flags |=
                        DESC_FLAG_IRREGULAR_B_TAIL_SCRATCHPAD_OUTPUT_HOLD |
                        DESC_FLAG_IRREGULAR_FUSED_RIGHT_EDGE_CLEAN_TIMING |
                        DESC_FLAG_IRREGULAR_NO_WAIT_FUSED_RIGHT |
                        DESC_FLAG_IRREGULAR_NO_WAIT_FUSED_BOTTOM |
                        DESC_FLAG_IRREGULAR_SINGLE_FUSED_DESCRIPTOR_CORNER_COLLAPSE;
                }
            } else if (pure_gemm_proto_mode ==
                       PURE_GEMM_PROTO_IRREGULAR_GEMM_FINAL_COMPLETION_CHAIN_AUTOPSY_FIRST_CUT) {
                for (uint32_t batch_i = 0; batch_i < 4; ++batch_i) {
                    batch_descs[batch_i].flags |=
                        DESC_FLAG_IRREGULAR_B_TAIL_SCRATCHPAD_OUTPUT_HOLD |
                        DESC_FLAG_IRREGULAR_FUSED_RIGHT_EDGE_CLEAN_TIMING |
                        DESC_FLAG_IRREGULAR_NO_WAIT_FUSED_RIGHT |
                        DESC_FLAG_IRREGULAR_NO_WAIT_FUSED_BOTTOM |
                        DESC_FLAG_IRREGULAR_SINGLE_FUSED_DESCRIPTOR_CORNER_COLLAPSE |
                        DESC_FLAG_IRREGULAR_FINAL_COMPLETION_CHAIN_AUTOPSY;
                }
            } else if (pure_gemm_proto_mode ==
                       PURE_GEMM_PROTO_IRREGULAR_GEMM_BOUNDARY_ONLY_HOLD_EARLY_BODY_WRITEBACK_FIRST_CUT) {
                for (uint32_t batch_i = 0; batch_i < 4; ++batch_i) {
                    batch_descs[batch_i].flags |=
                        DESC_FLAG_IRREGULAR_B_TAIL_SCRATCHPAD_OUTPUT_HOLD |
                        DESC_FLAG_IRREGULAR_FUSED_RIGHT_EDGE_CLEAN_TIMING |
                        DESC_FLAG_IRREGULAR_NO_WAIT_FUSED_RIGHT |
                        DESC_FLAG_IRREGULAR_NO_WAIT_FUSED_BOTTOM |
                        DESC_FLAG_IRREGULAR_SINGLE_FUSED_DESCRIPTOR_CORNER_COLLAPSE |
                        DESC_FLAG_IRREGULAR_FINAL_COMPLETION_CHAIN_AUTOPSY |
                        DESC_FLAG_IRREGULAR_BOUNDARY_ONLY_HOLD_EARLY_BODY_WRITEBACK;
                }
            } else if (pure_gemm_proto_mode ==
                       PURE_GEMM_PROTO_IRREGULAR_GEMM_STATIC_OUTPUT_TILE_CLASSIFIER_BOUNDARY_HOLD_FIRST_CUT) {
                for (uint32_t batch_i = 0; batch_i < 4; ++batch_i) {
                    batch_descs[batch_i].flags |=
                        DESC_FLAG_IRREGULAR_B_TAIL_SCRATCHPAD_OUTPUT_HOLD |
                        DESC_FLAG_IRREGULAR_FUSED_RIGHT_EDGE_CLEAN_TIMING |
                        DESC_FLAG_IRREGULAR_NO_WAIT_FUSED_RIGHT |
                        DESC_FLAG_IRREGULAR_NO_WAIT_FUSED_BOTTOM |
                        DESC_FLAG_IRREGULAR_SINGLE_FUSED_DESCRIPTOR_CORNER_COLLAPSE |
                        DESC_FLAG_IRREGULAR_FINAL_COMPLETION_CHAIN_AUTOPSY |
                        DESC_FLAG_IRREGULAR_BOUNDARY_ONLY_HOLD_EARLY_BODY_WRITEBACK |
                        DESC_FLAG_IRREGULAR_STATIC_OUTPUT_TILE_CLASSIFIER_BOUNDARY_HOLD;
                }
            } else if (pure_gemm_proto_mode ==
                       PURE_GEMM_PROTO_IRREGULAR_GEMM_BOUNDARY_WRITEBACK_COALESCING_FIRST_CUT) {
                for (uint32_t batch_i = 0; batch_i < 4; ++batch_i) {
                    batch_descs[batch_i].flags |=
                        DESC_FLAG_IRREGULAR_B_TAIL_SCRATCHPAD_OUTPUT_HOLD |
                        DESC_FLAG_IRREGULAR_FUSED_RIGHT_EDGE_CLEAN_TIMING |
                        DESC_FLAG_IRREGULAR_NO_WAIT_FUSED_RIGHT |
                        DESC_FLAG_IRREGULAR_NO_WAIT_FUSED_BOTTOM |
                        DESC_FLAG_IRREGULAR_SINGLE_FUSED_DESCRIPTOR_CORNER_COLLAPSE |
                        DESC_FLAG_IRREGULAR_FINAL_COMPLETION_CHAIN_AUTOPSY |
                        DESC_FLAG_IRREGULAR_BOUNDARY_ONLY_HOLD_EARLY_BODY_WRITEBACK |
                        DESC_FLAG_IRREGULAR_STATIC_OUTPUT_TILE_CLASSIFIER_BOUNDARY_HOLD |
                        DESC_FLAG_IRREGULAR_BOUNDARY_WRITEBACK_COALESCING;
                }
            } else if (pure_gemm_proto_mode ==
                       PURE_GEMM_PROTO_IRREGULAR_GEMM_STREAMING_BODY_WRITEBACK_FIRST_CUT) {
                for (uint32_t batch_i = 0; batch_i < 4; ++batch_i) {
                    batch_descs[batch_i].flags |=
                        DESC_FLAG_IRREGULAR_B_TAIL_SCRATCHPAD_OUTPUT_HOLD |
                        DESC_FLAG_IRREGULAR_FUSED_RIGHT_EDGE_CLEAN_TIMING |
                        DESC_FLAG_IRREGULAR_NO_WAIT_FUSED_RIGHT |
                        DESC_FLAG_IRREGULAR_NO_WAIT_FUSED_BOTTOM |
                        DESC_FLAG_IRREGULAR_SINGLE_FUSED_DESCRIPTOR_CORNER_COLLAPSE |
                        DESC_FLAG_IRREGULAR_FINAL_COMPLETION_CHAIN_AUTOPSY |
                        DESC_FLAG_IRREGULAR_BOUNDARY_ONLY_HOLD_EARLY_BODY_WRITEBACK |
                        DESC_FLAG_IRREGULAR_STATIC_OUTPUT_TILE_CLASSIFIER_BOUNDARY_HOLD |
                        DESC_FLAG_IRREGULAR_BOUNDARY_WRITEBACK_COALESCING |
                        DESC_FLAG_IRREGULAR_STREAMING_BODY_WRITEBACK;
                }
            }

            if (peeled_rect_batched_active) {
                printf("[Phase 1] peeled batch setup: subproblems=4 "
                       "single_doorbell=1\n");
                printf("[Phase 1] body prepared: dims=(%u,%u,%u)\n",
                       batch_descs[0].m, batch_descs[0].n, batch_descs[0].k);
                printf("[Phase 1] right_edge prepared: dims=(%u,%u,%u)\n",
                       batch_descs[1].m, batch_descs[1].n, batch_descs[1].k);
                printf("[Phase 1] bottom_edge prepared: dims=(%u,%u,%u)\n",
                       batch_descs[2].m, batch_descs[2].n, batch_descs[2].k);
                printf("[Phase 1] corner prepared: dims=(%u,%u,%u)\n",
                       batch_descs[3].m, batch_descs[3].n, batch_descs[3].k);
                if (launch_pure_gemm_rect_batch(
                        desc_array_ptr, flag_ptr, doorbell_ptr, desc_pa,
                        batch_descs, 4,
                        make_completion_token(logical_matrix_size, 0x1F0U),
                        &poll_count_phase1,
                        &adaptive_backoff_count_phase1,
                        &phase1_launch_timing,
                        "[Phase 1] peeled_batch") != 0) {
                    free(mlp_block);
                    free(residual_block);
                    free(hidden_block);
                    free(hidden_row);
                    free(score_row);
                    munmap(data_map, total_data_span);
                    munmap(flag_map, flag_map_size);
                    munmap(doorbell_map, pio_map_size);
                    munmap(desc_map, ctl_map_size);
                    close(fd);
                    return 3;
                }
            } else {
                if (launch_pure_gemm_rect_subproblem(
                        desc_ptr, flag_ptr, doorbell_ptr, desc_pa, &batch_descs[0],
                        make_completion_token(logical_matrix_size,
                                              0x100U + launch_idx++),
                        matrix_C_cpu_ptr,
                        0, 0,
                        pure_gemm_expected_c,
                        &poll_count_phase1,
                        "[Phase 1] body") != 0 ||
                    launch_pure_gemm_rect_subproblem(
                        desc_ptr, flag_ptr, doorbell_ptr, desc_pa, &batch_descs[1],
                        make_completion_token(logical_matrix_size,
                                              0x100U + launch_idx++),
                        matrix_C_cpu_ptr,
                        0, peeled_main_dim,
                        pure_gemm_expected_c,
                        &poll_count_phase1,
                        "[Phase 1] right_edge") != 0 ||
                    launch_pure_gemm_rect_subproblem(
                        desc_ptr, flag_ptr, doorbell_ptr, desc_pa, &batch_descs[2],
                        make_completion_token(logical_matrix_size,
                                              0x100U + launch_idx++),
                        matrix_C_cpu_ptr,
                        peeled_main_dim, 0,
                        pure_gemm_expected_c,
                        &poll_count_phase1,
                        "[Phase 1] bottom_edge") != 0 ||
                    launch_pure_gemm_rect_subproblem(
                        desc_ptr, flag_ptr, doorbell_ptr, desc_pa, &batch_descs[3],
                        make_completion_token(logical_matrix_size,
                                              0x100U + launch_idx++),
                        matrix_C_cpu_ptr,
                        peeled_main_dim, peeled_main_dim,
                        pure_gemm_expected_c,
                        &poll_count_phase1,
                        "[Phase 1] corner") != 0) {
                    free(mlp_block);
                    free(residual_block);
                    free(hidden_block);
                    free(hidden_row);
                    free(score_row);
                    munmap(data_map, total_data_span);
                    munmap(flag_map, flag_map_size);
                    munmap(doorbell_map, pio_map_size);
                    munmap(desc_map, ctl_map_size);
                    close(fd);
                    return 3;
                }
            }
        } else {
            if (small_full_residency_active) {
                desc.flags = DESC_FLAG_IRREGULAR_SMALL_FULL_RESIDENCY_PAD256;
            } else if (logical_zero_fill_active) {
                desc.flags =
                    DESC_FLAG_IRREGULAR_LOGICAL_ZERO_FILL_CLIPPED_EXECUTION;
            }
            if (launch_descriptor(desc_ptr, flag_ptr, doorbell_ptr, desc_pa,
                                  &desc,
                                  make_completion_token(matrix_size, 1),
                                  &poll_count_phase1,
                                  &adaptive_backoff_count_phase1,
                                  &phase1_launch_timing,
                                  "[Phase 1] main") != 0) {
                free(mlp_block);
                free(residual_block);
                free(hidden_block);
                free(hidden_row);
                free(score_row);
                munmap(data_map, total_data_span);
                munmap(flag_map, flag_map_size);
                munmap(doorbell_map, pio_map_size);
                munmap(desc_map, ctl_map_size);
                close(fd);
                return 3;
            }
        }
        phase1_end = now_sim_ns();
        printf("[Phase 1] done\n");

        if (!run_phase2) {
            const uint32_t pure_gemm_expected_c = logical_matrix_size * 2U;
            const int strided_validation_active =
                tail_pack_active || peeled_rect_active;
            refresh_device_buffer(
                matrix_C_cpu_ptr,
                strided_validation_active ? matrix_bytes : score_bytes);
            if ((strided_validation_active &&
                 validate_expected_square_result_stride(
                     matrix_C_cpu_ptr, logical_matrix_size, matrix_size,
                     pure_gemm_expected_c) != 0) ||
                (!strided_validation_active &&
                 validate_expected_result(matrix_C_cpu_ptr, score_elems,
                                          pure_gemm_expected_c) != 0)) {
                free(mlp_block);
                free(residual_block);
                free(hidden_block);
                free(hidden_row);
                free(score_row);
                munmap(data_map, total_data_span);
                munmap(flag_map, flag_map_size);
                munmap(doorbell_map, pio_map_size);
                munmap(desc_map, ctl_map_size);
                close(fd);
                return 4;
            }
        }

        phase2_begin = phase1_end;
        expand_begin = phase1_end;
        expand_end = phase1_end;
        softmax_begin = phase1_end;
        softmax_end = phase1_end;
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
        phase3_begin = phase1_end;
        phase3_end = phase1_end;

        if (run_phase2) {
            phase2_begin = now_sim_ns();
            printf("[Phase 2] mode=%s over CXL HDM...\n", cfg.phase2_mode_name);

            softmax_begin = now_sim_ns();
            if (phase2_mode == MODE_STAGED_BLOCK) {
                phase2_softmax_staged(score_buf, cfg.seq_len, score_row, &score_stats);
            } else {
                phase2_softmax_scalar(score_buf, cfg.seq_len, score_row, &score_stats);
            }
            softmax_end = now_sim_ns();

            expand_begin = now_sim_ns();
            if (phase2_mode == MODE_STAGED_BLOCK) {
                phase2_expand_staged(score_buf, hidden_buf, residual_buf, cfg.seq_len,
                                     cfg.hidden_dim, score_row, hidden_row,
                                     &score_stats, &hidden_stats, &residual_stats);
            } else {
                phase2_expand_scalar(score_buf, hidden_buf, residual_buf, cfg.seq_len,
                                     cfg.hidden_dim, &score_stats, &hidden_stats,
                                     &residual_stats);
            }
            expand_end = now_sim_ns();

            layernorm_begin = now_sim_ns();
            if (phase2_mode == MODE_STAGED_BLOCK) {
                phase2_layernorm_staged(hidden_buf, cfg.seq_len, cfg.hidden_dim,
                                        hidden_row, &hidden_stats);
            } else {
                phase2_layernorm_scalar(hidden_buf, cfg.seq_len, cfg.hidden_dim,
                                        hidden_row, &hidden_stats);
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
                                      cfg.hidden_dim, cfg.mlp_dim, &hidden_stats,
                                      &mlp_stats);
            }
            project_end = now_sim_ns();

            gelu_begin = now_sim_ns();
            if (phase2_mode == MODE_STAGED_BLOCK) {
                phase2_gelu_staged(mlp_buf, cfg.seq_len, cfg.mlp_dim, block_elems,
                                   mlp_block, &mlp_stats);
            } else {
                phase2_gelu_scalar(mlp_buf, cfg.seq_len, cfg.mlp_dim, &mlp_stats);
            }
            gelu_end = now_sim_ns();

            residual_begin = now_sim_ns();
            if (phase2_mode == MODE_STAGED_BLOCK) {
                phase2_residual_staged(hidden_buf, residual_buf, cfg.seq_len,
                                       cfg.hidden_dim, block_elems, hidden_block,
                                       residual_block, &hidden_stats,
                                       &residual_stats);
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
                                      cfg.hidden_dim, &hidden_stats, &score_stats);
            }
            compact_end = now_sim_ns();

            flush_remote_range(score_buf, score_bytes);
            phase2_end = now_sim_ns();
            printf("[Phase 2] done\n");
        } else {
            printf("[Phase 2] skipped for %s\n", workload_name);
        }

        if (run_phase3) {
            const uint64_t phase3_token = make_completion_token(matrix_size, 3);
            const uint64_t phase3_sentinel =
                make_completion_sentinel(phase3_token);
            phase3_begin = now_sim_ns();
            printf("[Phase 3] GEMM2 on device HDM...\n");

            if (peeled_rect_active && peeled_rect_batched_active) {
                struct Descriptor sub_desc = desc;
                struct Descriptor batch_descs[4];

                if (pure_gemm_proto_mode ==
                        PURE_GEMM_PROTO_IRREGULAR_GEMM_NO_WAIT_FUSED_RIGHT_FIRST_CUT ||
                    pure_gemm_proto_mode ==
                        PURE_GEMM_PROTO_IRREGULAR_GEMM_NO_WAIT_FUSED_BOTTOM_FIRST_CUT ||
                    pure_gemm_proto_mode ==
                        PURE_GEMM_PROTO_IRREGULAR_GEMM_SINGLE_FUSED_DESCRIPTOR_CORNER_COLLAPSE_FIRST_CUT ||
                    pure_gemm_proto_mode ==
                        PURE_GEMM_PROTO_IRREGULAR_GEMM_FINAL_COMPLETION_CHAIN_AUTOPSY_FIRST_CUT ||
                    pure_gemm_proto_mode ==
                        PURE_GEMM_PROTO_IRREGULAR_GEMM_BOUNDARY_ONLY_HOLD_EARLY_BODY_WRITEBACK_FIRST_CUT ||
                    pure_gemm_proto_mode ==
                        PURE_GEMM_PROTO_IRREGULAR_GEMM_STATIC_OUTPUT_TILE_CLASSIFIER_BOUNDARY_HOLD_FIRST_CUT ||
                    pure_gemm_proto_mode ==
                        PURE_GEMM_PROTO_IRREGULAR_GEMM_BOUNDARY_WRITEBACK_COALESCING_FIRST_CUT ||
                    pure_gemm_proto_mode ==
                        PURE_GEMM_PROTO_IRREGULAR_GEMM_STREAMING_BODY_WRITEBACK_FIRST_CUT) {
                    pack_b_tail_block(hidden_buf, matrix_B_cpu_ptr,
                                      logical_matrix_size, peeled_main_dim,
                                      peeled_tail_dim);
                    clflush_range((const void *)hidden_buf,
                                  (size_t)logical_matrix_size *
                                      peeled_tail_dim * sizeof(uint32_t));
                    asm volatile("mfence" ::: "memory");
                }

                sub_desc.addrA = matrix_base + matrix_span * 2ULL;
                sub_desc.addrB = matrix_base + matrix_span;
                sub_desc.addrC = matrix_base;
                sub_desc.k = logical_matrix_size;
                sub_desc.lda = logical_matrix_size;
                sub_desc.ldb = logical_matrix_size;
                sub_desc.ldc = logical_matrix_size;

                sub_desc.m = peeled_main_dim;
                sub_desc.n = peeled_main_dim;
                batch_descs[0] = sub_desc;

                sub_desc.addrA = matrix_base + matrix_span * 2ULL;
                sub_desc.addrB = matrix_base + matrix_span +
                    matrix_elem_offset_bytes(0, peeled_main_dim,
                                             logical_matrix_size);
                sub_desc.addrC = matrix_base +
                    matrix_elem_offset_bytes(0, peeled_main_dim,
                                             logical_matrix_size);
                sub_desc.m = peeled_main_dim;
                sub_desc.n = peeled_tail_dim;
                batch_descs[1] = sub_desc;

                sub_desc.addrA = matrix_base + matrix_span * 2ULL +
                    matrix_elem_offset_bytes(peeled_main_dim, 0,
                                             logical_matrix_size);
                sub_desc.addrB = matrix_base + matrix_span;
                sub_desc.addrC = matrix_base +
                    matrix_elem_offset_bytes(peeled_main_dim, 0,
                                             logical_matrix_size);
                sub_desc.m = peeled_tail_dim;
                sub_desc.n = peeled_main_dim;
                batch_descs[2] = sub_desc;

                sub_desc.addrA = matrix_base + matrix_span * 2ULL +
                    matrix_elem_offset_bytes(peeled_main_dim, 0,
                                             logical_matrix_size);
                sub_desc.addrB = matrix_base + matrix_span +
                    matrix_elem_offset_bytes(0, peeled_main_dim,
                                             logical_matrix_size);
                sub_desc.addrC = matrix_base +
                    matrix_elem_offset_bytes(peeled_main_dim, peeled_main_dim,
                                             logical_matrix_size);
                sub_desc.m = peeled_tail_dim;
                sub_desc.n = peeled_tail_dim;
                batch_descs[3] = sub_desc;

                if (pure_gemm_proto_mode ==
                    PURE_GEMM_PROTO_IRREGULAR_GEMM_B_TAIL_SCRATCHPAD_OUTPUT_HOLD_FIRST_CUT) {
                    for (uint32_t batch_i = 0; batch_i < 4; ++batch_i) {
                        batch_descs[batch_i].flags |=
                            DESC_FLAG_IRREGULAR_B_TAIL_SCRATCHPAD_OUTPUT_HOLD;
                    }
                } else if (pure_gemm_proto_mode ==
                           PURE_GEMM_PROTO_IRREGULAR_GEMM_FUSED_EDGES_COMPLETION_OPTIMIZED_FIRST_CUT) {
                    for (uint32_t batch_i = 0; batch_i < 4; ++batch_i) {
                        batch_descs[batch_i].flags |=
                            DESC_FLAG_IRREGULAR_B_TAIL_SCRATCHPAD_OUTPUT_HOLD |
                            DESC_FLAG_IRREGULAR_FUSED_EDGES_COMPLETION_OPTIMIZED;
                    }
                } else if (pure_gemm_proto_mode ==
                           PURE_GEMM_PROTO_IRREGULAR_GEMM_FUSED_RIGHT_EDGE_CLEAN_TIMING_FIRST_CUT) {
                    for (uint32_t batch_i = 0; batch_i < 4; ++batch_i) {
                        batch_descs[batch_i].flags |=
                            DESC_FLAG_IRREGULAR_B_TAIL_SCRATCHPAD_OUTPUT_HOLD |
                            DESC_FLAG_IRREGULAR_FUSED_RIGHT_EDGE_CLEAN_TIMING;
                    }
                } else if (pure_gemm_proto_mode ==
                           PURE_GEMM_PROTO_IRREGULAR_GEMM_NO_WAIT_FUSED_RIGHT_FIRST_CUT) {
                    for (uint32_t batch_i = 0; batch_i < 4; ++batch_i) {
                        batch_descs[batch_i].flags |=
                            DESC_FLAG_IRREGULAR_B_TAIL_SCRATCHPAD_OUTPUT_HOLD |
                            DESC_FLAG_IRREGULAR_FUSED_RIGHT_EDGE_CLEAN_TIMING |
                            DESC_FLAG_IRREGULAR_NO_WAIT_FUSED_RIGHT;
                    }
                } else if (pure_gemm_proto_mode ==
                           PURE_GEMM_PROTO_IRREGULAR_GEMM_NO_WAIT_FUSED_BOTTOM_FIRST_CUT) {
                    for (uint32_t batch_i = 0; batch_i < 4; ++batch_i) {
                        batch_descs[batch_i].flags |=
                            DESC_FLAG_IRREGULAR_B_TAIL_SCRATCHPAD_OUTPUT_HOLD |
                            DESC_FLAG_IRREGULAR_FUSED_RIGHT_EDGE_CLEAN_TIMING |
                            DESC_FLAG_IRREGULAR_NO_WAIT_FUSED_RIGHT |
                            DESC_FLAG_IRREGULAR_NO_WAIT_FUSED_BOTTOM;
                    }
                } else if (pure_gemm_proto_mode ==
                           PURE_GEMM_PROTO_IRREGULAR_GEMM_SINGLE_FUSED_DESCRIPTOR_CORNER_COLLAPSE_FIRST_CUT) {
                    for (uint32_t batch_i = 0; batch_i < 4; ++batch_i) {
                        batch_descs[batch_i].flags |=
                            DESC_FLAG_IRREGULAR_B_TAIL_SCRATCHPAD_OUTPUT_HOLD |
                            DESC_FLAG_IRREGULAR_FUSED_RIGHT_EDGE_CLEAN_TIMING |
                            DESC_FLAG_IRREGULAR_NO_WAIT_FUSED_RIGHT |
                            DESC_FLAG_IRREGULAR_NO_WAIT_FUSED_BOTTOM |
                            DESC_FLAG_IRREGULAR_SINGLE_FUSED_DESCRIPTOR_CORNER_COLLAPSE;
                    }
                } else if (pure_gemm_proto_mode ==
                           PURE_GEMM_PROTO_IRREGULAR_GEMM_FINAL_COMPLETION_CHAIN_AUTOPSY_FIRST_CUT) {
                    for (uint32_t batch_i = 0; batch_i < 4; ++batch_i) {
                        batch_descs[batch_i].flags |=
                            DESC_FLAG_IRREGULAR_B_TAIL_SCRATCHPAD_OUTPUT_HOLD |
                            DESC_FLAG_IRREGULAR_FUSED_RIGHT_EDGE_CLEAN_TIMING |
                            DESC_FLAG_IRREGULAR_NO_WAIT_FUSED_RIGHT |
                            DESC_FLAG_IRREGULAR_NO_WAIT_FUSED_BOTTOM |
                            DESC_FLAG_IRREGULAR_SINGLE_FUSED_DESCRIPTOR_CORNER_COLLAPSE |
                            DESC_FLAG_IRREGULAR_FINAL_COMPLETION_CHAIN_AUTOPSY;
                    }
                } else if (pure_gemm_proto_mode ==
                           PURE_GEMM_PROTO_IRREGULAR_GEMM_BOUNDARY_ONLY_HOLD_EARLY_BODY_WRITEBACK_FIRST_CUT) {
                    for (uint32_t batch_i = 0; batch_i < 4; ++batch_i) {
                        batch_descs[batch_i].flags |=
                            DESC_FLAG_IRREGULAR_B_TAIL_SCRATCHPAD_OUTPUT_HOLD |
                            DESC_FLAG_IRREGULAR_FUSED_RIGHT_EDGE_CLEAN_TIMING |
                            DESC_FLAG_IRREGULAR_NO_WAIT_FUSED_RIGHT |
                            DESC_FLAG_IRREGULAR_NO_WAIT_FUSED_BOTTOM |
                            DESC_FLAG_IRREGULAR_SINGLE_FUSED_DESCRIPTOR_CORNER_COLLAPSE |
                            DESC_FLAG_IRREGULAR_FINAL_COMPLETION_CHAIN_AUTOPSY |
                            DESC_FLAG_IRREGULAR_BOUNDARY_ONLY_HOLD_EARLY_BODY_WRITEBACK;
                    }
                } else if (pure_gemm_proto_mode ==
                           PURE_GEMM_PROTO_IRREGULAR_GEMM_STATIC_OUTPUT_TILE_CLASSIFIER_BOUNDARY_HOLD_FIRST_CUT) {
                    for (uint32_t batch_i = 0; batch_i < 4; ++batch_i) {
                        batch_descs[batch_i].flags |=
                            DESC_FLAG_IRREGULAR_B_TAIL_SCRATCHPAD_OUTPUT_HOLD |
                            DESC_FLAG_IRREGULAR_FUSED_RIGHT_EDGE_CLEAN_TIMING |
                            DESC_FLAG_IRREGULAR_NO_WAIT_FUSED_RIGHT |
                            DESC_FLAG_IRREGULAR_NO_WAIT_FUSED_BOTTOM |
                            DESC_FLAG_IRREGULAR_SINGLE_FUSED_DESCRIPTOR_CORNER_COLLAPSE |
                            DESC_FLAG_IRREGULAR_FINAL_COMPLETION_CHAIN_AUTOPSY |
                            DESC_FLAG_IRREGULAR_BOUNDARY_ONLY_HOLD_EARLY_BODY_WRITEBACK |
                            DESC_FLAG_IRREGULAR_STATIC_OUTPUT_TILE_CLASSIFIER_BOUNDARY_HOLD;
                    }
                } else if (pure_gemm_proto_mode ==
                           PURE_GEMM_PROTO_IRREGULAR_GEMM_BOUNDARY_WRITEBACK_COALESCING_FIRST_CUT) {
                    for (uint32_t batch_i = 0; batch_i < 4; ++batch_i) {
                        batch_descs[batch_i].flags |=
                            DESC_FLAG_IRREGULAR_B_TAIL_SCRATCHPAD_OUTPUT_HOLD |
                            DESC_FLAG_IRREGULAR_FUSED_RIGHT_EDGE_CLEAN_TIMING |
                            DESC_FLAG_IRREGULAR_NO_WAIT_FUSED_RIGHT |
                            DESC_FLAG_IRREGULAR_NO_WAIT_FUSED_BOTTOM |
                            DESC_FLAG_IRREGULAR_SINGLE_FUSED_DESCRIPTOR_CORNER_COLLAPSE |
                            DESC_FLAG_IRREGULAR_FINAL_COMPLETION_CHAIN_AUTOPSY |
                            DESC_FLAG_IRREGULAR_BOUNDARY_ONLY_HOLD_EARLY_BODY_WRITEBACK |
                            DESC_FLAG_IRREGULAR_STATIC_OUTPUT_TILE_CLASSIFIER_BOUNDARY_HOLD |
                            DESC_FLAG_IRREGULAR_BOUNDARY_WRITEBACK_COALESCING;
                    }
                } else if (pure_gemm_proto_mode ==
                           PURE_GEMM_PROTO_IRREGULAR_GEMM_STREAMING_BODY_WRITEBACK_FIRST_CUT) {
                    for (uint32_t batch_i = 0; batch_i < 4; ++batch_i) {
                        batch_descs[batch_i].flags |=
                            DESC_FLAG_IRREGULAR_B_TAIL_SCRATCHPAD_OUTPUT_HOLD |
                            DESC_FLAG_IRREGULAR_FUSED_RIGHT_EDGE_CLEAN_TIMING |
                            DESC_FLAG_IRREGULAR_NO_WAIT_FUSED_RIGHT |
                            DESC_FLAG_IRREGULAR_NO_WAIT_FUSED_BOTTOM |
                            DESC_FLAG_IRREGULAR_SINGLE_FUSED_DESCRIPTOR_CORNER_COLLAPSE |
                            DESC_FLAG_IRREGULAR_FINAL_COMPLETION_CHAIN_AUTOPSY |
                            DESC_FLAG_IRREGULAR_BOUNDARY_ONLY_HOLD_EARLY_BODY_WRITEBACK |
                            DESC_FLAG_IRREGULAR_STATIC_OUTPUT_TILE_CLASSIFIER_BOUNDARY_HOLD |
                            DESC_FLAG_IRREGULAR_BOUNDARY_WRITEBACK_COALESCING |
                            DESC_FLAG_IRREGULAR_STREAMING_BODY_WRITEBACK;
                    }
                }

                printf("[Phase 3] peeled batch setup: subproblems=4 single_doorbell=1\n");
                if (launch_pure_gemm_rect_batch(
                        desc_array_ptr, flag_ptr, doorbell_ptr, desc_pa,
                        batch_descs, 4, phase3_token, &poll_count_phase3,
                        &adaptive_backoff_count_phase3, NULL,
                        "[Phase 3] peeled_batch") != 0) {
                    free(mlp_block);
                    free(residual_block);
                    free(hidden_block);
                    free(hidden_row);
                    free(score_row);
                    munmap(data_map, total_data_span);
                    munmap(flag_map, flag_map_size);
                    munmap(doorbell_map, pio_map_size);
                    munmap(desc_map, ctl_map_size);
                    close(fd);
                    return 6;
                }
            } else {
                desc.addrA = matrix_base + matrix_span * 2ULL;
                desc.addrC = matrix_base;
                desc.m = matrix_size;
                desc.n = matrix_size;
                desc.k = matrix_size;
                desc.lda = matrix_size;
                desc.ldb = matrix_size;
                desc.ldc = matrix_size;
                const uint32_t logical_zero_fill_flag = 1u << 15;

                if (small_full_residency_active) {
                    desc.flags =
                        DESC_FLAG_IRREGULAR_SMALL_FULL_RESIDENCY_PAD256;
                } else if (logical_zero_fill_active) {
                    desc.flags = logical_zero_fill_flag;
                } else {
                    desc.flags = 0;
                }
                desc.completion_value = phase3_token;
                if (prime_completion_flag(flag_ptr, phase3_sentinel) != 0) {
                    free(mlp_block);
                    free(residual_block);
                    free(hidden_block);
                    free(hidden_row);
                    free(score_row);
                    munmap(data_map, total_data_span);
                    munmap(flag_map, flag_map_size);
                    munmap(doorbell_map, pio_map_size);
                    munmap(desc_map, ctl_map_size);
                    close(fd);
                    return 5;
                }
                __builtin_memcpy((void *)desc_ptr, &desc, sizeof(desc));
                clflush_range((const void *)desc_ptr, sizeof(desc));
                asm volatile("mfence" ::: "memory");
                *doorbell_ptr = desc_pa;
                asm volatile("mfence" ::: "memory");
                if (wait_for_completion_token(flag_ptr, phase3_token,
                                              &poll_count_phase3,
                                              &adaptive_backoff_count_phase3,
                                              NULL) != 0) {
                    free(mlp_block);
                    free(residual_block);
                    free(hidden_block);
                    free(hidden_row);
                    free(score_row);
                    munmap(data_map, total_data_span);
                    munmap(flag_map, flag_map_size);
                    munmap(doorbell_map, pio_map_size);
                    munmap(desc_map, ctl_map_size);
                    close(fd);
                    return 6;
                }
            }
            phase3_end = now_sim_ns();
            printf("[Phase 3] done\n");
        } else {
            printf("[Phase 3] skipped for %s\n", workload_name);
        }

        total_end = now_sim_ns();
        m5_dump_stats(0, 0);

        phase2_non_gemm_ms = (softmax_end - softmax_begin +
                              layernorm_end - layernorm_begin +
                              gelu_end - gelu_begin +
                              residual_end - residual_begin) / 1.0e6;
        phase2_inplace_ms = (phase2_end - phase2_begin) / 1.0e6;
        phase1_ms = (phase1_end - phase1_begin) / 1.0e6;
        clean_phase1_device_us =
            (phase1_launch_timing.token_observed_ns >=
             phase1_launch_timing.doorbell_write_ns)
                ? (phase1_launch_timing.token_observed_ns -
                   phase1_launch_timing.doorbell_write_ns) /
                      1.0e3
                : 0.0;
        phase3_ms = (phase3_end - phase3_begin) / 1.0e6;

        printf("[Timing] preset=%s\n", cfg.preset_name);
        printf("[Timing] workload=%s\n", workload_name);
        printf("[Timing] phase2_mode=%s\n",
               run_phase2 ? cfg.phase2_mode_name : "none");
        printf("[Timing] staged_block_bytes=%u\n", cfg.staged_block_bytes);
        printf("[Timing] gemm_tile_count_per_gemm=%llu\n",
               (unsigned long long)tile_count_per_gemm);
        printf("[Timing] gemm_tile_count_total=%llu\n",
               (unsigned long long)total_tile_count);
        printf("[Timing] descriptor_launch_count=%u\n", descriptor_launch_count);
        printf("[Timing] doorbell_launch_count=%u\n", doorbell_launch_count);
        printf("[Timing] completion_count=%u\n", completion_wait_count);
        printf("[Timing] peeledBatchLaunchCount=%u\n", batch_launch_count);
        printf("[Timing] peeledBatchSubproblemCount=%u\n",
               peeled_rect_batched_active ? 4U : 0U);
        printf("[Timing] peeledBatchSingleDoorbellCount=%u\n",
               peeled_rect_batched_active ? 1U : 0U);
        printf("[Timing] peeledBatchCompletionCount=%u\n",
               peeled_rect_batched_active ? 1U : 0U);
        printf("[Timing] peeledBatchGuestWaitCount=%u\n",
               peeled_rect_batched_active ? 1U : 0U);
        printf("[Timing] peeledBatchInternalStepCount=%u\n",
               batch_internal_step_count);
        printf("[Timing] peeledLegacyLaunchCount=%u\n",
               peeled_rect_active && !peeled_rect_batched_active ? 4U : 0U);
        printf("[Timing] peeledLegacyDoorbellCount=%u\n",
               peeled_rect_active && !peeled_rect_batched_active ? 4U : 0U);
        printf("[Timing] peeledLegacyCompletionWaitCount=%u\n",
               peeled_rect_active && !peeled_rect_batched_active ? 4U : 0U);
        printf("[Timing] phase1_poll_count=%llu\n", poll_count_phase1);
        printf("[Timing] phase3_poll_count=%llu\n", poll_count_phase3);
        printf("[Timing] adaptivePollBackoffCount=%llu\n",
               adaptive_backoff_count_phase1 + adaptive_backoff_count_phase3);
        printf("[TimingRaw] phase1_begin_ns=%llu\n", phase1_begin);
        printf("[TimingRaw] phase1_desc_setup_begin_ns=%llu\n",
               phase1_launch_timing.desc_setup_begin_ns);
        printf("[TimingRaw] phase1_desc_setup_end_ns=%llu\n",
               phase1_launch_timing.desc_setup_end_ns);
        printf("[TimingRaw] phase1_device_submit_begin_ns=%llu\n",
               phase1_launch_timing.doorbell_write_ns);
        printf("[TimingRaw] phase1_doorbell_write_ns=%llu\n",
               phase1_launch_timing.doorbell_write_ns);
        printf("[TimingRaw] phase1_device_done_ns=%llu\n",
               phase1_launch_timing.token_observed_ns);
        printf("[TimingRaw] phase1_token_observed_ns=%llu\n",
               phase1_launch_timing.token_observed_ns);
        printf("[TimingRaw] phase1_end_ns=%llu\n", phase1_end);
        printf("[TimingRaw] phase2_begin_ns=%llu\n", phase2_begin);
        printf("[TimingRaw] phase2_end_ns=%llu\n", phase2_end);
        printf("[TimingRaw] phase3_begin_ns=%llu\n", phase3_begin);
        printf("[TimingRaw] phase3_end_ns=%llu\n", phase3_end);
        printf("[TimingRaw] total_end_ns=%llu\n", total_end);
        printf("[Timing] phase1_ms=%.6f\n", phase1_ms);
        printf("[Timing] clean_phase1_device_us=%.6f\n",
               clean_phase1_device_us);
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
               (unsigned long long)(run_phase2 ? (score_stats.read_bytes +
                                    score_stats.write_bytes +
                                    hidden_stats.read_bytes +
                                    hidden_stats.write_bytes +
                                    residual_stats.read_bytes +
                                    residual_stats.write_bytes +
                                    mlp_stats.read_bytes +
                                    mlp_stats.write_bytes) : 0ULL));
        printf("[Timing] proxy_remote_footprint_bytes=%llu\n",
               (unsigned long long)(run_phase2 ? (score_bytes + hidden_bytes +
                                    hidden_bytes + mlp_bytes) : 0ULL));
        printf("[Timing] total_proxy_movement_bytes=%llu\n",
               (unsigned long long)(run_phase2 ? (score_stats.read_bytes +
                                    score_stats.write_bytes +
                                    hidden_stats.read_bytes +
                                    hidden_stats.write_bytes +
                                    residual_stats.read_bytes +
                                    residual_stats.write_bytes +
                                    mlp_stats.read_bytes +
                                    mlp_stats.write_bytes) : 0ULL));
        printf("========== MatrixFlow benchmark finished ==========\n");

        free(mlp_block);
        free(residual_block);
        free(hidden_block);
        free(hidden_row);
        free(score_row);
        munmap(data_map, total_data_span);
        munmap(flag_map, flag_map_size);
        munmap(doorbell_map, pio_map_size);
        munmap(desc_map, ctl_map_size);
        close(fd);
    }

    return 0;
}
