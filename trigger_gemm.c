#define _FILE_OFFSET_BITS 64
#include <fcntl.h>
#include <gem5/m5ops.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#include <x86intrin.h>

struct Descriptor {
    uint64_t addrA;
    uint64_t addrB;
    uint64_t addrC;
    uint64_t flag_addr;
    uint32_t size;
    uint64_t completion_value;
};

_Static_assert(sizeof(struct Descriptor) == 48,
               "Descriptor layout must match MatrixFlowEngine");

struct VitProxyConfig {
    const char *preset_name;
    uint32_t seq_len;
    uint32_t hidden_dim;
    uint32_t mlp_dim;
    uint32_t num_heads;
};

enum WorkloadMode {
    WORKLOAD_VIT_PROXY = 0,
    WORKLOAD_PURE_GEMM = 1,
    WORKLOAD_HOST_LINK_GEMM = 2,
};

static unsigned long long
align_up_ull(unsigned long long value, unsigned long long align)
{
    return align == 0 ? value : ((value + align - 1) / align) * align;
}

static unsigned long long
now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (unsigned long long)ts.tv_sec * 1000000000ULL +
           (unsigned long long)ts.tv_nsec;
}

static void
clflush_range(const void *ptr, size_t size)
{
    const uintptr_t start = (uintptr_t)ptr;
    const uintptr_t end = start + size;

    for (uintptr_t p = start; p < end; p += 64) {
        _mm_clflush((const void *)p);
    }
}

static uint64_t
make_completion_token(uint32_t matrix_size, uint32_t phase_id)
{
    uint64_t token = now_ns();
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

static int
wait_for_completion_token(volatile uint64_t *flag_ptr, uint64_t token,
                          unsigned long long *poll_count)
{
    const unsigned long long max_polls = 100000000ULL;

    while (*flag_ptr != token) {
        *poll_count += 1;
        if (*poll_count >= max_polls) {
            fprintf(stderr,
                    "[ERROR] completion timeout: expected=%#llx observed=%#llx polls=%llu\n",
                    (unsigned long long)token,
                    (unsigned long long)*flag_ptr,
                    (unsigned long long)*poll_count);
            return -1;
        }
        asm volatile("pause");
    }

    return 0;
}

static int
validate_expected_result(const volatile uint32_t *matrix_C_cpu_ptr,
                         size_t matrix_elems, uint32_t expected_value)
{
    const size_t probe_count = matrix_elems < 16 ? matrix_elems : 16;

    for (size_t i = 0; i < probe_count; ++i) {
        if (matrix_C_cpu_ptr[i] != expected_value) {
            fprintf(stderr,
                    "[ERROR] invalid result detected: C[%zu]=%u expected=%u\n",
                    i, matrix_C_cpu_ptr[i], expected_value);
            return -1;
        }
    }

    return 0;
}

static void
refresh_device_buffer(const volatile void *ptr, size_t size);

static int
wait_for_expected_result(const volatile uint32_t *matrix_C_cpu_ptr,
                         size_t matrix_elems, uint32_t expected_value,
                         unsigned long long *poll_count)
{
    const size_t probe_count = matrix_elems < 16 ? matrix_elems : 16;
    const unsigned long long max_polls = 100000000ULL;

    while (*poll_count < max_polls) {
        int all_match = 1;

        refresh_device_buffer(matrix_C_cpu_ptr, probe_count * sizeof(uint32_t));
        for (size_t i = 0; i < probe_count; ++i) {
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

static void
fill_u32_buffer(volatile uint32_t *ptr, size_t elems, uint32_t value)
{
    for (size_t i = 0; i < elems; ++i) {
        ptr[i] = value;
    }
}

static void
refresh_device_buffer(const volatile void *ptr, size_t size)
{
    clflush_range((const void *)ptr, size);
    asm volatile("mfence" ::: "memory");
}

static enum WorkloadMode
parse_workload_mode(const char *name)
{
    if (name != NULL && strcmp(name, "pure_gemm") == 0) {
        return WORKLOAD_PURE_GEMM;
    }
    if (name != NULL && strcmp(name, "host_link_gemm") == 0) {
        return WORKLOAD_HOST_LINK_GEMM;
    }
    return WORKLOAD_VIT_PROXY;
}

static uint64_t
calc_tile_count(uint32_t n, uint32_t tile_dim)
{
    uint64_t per_dim = (n + tile_dim - 1U) / tile_dim;
    return per_dim * per_dim * per_dim;
}

static float
approx_gelu(float x)
{
    const float alpha = 0.7978845608f;
    const float beta = 0.044715f;
    float inner = alpha * (x + beta * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

static void
softmax_rows(float *scores, uint32_t rows, uint32_t cols)
{
    for (uint32_t r = 0; r < rows; ++r) {
        float *row = scores + (size_t)r * cols;
        float max_val = row[0];
        for (uint32_t c = 1; c < cols; ++c) {
            if (row[c] > max_val) {
                max_val = row[c];
            }
        }

        float sum = 0.0f;
        for (uint32_t c = 0; c < cols; ++c) {
            row[c] = expf(row[c] - max_val);
            sum += row[c];
        }

        float inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
        for (uint32_t c = 0; c < cols; ++c) {
            row[c] *= inv_sum;
        }
    }
}

static void
layernorm_rows(float *data, uint32_t rows, uint32_t cols)
{
    for (uint32_t r = 0; r < rows; ++r) {
        float *row = data + (size_t)r * cols;
        float mean = 0.0f;
        for (uint32_t c = 0; c < cols; ++c) {
            mean += row[c];
        }
        mean /= (float)cols;

        float var = 0.0f;
        for (uint32_t c = 0; c < cols; ++c) {
            float delta = row[c] - mean;
            var += delta * delta;
        }
        var /= (float)cols;

        float inv_std = 1.0f / sqrtf(var + 1e-5f);
        for (uint32_t c = 0; c < cols; ++c) {
            row[c] = (row[c] - mean) * inv_std;
        }
    }
}

static void
gelu_vector(float *data, size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        data[i] = approx_gelu(data[i]);
    }
}

static void
residual_add(float *dst, const float *src, size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        dst[i] += src[i];
    }
}

static void
populate_token_proxy(float *dst, const float *attn, size_t token_elems,
                     size_t attn_elems, uint32_t hidden_dim,
                     uint32_t num_heads)
{
    const float scale = 1.0f / (float)(num_heads ? num_heads : 1);
    for (size_t i = 0; i < token_elems; ++i) {
        float base = attn[i % attn_elems];
        float mod = (float)((i % hidden_dim) + 1) * 0.0005f;
        dst[i] = base * scale + mod;
    }
}

static void
project_to_mlp(float *dst, const float *src, size_t mlp_elems,
               size_t src_elems)
{
    for (size_t i = 0; i < mlp_elems; ++i) {
        float x = src[i % src_elems];
        float y = src[(i * 7) % src_elems];
        dst[i] = 0.7f * x + 0.3f * y;
    }
}

int
main(int argc, char *argv[])
{
    unsigned long long cxl_base = 0x200000000ULL;
    unsigned long long hdm_base = 0x400000000ULL;
    if (argc >= 2) {
        cxl_base = strtoull(argv[1], NULL, 0);
    }

    uint32_t default_seq_len = 257;
    uint32_t default_hidden_dim = 1280;
    uint32_t default_mlp_dim = 5120;
    uint32_t default_num_heads = 16;
    const char *default_workload = "vit_proxy";
    uint32_t configured_host_link_gbs = 0;
    unsigned long long configured_host_link_window_base = 0;

    if (argc >= 3) {
        default_seq_len = (uint32_t)strtoul(argv[2], NULL, 0);
    }
    if (argc >= 4) {
        default_workload = argv[3];
    }
    if (argc >= 5) {
        configured_host_link_gbs = (uint32_t)strtoul(argv[4], NULL, 0);
    }
    if (argc >= 6) {
        configured_host_link_window_base = strtoull(argv[5], NULL, 0);
    }

    struct VitProxyConfig cfg = {
        .preset_name = "ViT-Huge-like",
        .seq_len = default_seq_len,
        .hidden_dim = default_hidden_dim,
        .mlp_dim = default_mlp_dim,
        .num_heads = default_num_heads,
    };
    enum WorkloadMode workload_mode = parse_workload_mode(default_workload);
    const int run_host_link_gemm = workload_mode == WORKLOAD_HOST_LINK_GEMM;
    const int run_phase2 = workload_mode == WORKLOAD_VIT_PROXY;
    const int run_phase3 = workload_mode == WORKLOAD_VIT_PROXY;
    const char *workload_name = "vit_proxy";
    if (workload_mode == WORKLOAD_PURE_GEMM) {
        workload_name = "pure_gemm";
    } else if (run_host_link_gemm) {
        workload_name = "host_link_gemm";
    }

    const uint32_t matrix_size = cfg.seq_len;
    const unsigned long long doorbell_pa = cxl_base + 0x10000ULL;
    const unsigned long long desc_pa = hdm_base;
    const unsigned long long flag_pa = hdm_base + 0x1000ULL;
    const unsigned long long matrix_span = align_up_ull(
        (unsigned long long)matrix_size * matrix_size * sizeof(uint32_t), 64ULL);
    const unsigned long long matrix_base =
        align_up_ull(hdm_base + 0x2000ULL, 64ULL);

    struct Descriptor desc = {
        .addrA = matrix_base,
        .addrB = matrix_base + matrix_span,
        .addrC = matrix_base + matrix_span * 2ULL,
        .flag_addr = flag_pa,
        .size = matrix_size,
        .completion_value = 0,
    };

    const size_t ctl_map_size = 4096;
    const size_t flag_map_size = 4096;
    const size_t pio_map_size = 4096;
    const size_t matrix_elems = (size_t)matrix_size * (size_t)matrix_size;
    const size_t matrix_bytes = matrix_elems * sizeof(uint32_t);
    const size_t token_elems = (size_t)cfg.seq_len * (size_t)cfg.hidden_dim;
    const size_t mlp_elems = (size_t)cfg.seq_len * (size_t)cfg.mlp_dim;
    const uint64_t tile_count_per_gemm = calc_tile_count(cfg.seq_len, 128);
    const uint64_t total_tile_count = tile_count_per_gemm *
        (run_phase3 ? 2ULL : 1ULL);
    const uint32_t descriptor_launch_count = run_phase3 ? 2U : 1U;

    setbuf(stdout, NULL);
    printf("========== MatrixFlow Benchmark ==========\n");
    printf("[Config] workload=%s preset=%s seq_len=%u hidden_dim=%u mlp_dim=%u num_heads=%u\n",
           workload_name, cfg.preset_name, cfg.seq_len, cfg.hidden_dim, cfg.mlp_dim,
           cfg.num_heads);
    if (configured_host_link_gbs > 0) {
        printf("[Config] matrix_size=%u x %u, configured_host_link=%uGB/s\n",
               matrix_size, matrix_size, configured_host_link_gbs);
    } else {
        printf("[Config] matrix_size=%u x %u, configured_host_link=unknown\n",
               matrix_size, matrix_size);
    }

    int ctl_fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (ctl_fd < 0) {
        perror("open /dev/mem (control)");
        return -1;
    }

    int data_fd = ctl_fd;
    if (run_host_link_gemm) {
        data_fd = open("/dev/mem", O_RDWR);
        if (data_fd < 0) {
            perror("open /dev/mem (data)");
            close(ctl_fd);
            return -1;
        }
        printf("[Config] host_link_data_path=reserved_host_dram_window\n");
    } else {
        printf("[Config] host_link_data_map=sync_mmio (/dev/mem with O_SYNC)\n");
    }

    void *desc_map = mmap(NULL, ctl_map_size, PROT_READ | PROT_WRITE,
                          MAP_SHARED, ctl_fd, (off_t)desc_pa);
    if (desc_map == MAP_FAILED) {
        perror("mmap desc");
        if (data_fd != ctl_fd) {
            close(data_fd);
        }
        close(ctl_fd);
        return -1;
    }

    void *doorbell_map = mmap(NULL, pio_map_size, PROT_READ | PROT_WRITE,
                              MAP_SHARED, ctl_fd, (off_t)doorbell_pa);
    if (doorbell_map == MAP_FAILED) {
        perror("mmap doorbell");
        munmap(desc_map, ctl_map_size);
        if (data_fd != ctl_fd) {
            close(data_fd);
        }
        close(ctl_fd);
        return -1;
    }

    void *flag_map = mmap(NULL, flag_map_size, PROT_READ | PROT_WRITE,
                          MAP_SHARED, ctl_fd, (off_t)flag_pa);
    if (flag_map == MAP_FAILED) {
        perror("mmap flag");
        munmap(doorbell_map, pio_map_size);
        munmap(desc_map, ctl_map_size);
        if (data_fd != ctl_fd) {
            close(data_fd);
        }
        close(ctl_fd);
        return -1;
    }

    void *data_map = NULL;
    void *host_link_map = NULL;
    if (!run_host_link_gemm) {
        data_map = mmap(NULL, (size_t)(matrix_span * 3ULL), PROT_READ | PROT_WRITE,
                        MAP_SHARED, data_fd, (off_t)matrix_base);
        if (data_map == MAP_FAILED) {
            perror("mmap data (matrix_span*3)");
            munmap(flag_map, flag_map_size);
            munmap(doorbell_map, pio_map_size);
            munmap(desc_map, ctl_map_size);
            if (data_fd != ctl_fd) {
                close(data_fd);
            }
            close(ctl_fd);
            return -1;
        }
    } else {
        if (configured_host_link_window_base == 0) {
            fprintf(stderr,
                    "host_link_gemm requires a reserved host DRAM window base\n");
            munmap(flag_map, flag_map_size);
            munmap(doorbell_map, pio_map_size);
            munmap(desc_map, ctl_map_size);
            if (data_fd != ctl_fd) {
                close(data_fd);
            }
            close(ctl_fd);
            return -1;
        }
        host_link_map = mmap(NULL, (size_t)(matrix_span * 3ULL),
                             PROT_READ | PROT_WRITE, MAP_SHARED, data_fd,
                             (off_t)configured_host_link_window_base);
        if (host_link_map == MAP_FAILED) {
            perror("mmap host_link_window");
            munmap(flag_map, flag_map_size);
            munmap(doorbell_map, pio_map_size);
            munmap(desc_map, ctl_map_size);
            if (data_fd != ctl_fd) {
                close(data_fd);
            }
            close(ctl_fd);
            return -1;
        }
    }

    volatile struct Descriptor *desc_ptr = (volatile struct Descriptor *)desc_map;
    volatile uint64_t *flag_ptr = (volatile uint64_t *)flag_map;
    volatile uint64_t *doorbell_ptr = (volatile uint64_t *)doorbell_map;
    volatile uint32_t *matrix_A_cpu_ptr = NULL;
    volatile uint32_t *matrix_B_cpu_ptr = NULL;
    volatile uint32_t *matrix_C_cpu_ptr = NULL;
    if (!run_host_link_gemm) {
        matrix_A_cpu_ptr = (volatile uint32_t *)data_map;
        matrix_B_cpu_ptr = (volatile uint32_t *)((char *)data_map + matrix_span);
        matrix_C_cpu_ptr =
            (volatile uint32_t *)((char *)data_map + matrix_span * 2ULL);
    } else {
        matrix_A_cpu_ptr = (volatile uint32_t *)host_link_map;
        matrix_B_cpu_ptr =
            (volatile uint32_t *)((char *)host_link_map + matrix_span);
        matrix_C_cpu_ptr =
            (volatile uint32_t *)((char *)host_link_map + matrix_span * 2ULL);
    }

    uint32_t *host_shadow_C = NULL;
    float *attn_scores = NULL;
    float *token_buf = NULL;
    float *residual_buf = NULL;
    float *mlp_buf = NULL;
    unsigned long long phase1_poll_count = 0;
    unsigned long long phase3_poll_count = 0;
    unsigned long long host_link_begin = 0;
    unsigned long long host_link_h2d_begin = 0;
    unsigned long long host_link_h2d_end = 0;
    unsigned long long host_link_d2h_begin = 0;
    unsigned long long host_link_d2h_end = 0;
    unsigned long long host_link_read_bytes = 0;
    unsigned long long host_link_write_bytes = 0;
    unsigned long long host_link_read_accesses = 0;
    unsigned long long host_link_write_accesses = 0;

    const int alloc_failed =
        (run_phase2 &&
         (posix_memalign((void **)&host_shadow_C, 64, matrix_bytes) != 0 ||
          posix_memalign((void **)&attn_scores, 64,
                         matrix_elems * sizeof(float)) != 0 ||
          posix_memalign((void **)&token_buf, 64,
                         token_elems * sizeof(float)) != 0 ||
          posix_memalign((void **)&residual_buf, 64,
                         token_elems * sizeof(float)) != 0 ||
          posix_memalign((void **)&mlp_buf, 64,
                         mlp_elems * sizeof(float)) != 0));
    if (alloc_failed) {
        perror("posix_memalign proxy buffers");
        free(mlp_buf);
        free(residual_buf);
        free(token_buf);
        free(attn_scores);
        free(host_shadow_C);
        if (data_map != NULL) {
            munmap(data_map, (size_t)(matrix_span * 3ULL));
        }
        if (host_link_map != NULL) {
            munmap(host_link_map, (size_t)(matrix_span * 3ULL));
        }
        munmap(doorbell_map, pio_map_size);
        munmap(desc_map, ctl_map_size);
        if (data_fd != ctl_fd) {
            close(data_fd);
        }
        close(ctl_fd);
        return -1;
    }

    const uint32_t pure_gemm_a_value = 1;
    const uint32_t pure_gemm_b_value = 2;
    const uint32_t pure_gemm_c_init = 0;
    const uint32_t pure_gemm_expected_c = matrix_size * pure_gemm_b_value;

    if (!run_host_link_gemm) {
        fill_u32_buffer(matrix_A_cpu_ptr, matrix_elems, pure_gemm_a_value);
        fill_u32_buffer(matrix_B_cpu_ptr, matrix_elems, pure_gemm_b_value);
        fill_u32_buffer(matrix_C_cpu_ptr, matrix_elems, pure_gemm_c_init);
        clflush_range((const void *)matrix_A_cpu_ptr, matrix_bytes);
        clflush_range((const void *)matrix_B_cpu_ptr, matrix_bytes);
        clflush_range((const void *)matrix_C_cpu_ptr, matrix_bytes);
        asm volatile("mfence" ::: "memory");
    } else {
        fill_u32_buffer(matrix_A_cpu_ptr, matrix_elems, pure_gemm_a_value);
        fill_u32_buffer(matrix_B_cpu_ptr, matrix_elems, pure_gemm_b_value);
        fill_u32_buffer(matrix_C_cpu_ptr, matrix_elems, pure_gemm_c_init);
        clflush_range((const void *)matrix_A_cpu_ptr, matrix_bytes);
        clflush_range((const void *)matrix_B_cpu_ptr, matrix_bytes);
        clflush_range((const void *)matrix_C_cpu_ptr, matrix_bytes);
        asm volatile("mfence" ::: "memory");

        desc.addrA = configured_host_link_window_base;
        desc.addrB = configured_host_link_window_base + matrix_span;
        desc.addrC = configured_host_link_window_base + matrix_span * 2ULL;
        printf("[HostLink] host_phys A=%#llx B=%#llx C=%#llx\n",
               (unsigned long long)desc.addrA,
               (unsigned long long)desc.addrB,
               (unsigned long long)desc.addrC);
    }
    if (run_phase2) {
        for (size_t i = 0; i < matrix_elems; ++i) {
            host_shadow_C[i] = (uint32_t)((i * 17ULL + 13ULL) & 0xff);
        }
    }

    if (run_host_link_gemm) {
        host_link_begin = now_ns();
        m5_reset_stats(0, 0);
        host_link_h2d_begin = host_link_begin;
        host_link_h2d_end = host_link_begin;
        host_link_read_bytes = matrix_bytes * 2ULL;
        host_link_write_bytes = matrix_bytes;
        host_link_read_accesses = 2;
        host_link_write_accesses = 1;
    } else {
        m5_reset_stats(0, 0);
    }

    const uint64_t phase1_token = make_completion_token(matrix_size, 1);
    const uint64_t phase1_sentinel = make_completion_sentinel(phase1_token);
    desc.completion_value = phase1_token;
    unsigned long long phase1_begin = now_ns();
    printf("[Phase 1] GEMM1 on device HBM...\n");
    if (prime_completion_flag(flag_ptr, phase1_sentinel) != 0) {
        free(mlp_buf);
        free(residual_buf);
        free(token_buf);
        free(attn_scores);
        free(host_shadow_C);
        if (data_map != NULL) {
            munmap(data_map, (size_t)(matrix_span * 3ULL));
        }
        if (host_link_map != NULL) {
            munmap(host_link_map, (size_t)(matrix_span * 3ULL));
        }
        munmap(doorbell_map, pio_map_size);
        munmap(desc_map, ctl_map_size);
        if (data_fd != ctl_fd) {
            close(data_fd);
        }
        close(ctl_fd);
        return 2;
    }
    printf("[Phase 1] token=%#llx sentinel=%#llx primed_flag=%#llx\n",
           (unsigned long long)phase1_token,
           (unsigned long long)phase1_sentinel,
           (unsigned long long)*flag_ptr);
    __builtin_memcpy((void *)desc_ptr, &desc, sizeof(desc));
    _mm_clflush((const void *)desc_ptr);
    asm volatile("mfence" ::: "memory");
    printf("[Phase 1] pre-doorbell flag=%#llx\n",
           (unsigned long long)*flag_ptr);
    printf("[Phase 1] initial C[0]=%u expected=%u\n",
           matrix_C_cpu_ptr[0], pure_gemm_expected_c);
    *doorbell_ptr = desc_pa;
    asm volatile("mfence" ::: "memory");
    printf("[Phase 1] post-doorbell flag=%#llx\n",
           (unsigned long long)*flag_ptr);
    if (!run_phase2) {
        if (wait_for_expected_result(matrix_C_cpu_ptr, matrix_elems,
                                     pure_gemm_expected_c,
                                     &phase1_poll_count) != 0) {
            free(mlp_buf);
            free(residual_buf);
            free(token_buf);
            free(attn_scores);
            free(host_shadow_C);
            if (data_map != NULL) {
                munmap(data_map, (size_t)(matrix_span * 3ULL));
            }
            if (host_link_map != NULL) {
                munmap(host_link_map, (size_t)(matrix_span * 3ULL));
            }
            munmap(doorbell_map, pio_map_size);
            munmap(desc_map, ctl_map_size);
            if (data_fd != ctl_fd) {
                close(data_fd);
            }
            close(ctl_fd);
            return 3;
        }
    } else if (wait_for_completion_token(flag_ptr, phase1_token,
                                         &phase1_poll_count) != 0) {
        free(mlp_buf);
        free(residual_buf);
        free(token_buf);
        free(attn_scores);
        free(host_shadow_C);
        if (data_map != NULL) {
            munmap(data_map, (size_t)(matrix_span * 3ULL));
        }
        if (host_link_map != NULL) {
            munmap(host_link_map, (size_t)(matrix_span * 3ULL));
        }
        munmap(doorbell_map, pio_map_size);
        munmap(desc_map, ctl_map_size);
        if (data_fd != ctl_fd) {
            close(data_fd);
        }
        close(ctl_fd);
        return 3;
    }
    unsigned long long phase1_end = now_ns();
    printf("[Phase 1] done flag=%#llx polls=%llu\n",
           (unsigned long long)*flag_ptr,
           (unsigned long long)phase1_poll_count);

    if (!run_phase2) {
        refresh_device_buffer(matrix_C_cpu_ptr, matrix_bytes);
    }

    if (!run_phase2 &&
        validate_expected_result(matrix_C_cpu_ptr, matrix_elems,
                                 pure_gemm_expected_c) != 0) {
        free(mlp_buf);
        free(residual_buf);
        free(token_buf);
        free(attn_scores);
        free(host_shadow_C);
        if (data_map != NULL) {
            munmap(data_map, (size_t)(matrix_span * 3ULL));
        }
        if (host_link_map != NULL) {
            munmap(host_link_map, (size_t)(matrix_span * 3ULL));
        }
        munmap(doorbell_map, pio_map_size);
        munmap(desc_map, ctl_map_size);
        if (data_fd != ctl_fd) {
            close(data_fd);
        }
        close(ctl_fd);
        return 4;
    }

    unsigned long long phase2_begin = phase1_end;
    unsigned long long phase2_end = phase1_end;
    unsigned long long phase3_begin = phase1_end;
    unsigned long long phase3_end = phase1_end;
    unsigned long long copy_d2h_begin = phase1_end;
    unsigned long long copy_d2h_end = phase1_end;
    unsigned long long softmax_begin = phase1_end;
    unsigned long long softmax_end = phase1_end;
    unsigned long long layernorm_begin = phase1_end;
    unsigned long long layernorm_end = phase1_end;
    unsigned long long gelu_begin = phase1_end;
    unsigned long long gelu_end = phase1_end;
    unsigned long long residual_begin = phase1_end;
    unsigned long long residual_end = phase1_end;
    unsigned long long copy_h2d_begin = phase1_end;
    unsigned long long copy_h2d_end = phase1_end;
    unsigned long long phase2_remote_read_bytes = 0;
    unsigned long long phase2_remote_write_bytes = 0;
    unsigned long long phase2_remote_read_accesses = 0;
    unsigned long long phase2_remote_write_accesses = 0;

    if (run_phase2) {
        phase2_begin = now_ns();
        printf("[Phase 2] host-side devm-copy + Non-GEMM proxy...\n");
        phase2_remote_read_bytes = matrix_bytes;
        phase2_remote_write_bytes = matrix_bytes;
        phase2_remote_read_accesses = 1;
        phase2_remote_write_accesses = 1;

        copy_d2h_begin = now_ns();
        refresh_device_buffer(matrix_C_cpu_ptr, matrix_bytes);
        memcpy(host_shadow_C, (const void *)matrix_C_cpu_ptr, matrix_bytes);
        copy_d2h_end = now_ns();

        for (size_t i = 0; i < matrix_elems; ++i) {
            attn_scores[i] = (float)(host_shadow_C[i] & 0xffff) * 0.001f;
        }

        softmax_begin = now_ns();
        softmax_rows(attn_scores, cfg.seq_len, cfg.seq_len);
        softmax_end = now_ns();

        populate_token_proxy(token_buf, attn_scores, token_elems, matrix_elems,
                             cfg.hidden_dim, cfg.num_heads);
        memcpy(residual_buf, token_buf, token_elems * sizeof(float));

        layernorm_begin = now_ns();
        layernorm_rows(token_buf, cfg.seq_len, cfg.hidden_dim);
        layernorm_end = now_ns();

        project_to_mlp(mlp_buf, token_buf, mlp_elems, token_elems);

        gelu_begin = now_ns();
        gelu_vector(mlp_buf, mlp_elems);
        gelu_end = now_ns();

        residual_begin = now_ns();
        residual_add(token_buf, residual_buf, token_elems);
        residual_end = now_ns();

        for (size_t i = 0; i < matrix_elems; ++i) {
            float combined = attn_scores[i] + token_buf[i % token_elems] +
                             mlp_buf[i % mlp_elems];
            if (combined < 0.0f) {
                combined = 0.0f;
            }
            host_shadow_C[i] = (uint32_t)(combined * 1024.0f);
        }

        copy_h2d_begin = now_ns();
        memcpy((void *)matrix_C_cpu_ptr, host_shadow_C, matrix_bytes);
        for (uintptr_t p = (uintptr_t)matrix_C_cpu_ptr;
             p < (uintptr_t)matrix_C_cpu_ptr + matrix_bytes; p += 64) {
            _mm_clflush((const void *)p);
        }
        asm volatile("mfence" ::: "memory");
        copy_h2d_end = now_ns();
        phase2_end = now_ns();
        printf("[Phase 2] done\n");
    } else {
        printf("[Phase 2] skipped for %s\n", workload_name);
    }

    if (run_phase3) {
        const uint64_t phase3_token = make_completion_token(matrix_size, 3);
        const uint64_t phase3_sentinel = make_completion_sentinel(phase3_token);
        phase3_begin = now_ns();
        printf("[Phase 3] GEMM2 on device HBM...\n");
        desc.addrA = matrix_base + matrix_span * 2ULL;
        desc.addrC = matrix_base;
        desc.completion_value = phase3_token;
        if (prime_completion_flag(flag_ptr, phase3_sentinel) != 0) {
            free(mlp_buf);
            free(residual_buf);
            free(token_buf);
            free(attn_scores);
            free(host_shadow_C);
            if (data_map != NULL) {
                munmap(data_map, (size_t)(matrix_span * 3ULL));
            }
            if (host_link_map != NULL) {
                munmap(host_link_map, (size_t)(matrix_span * 3ULL));
            }
            munmap(doorbell_map, pio_map_size);
            munmap(desc_map, ctl_map_size);
            if (data_fd != ctl_fd) {
                close(data_fd);
            }
            close(ctl_fd);
            return 5;
        }
        printf("[Phase 3] token=%#llx sentinel=%#llx primed_flag=%#llx\n",
               (unsigned long long)phase3_token,
               (unsigned long long)phase3_sentinel,
               (unsigned long long)*flag_ptr);
        __builtin_memcpy((void *)desc_ptr, &desc, sizeof(desc));
        _mm_clflush((const void *)desc_ptr);
        asm volatile("mfence" ::: "memory");
        printf("[Phase 3] pre-doorbell flag=%#llx\n",
               (unsigned long long)*flag_ptr);
        *doorbell_ptr = desc_pa;
        asm volatile("mfence" ::: "memory");
        printf("[Phase 3] post-doorbell flag=%#llx\n",
               (unsigned long long)*flag_ptr);
        if (wait_for_completion_token(flag_ptr, phase3_token,
                                      &phase3_poll_count) != 0) {
            free(mlp_buf);
            free(residual_buf);
            free(token_buf);
            free(attn_scores);
            free(host_shadow_C);
            if (data_map != NULL) {
                munmap(data_map, (size_t)(matrix_span * 3ULL));
            }
            if (host_link_map != NULL) {
                munmap(host_link_map, (size_t)(matrix_span * 3ULL));
            }
            munmap(doorbell_map, pio_map_size);
            munmap(desc_map, ctl_map_size);
            if (data_fd != ctl_fd) {
                close(data_fd);
            }
            close(ctl_fd);
            return 6;
        }
        phase3_end = now_ns();
        printf("[Phase 3] done flag=%#llx polls=%llu\n",
               (unsigned long long)*flag_ptr,
               (unsigned long long)phase3_poll_count);
    } else {
        printf("[Phase 3] skipped for %s\n", workload_name);
    }

    host_link_d2h_begin = phase1_end;
    host_link_d2h_end = phase1_end;

    unsigned long long total_end = now_ns();
    m5_dump_stats(0, 0);

    printf("[Timing] preset=%s\n", cfg.preset_name);
    printf("[Timing] workload=%s\n", workload_name);
    printf("[Timing] phase2_mode=%s\n",
           run_phase2 ? "devm_copy" :
           (run_host_link_gemm ? "host_link_gemm" : "none"));
    printf("[Timing] gemm_tile_count_per_gemm=%llu\n",
           (unsigned long long)tile_count_per_gemm);
    printf("[Timing] gemm_tile_count_total=%llu\n",
           (unsigned long long)total_tile_count);
    printf("[Timing] descriptor_launch_count=%u\n", descriptor_launch_count);
    printf("[Timing] doorbell_launch_count=%u\n", descriptor_launch_count);
    printf("[Timing] phase1_poll_count=%llu\n", phase1_poll_count);
    printf("[Timing] phase3_poll_count=%llu\n", phase3_poll_count);
    printf("[Timing] phase1_ms=%.6f\n", (phase1_end - phase1_begin) / 1.0e6);
    printf("[Timing] phase2_d2h_ms=%.6f\n",
           (copy_d2h_end - copy_d2h_begin) / 1.0e6);
    printf("[Timing] phase2_softmax_ms=%.6f\n",
           (softmax_end - softmax_begin) / 1.0e6);
    printf("[Timing] phase2_layernorm_ms=%.6f\n",
           (layernorm_end - layernorm_begin) / 1.0e6);
    printf("[Timing] phase2_gelu_ms=%.6f\n",
           (gelu_end - gelu_begin) / 1.0e6);
    printf("[Timing] phase2_residual_ms=%.6f\n",
           (residual_end - residual_begin) / 1.0e6);
    printf("[Timing] phase2_h2d_ms=%.6f\n",
           (copy_h2d_end - copy_h2d_begin) / 1.0e6);
    printf("[Timing] phase2_non_gemm_ms=%.6f\n",
           (softmax_end - softmax_begin +
            layernorm_end - layernorm_begin +
            gelu_end - gelu_begin +
            residual_end - residual_begin) / 1.0e6);
    printf("[Timing] phase2_total_ms=%.6f\n",
           (phase2_end - phase2_begin) / 1.0e6);
    printf("[Timing] phase3_ms=%.6f\n", (phase3_end - phase3_begin) / 1.0e6);
    printf("[Timing] end_to_end_ms=%.6f\n",
           (total_end - phase1_begin) / 1.0e6);
    printf("[Timing] host_link_h2d_ms=%.6f\n",
           run_host_link_gemm ? 0.0 :
           (host_link_h2d_end - host_link_h2d_begin) / 1.0e6);
    printf("[Timing] host_link_d2h_ms=%.6f\n",
           run_host_link_gemm ? 0.0 :
           (host_link_d2h_end - host_link_d2h_begin) / 1.0e6);
    printf("[Timing] host_link_total_ms=%.6f\n",
           run_host_link_gemm ? (phase1_end - phase1_begin) / 1.0e6 :
           ((host_link_h2d_end - host_link_h2d_begin) +
            (host_link_d2h_end - host_link_d2h_begin)) / 1.0e6);
    printf("[Timing] host_link_read_bytes=%llu\n", host_link_read_bytes);
    printf("[Timing] host_link_write_bytes=%llu\n", host_link_write_bytes);
    printf("[Timing] host_link_total_bytes=%llu\n",
           host_link_read_bytes + host_link_write_bytes);
    printf("[Timing] host_link_read_accesses=%llu\n", host_link_read_accesses);
    printf("[Timing] host_link_write_accesses=%llu\n", host_link_write_accesses);
    printf("[Timing] host_mediated_copy_bytes=%llu\n",
           phase2_remote_read_bytes + phase2_remote_write_bytes +
           host_link_read_bytes + host_link_write_bytes);
    printf("[Timing] phase2_read_bytes=%llu\n",
           phase2_remote_read_bytes + host_link_read_bytes);
    printf("[Timing] phase2_write_bytes=%llu\n",
           phase2_remote_write_bytes + host_link_write_bytes);
    printf("[Timing] phase2_read_accesses=%llu\n",
           phase2_remote_read_accesses + host_link_read_accesses);
    printf("[Timing] phase2_write_accesses=%llu\n",
           phase2_remote_write_accesses + host_link_write_accesses);
    printf("========== MatrixFlow benchmark finished ==========\n");

    free(mlp_buf);
    free(residual_buf);
    free(token_buf);
    free(attn_scores);
    free(host_shadow_C);
    if (data_map != NULL) {
        munmap(data_map, (size_t)(matrix_span * 3ULL));
    }
    if (host_link_map != NULL) {
        munmap(host_link_map, (size_t)(matrix_span * 3ULL));
    }
    munmap(doorbell_map, pio_map_size);
    munmap(desc_map, ctl_map_size);
    if (data_fd != ctl_fd) {
        close(data_fd);
    }
    close(ctl_fd);
    return 0;
}
