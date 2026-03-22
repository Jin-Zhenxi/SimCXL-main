#define _FILE_OFFSET_BITS 64
#include <fcntl.h>
#include <gem5/m5ops.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
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

struct VitProxyConfig {
    const char *preset_name;
    uint32_t seq_len;
    uint32_t hidden_dim;
    uint32_t mlp_dim;
    uint32_t num_heads;
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

    if (argc >= 3) {
        default_seq_len = (uint32_t)strtoul(argv[2], NULL, 0);
    }

    struct VitProxyConfig cfg = {
        .preset_name = "ViT-Huge-like",
        .seq_len = default_seq_len,
        .hidden_dim = default_hidden_dim,
        .mlp_dim = default_mlp_dim,
        .num_heads = default_num_heads,
    };

    const uint32_t matrix_size = cfg.seq_len;
    const unsigned long long doorbell_pa = cxl_base + 0x10000ULL;
    const unsigned long long desc_pa = hdm_base;
    const unsigned long long flag_pa = hdm_base + 0x80ULL;
    const unsigned long long matrix_span = align_up_ull(
        (unsigned long long)matrix_size * matrix_size * sizeof(uint32_t), 64ULL);
    const unsigned long long matrix_base = align_up_ull(hdm_base + 0x1000ULL, 64ULL);

    struct Descriptor desc = {
        .addrA = matrix_base,
        .addrB = matrix_base + matrix_span,
        .addrC = matrix_base + matrix_span * 2ULL,
        .flag_addr = flag_pa,
        .size = matrix_size,
    };

    const size_t map_size = 4096;
    const size_t matrix_elems = (size_t)matrix_size * (size_t)matrix_size;
    const size_t matrix_bytes = matrix_elems * sizeof(uint32_t);
    const size_t token_elems = (size_t)cfg.seq_len * (size_t)cfg.hidden_dim;
    const size_t mlp_elems = (size_t)cfg.seq_len * (size_t)cfg.mlp_dim;

    setbuf(stdout, NULL);
    printf("========== ViT-inspired Layer Proxy Benchmark ==========\n");
    printf("[Config] preset=%s seq_len=%u hidden_dim=%u mlp_dim=%u num_heads=%u\n",
           cfg.preset_name, cfg.seq_len, cfg.hidden_dim, cfg.mlp_dim,
           cfg.num_heads);
    printf("[Config] GEMM proxy size=%u x %u, PCIe link=64GB/s, packetization penalty=0\n",
           matrix_size, matrix_size);

    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) {
        perror("open /dev/mem");
        return -1;
    }

    void *desc_map = mmap(NULL, map_size, PROT_READ | PROT_WRITE,
                          MAP_SHARED, fd, (off_t)desc_pa);
    if (desc_map == MAP_FAILED) {
        perror("mmap desc");
        close(fd);
        return -1;
    }

    void *doorbell_map = mmap(NULL, map_size, PROT_READ | PROT_WRITE,
                              MAP_SHARED, fd, (off_t)doorbell_pa);
    if (doorbell_map == MAP_FAILED) {
        perror("mmap doorbell");
        munmap(desc_map, map_size);
        close(fd);
        return -1;
    }

    void *data_map = mmap(NULL, (size_t)(matrix_span * 3ULL), PROT_READ | PROT_WRITE,
                          MAP_SHARED, fd, (off_t)matrix_base);
    if (data_map == MAP_FAILED) {
        perror("mmap data (matrix_span*3)");
        munmap(doorbell_map, map_size);
        munmap(desc_map, map_size);
        close(fd);
        return -1;
    }

    volatile struct Descriptor *desc_ptr = (volatile struct Descriptor *)desc_map;
    volatile uint64_t *flag_ptr =
        (volatile uint64_t *)((char *)desc_map + (flag_pa - desc_pa));
    volatile uint64_t *doorbell_ptr = (volatile uint64_t *)doorbell_map;
    volatile uint32_t *matrix_C_cpu_ptr =
        (volatile uint32_t *)((char *)data_map + matrix_span * 2ULL);

    uint32_t *host_shadow_C = NULL;
    float *attn_scores = NULL;
    float *token_buf = NULL;
    float *residual_buf = NULL;
    float *mlp_buf = NULL;

    if (posix_memalign((void **)&host_shadow_C, 64, matrix_bytes) != 0 ||
        posix_memalign((void **)&attn_scores, 64, matrix_elems * sizeof(float)) != 0 ||
        posix_memalign((void **)&token_buf, 64, token_elems * sizeof(float)) != 0 ||
        posix_memalign((void **)&residual_buf, 64, token_elems * sizeof(float)) != 0 ||
        posix_memalign((void **)&mlp_buf, 64, mlp_elems * sizeof(float)) != 0) {
        perror("posix_memalign proxy buffers");
        free(mlp_buf);
        free(residual_buf);
        free(token_buf);
        free(attn_scores);
        free(host_shadow_C);
        munmap(data_map, (size_t)(matrix_span * 3ULL));
        munmap(doorbell_map, map_size);
        munmap(desc_map, map_size);
        close(fd);
        return -1;
    }

    for (size_t i = 0; i < matrix_elems; ++i) {
        host_shadow_C[i] = (uint32_t)((i * 17ULL + 13ULL) & 0xff);
    }

    m5_reset_stats(0, 0);

    unsigned long long phase1_begin = now_ns();
    printf("[Phase 1] GEMM1 on device HBM...\n");
    *flag_ptr = 0;
    __builtin_memcpy((void *)desc_ptr, &desc, sizeof(desc));
    _mm_clflush((const void *)desc_ptr);
    _mm_clflush((const void *)flag_ptr);
    asm volatile("mfence" ::: "memory");
    *doorbell_ptr = desc_pa;
    asm volatile("mfence" ::: "memory");
    while (*flag_ptr == 0) {
        asm volatile("pause");
    }
    unsigned long long phase1_end = now_ns();
    printf("[Phase 1] done\n");

    unsigned long long phase2_begin = now_ns();
    printf("[Phase 2] host-side devm-copy + Non-GEMM proxy...\n");

    unsigned long long copy_d2h_begin = now_ns();
    memcpy(host_shadow_C, (const void *)matrix_C_cpu_ptr, matrix_bytes);
    unsigned long long copy_d2h_end = now_ns();

    for (size_t i = 0; i < matrix_elems; ++i) {
        attn_scores[i] = (float)(host_shadow_C[i] & 0xffff) * 0.001f;
    }

    unsigned long long softmax_begin = now_ns();
    softmax_rows(attn_scores, cfg.seq_len, cfg.seq_len);
    unsigned long long softmax_end = now_ns();

    populate_token_proxy(token_buf, attn_scores, token_elems, matrix_elems,
                         cfg.hidden_dim, cfg.num_heads);
    memcpy(residual_buf, token_buf, token_elems * sizeof(float));

    unsigned long long layernorm_begin = now_ns();
    layernorm_rows(token_buf, cfg.seq_len, cfg.hidden_dim);
    unsigned long long layernorm_end = now_ns();

    project_to_mlp(mlp_buf, token_buf, mlp_elems, token_elems);

    unsigned long long gelu_begin = now_ns();
    gelu_vector(mlp_buf, mlp_elems);
    unsigned long long gelu_end = now_ns();

    unsigned long long residual_begin = now_ns();
    residual_add(token_buf, residual_buf, token_elems);
    unsigned long long residual_end = now_ns();

    for (size_t i = 0; i < matrix_elems; ++i) {
        float combined = attn_scores[i] + token_buf[i % token_elems] +
                         mlp_buf[i % mlp_elems];
        if (combined < 0.0f) {
            combined = 0.0f;
        }
        host_shadow_C[i] = (uint32_t)(combined * 1024.0f);
    }

    unsigned long long copy_h2d_begin = now_ns();
    memcpy((void *)matrix_C_cpu_ptr, host_shadow_C, matrix_bytes);
    for (uintptr_t p = (uintptr_t)matrix_C_cpu_ptr;
         p < (uintptr_t)matrix_C_cpu_ptr + matrix_bytes; p += 64) {
        _mm_clflush((const void *)p);
    }
    asm volatile("mfence" ::: "memory");
    unsigned long long copy_h2d_end = now_ns();
    unsigned long long phase2_end = now_ns();
    printf("[Phase 2] done\n");

    unsigned long long phase3_begin = now_ns();
    printf("[Phase 3] GEMM2 on device HBM...\n");
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
        asm volatile("pause");
    }
    unsigned long long phase3_end = now_ns();
    printf("[Phase 3] done\n");

    unsigned long long total_end = now_ns();
    m5_dump_stats(0, 0);

    printf("[Timing] preset=%s\n", cfg.preset_name);
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
    printf("[Timing] phase2_total_ms=%.6f\n",
           (phase2_end - phase2_begin) / 1.0e6);
    printf("[Timing] phase3_ms=%.6f\n", (phase3_end - phase3_begin) / 1.0e6);
    printf("[Timing] end_to_end_ms=%.6f\n", (total_end - phase1_begin) / 1.0e6);
    printf("========== ViT-inspired proxy benchmark finished ==========\n");

    free(mlp_buf);
    free(residual_buf);
    free(token_buf);
    free(attn_scores);
    free(host_shadow_C);
    munmap(data_map, (size_t)(matrix_span * 3ULL));
    munmap(doorbell_map, map_size);
    munmap(desc_map, map_size);
    close(fd);
    return 0;
}
