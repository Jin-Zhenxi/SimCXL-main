#define _FILE_OFFSET_BITS 64
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <x86intrin.h>
#include <gem5/m5ops.h>

struct Descriptor {
    uint64_t addrA;
    uint64_t addrB;
    uint64_t addrC;
    uint64_t flag_addr;
    uint32_t size;
};

_Static_assert(sizeof(struct Descriptor) == 40,
               "Descriptor layout must match MatrixFlowEngine");

static unsigned long long align_up_ull(
    unsigned long long value, unsigned long long align)
{
    return align == 0 ? value : ((value + align - 1) / align) * align;
}

int main(int argc, char *argv[]) {
    unsigned long long cxl_base = 0x200000000ULL; // 默认打向 BAR0 Doorbell 基址
    unsigned long long hdm_base = 0x400000000ULL; // CXL HDM 数据面基址
    if (argc >= 2) {
        cxl_base = strtoull(argv[1], NULL, 0);
    }
    uint32_t matrix_size = 2048;
    if (argc >= 3) {
        matrix_size = (uint32_t)strtoul(argv[2], NULL, 0);
    }
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
    size_t map_size = 4096;

    setbuf(stdout, NULL);

    printf("[Host CPU][DBG-01] 程序启动\n");
    printf("[Host CPU][DBG-02] doorbell_base=0x%llx doorbell_pa=0x%llx hdm_base=0x%llx\n",
           cxl_base, doorbell_pa, hdm_base);
    printf("[Host CPU][DBG-03] desc=0x%llx flag=0x%llx A=0x%llx B=0x%llx C=0x%llx span=0x%llx\n",
           desc_pa, flag_pa,
           (unsigned long long)desc.addrA,
           (unsigned long long)desc.addrB,
           (unsigned long long)desc.addrC,
           matrix_span);

    // 恢复你要求的 O_SYNC
    printf("[Host CPU][DBG-04] 即将 open(/dev/mem, O_RDWR | O_SYNC)\n");
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) {
        perror("open /dev/mem 失败");
        return -1;
    }
    printf("[Host CPU][DBG-05] open 成功，fd=%d\n", fd);

    printf("[Host CPU][DBG-06] 即将 mmap HDM 描述符页，offset=0x%llx\n", desc_pa);
    void *desc_map = mmap(NULL, map_size, PROT_READ | PROT_WRITE,
                          MAP_SHARED, fd, (off_t)desc_pa);
    if (desc_map == MAP_FAILED) {
        perror("mmap 失败");
        close(fd);
        return -1;
    }
    printf("[Host CPU][DBG-07] HDM 描述符页 mmap 成功，desc_map=%p\n", desc_map);

    printf("[Host CPU][DBG-08] 即将 mmap Doorbell 页，offset=0x%llx\n", doorbell_pa);
    void *doorbell_map = mmap(NULL, map_size, PROT_READ | PROT_WRITE,
                              MAP_SHARED, fd, (off_t)doorbell_pa);
    if (doorbell_map == MAP_FAILED) {
        perror("Doorbell mmap 失败");
        munmap(desc_map, map_size);
        close(fd);
        return -1;
    }
    printf("[Host CPU][DBG-09] Doorbell 页 mmap 成功，doorbell_map=%p\n", doorbell_map);

    volatile struct Descriptor *desc_ptr = (volatile struct Descriptor *)desc_map;
    volatile uint64_t *flag_ptr =
        (volatile uint64_t *)((char *)desc_map + (flag_pa - desc_pa));
    volatile uint64_t *doorbell_ptr = (volatile uint64_t *)doorbell_map;

    printf("[Host CPU][DBG-10] 写入 descriptor 并清零 completion flag\n");
    *flag_ptr = 0;
    __builtin_memcpy((void *)desc_ptr, &desc, sizeof(desc));
    _mm_clflush((const void *)desc_ptr);
    _mm_clflush((const void *)flag_ptr);
    asm volatile("mfence" ::: "memory");

    printf("[Host CPU][DBG-11] 即将敲响 Doorbell，payload(desc_pa)=0x%llx\n", desc_pa);
    m5_reset_stats(0, 0);
    *doorbell_ptr = desc_pa;
    asm volatile("mfence" ::: "memory");
    printf("[Host CPU][DBG-12] Doorbell 写入完成\n");

    printf("[Host CPU][DBG-13] 等待 MatrixFlowEngine 写回 completion flag...\n");
    while (*flag_ptr == 0) {
        asm volatile("pause");
    }
    m5_dump_stats(0, 0);
    printf("[Host CPU][DBG-14] completion flag 已置位，value=%llu\n",
           (unsigned long long)*flag_ptr);

    printf("[Host CPU][DBG-15] 即将执行最终 _mm_mfence()\n");
    _mm_mfence();
    printf("[Host CPU][DBG-16] 最终 mfence 完成\n");

    printf("[Host CPU] 探测包已发射！\n");
    printf("[Host CPU][DBG-17] 即将 munmap\n");
    munmap(doorbell_map, map_size);
    munmap(desc_map, map_size);
    printf("[Host CPU][DBG-18] munmap 完成，即将 close(fd)\n");
    close(fd);
    printf("[Host CPU][DBG-19] close 完成，程序退出\n");
    return 0;
}
