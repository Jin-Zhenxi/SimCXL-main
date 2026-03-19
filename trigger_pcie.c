
#define _FILE_OFFSET_BITS 64
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <x86intrin.h>
#include <sys/mman.h>
#include <gem5/m5ops.h>

struct Descriptor {
    uint64_t addrA;
    uint64_t addrB;
    uint64_t addrC;
    uint64_t flag_addr;
    uint32_t size;
};

// 【核心黑魔法】：Linux 虚拟地址转真实物理地址
uint64_t get_physical_addr(void *vaddr) {
    int fd = open("/proc/self/pagemap", O_RDONLY);
    if (fd < 0) { perror("open pagemap"); exit(1); }
    uint64_t offset = ((uintptr_t)vaddr / 4096) * sizeof(uint64_t);
    uint64_t pfn;
    if (lseek(fd, offset, SEEK_SET) == (off_t)-1) { perror("lseek"); exit(1); }
    if (read(fd, &pfn, sizeof(pfn)) != sizeof(pfn)) { perror("read"); exit(1); }
    close(fd);
    return (pfn & ((1ULL << 54) - 1)) * 4096 + ((uintptr_t)vaddr % 4096);
}

static unsigned long long align_up_ull(unsigned long long value, unsigned long long align) {
    return align == 0 ? value : ((value + align - 1) / align) * align;
}

int main(int argc, char *argv[]) {
    unsigned long long cxl_base = 0x200000000ULL;
    unsigned long long hdm_base = 0x400000000ULL;
    uint32_t matrix_size = 2048; // 动态替换

    const unsigned long long doorbell_pa = cxl_base + 0x10000ULL;
    const unsigned long long flag_pa = hdm_base + 0x80ULL; // Flag 依然留在 CXL 卡上用于同步
    const unsigned long long matrix_span = align_up_ull((unsigned long long)matrix_size * matrix_size * sizeof(uint32_t), 64ULL);

    setbuf(stdout, NULL);
    printf("[Host CPU] 开始分配 Host 侧主板内存...\n");

    // 1. 在 Host CPU 主板内存中分配矩阵空间
    void *host_matrix_mem;
    if (posix_memalign(&host_matrix_mem, 4096, matrix_span * 3) != 0) {
        perror("posix_memalign 失败"); return -1;
    }
    // 【必须加上这一句】：强行写入数据，触摸每一页，逼迫 Linux 吐出真实的物理内存！
    memset(host_matrix_mem, 0x1, matrix_span * 3);
    // 2. 锁住内存，强制 Linux 分配物理页，防止被 swap 到硬盘
    if (mlock(host_matrix_mem, matrix_span * 3) != 0) {
        perror("mlock 失败"); return -1;
    }

    // 3. 查出这块 Host 内存的真实物理地址
    uint64_t pa_A = get_physical_addr(host_matrix_mem);
    uint64_t pa_B = get_physical_addr((char*)host_matrix_mem + matrix_span);
    uint64_t pa_C = get_physical_addr((char*)host_matrix_mem + matrix_span * 2);

    printf("[Host CPU] 矩阵已分配在 Host 物理地址: A=0x%llx, B=0x%llx, C=0x%llx\n",
           (unsigned long long)pa_A, (unsigned long long)pa_B, (unsigned long long)pa_C);

    struct Descriptor desc = {
        .addrA = pa_A,     // 强迫加速器跨越 PCIe 桥来 Host 拿数据！
        .addrB = pa_B,
        .addrC = pa_C,
        .flag_addr = flag_pa,
        .size = matrix_size,
    };

    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    void *desc_map = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)hdm_base);
    void *doorbell_map = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)doorbell_pa);

    volatile struct Descriptor *desc_ptr = (volatile struct Descriptor *)desc_map;
    volatile uint64_t *flag_ptr = (volatile uint64_t *)((char *)desc_map + (flag_pa - hdm_base));
    volatile uint64_t *doorbell_ptr = (volatile uint64_t *)doorbell_map;

    *flag_ptr = 0;
    __builtin_memcpy((void *)desc_ptr, &desc, sizeof(desc));
    _mm_clflush((const void *)desc_ptr);
    _mm_clflush((const void *)flag_ptr);
    asm volatile("mfence" ::: "memory");

    printf("[Host CPU] 准备就绪，敲击 Doorbell！\n");

    // ======== 核心 ROI 开始 ========
    m5_reset_stats(0, 0);

    *doorbell_ptr = hdm_base;
    asm volatile("mfence" ::: "memory");

    while (*flag_ptr == 0) {
        asm volatile("pause");
    }

    m5_dump_stats(0, 0);
    // ======== 核心 ROI 结束 ========

    printf("[Host CPU] 跨桥计算完成！\n");
    return 0;
}
