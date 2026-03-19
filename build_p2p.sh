#!/bin/bash

set -e 

echo "=================================================="
echo "🚀 步骤 1: 呼叫 SCons 进行极速增量编译 (生成/更新 .o)"
echo "=================================================="
scons build/X86/gem5.opt -j4 USE_TCMALLOC=False || true

echo ""
echo "=================================================="
echo "📦 步骤 2: 扫描车间，收集所有 .o 零件清单 (已过滤废料)"
echo "=================================================="
# 新增 grep -v "scons_config" 彻底过滤掉测试废料
find build/X86 -type f \( -name "*.o" -o -name "*.a" \) \
    | grep -v "gem5py" \
    | grep -v "unittests" \
    | grep -v "scons_config" \
    > build/X86/all_objs.txt

PART_COUNT=$(wc -l < build/X86/all_objs.txt)
echo "✅ 成功找到 $PART_COUNT 个纯净零件，准备总装！"

echo ""
echo "=================================================="
echo "🔥 步骤 3: 踢开 SCons，物理硬链接 (g++ 暴力电焊)"
echo "=================================================="
rm -f build/X86/gem5.opt

# 新增了 -lelf 库
g++ -o build/X86/gem5.opt \
    -Wl,--start-group @build/X86/all_objs.txt -Wl,--end-group \
    $(python3-config --ldflags --embed || python3-config --ldflags) \
    $(pkg-config --libs protobuf) \
    -lelf -lz -lrt -ldl -lpthread

echo ""
echo "=================================================="
if [ -f "build/X86/gem5.opt" ]; then
    echo "🎉 大功告成！Gem5.opt 战车已成功下线！"
    ls -lh build/X86/gem5.opt
else
    echo "❌ 链接失败！请检查上方报错。"
    exit 1
fi
echo "=================================================="
