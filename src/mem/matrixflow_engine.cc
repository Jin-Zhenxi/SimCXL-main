#include "mem/matrixflow_engine.hh"

#include <algorithm>
#include <cstring>
#include <string>

#include "base/logging.hh"
#include "base/trace.hh"
#include "debug/MatrixFlow.hh"

namespace gem5
{

namespace
{

inline uint64_t
divCeil(uint64_t a, uint64_t b)
{
    return (a + b - 1) / b;
}

} // namespace

MatrixFlowEngine::MatrixFlowEngine(const Params &p)
    : ClockedObject(p),
      dmaPort(*this, p.system),
      macArraySize(p.mac_array_size),
      dmaChunkSize(p.dma_chunk_size),
      computeLatencyPerOp(p.compute_latency_per_op),
      phase(Phase::Idle),
      tileABuffer(kMaxTileDim * kMaxTileDim * sizeof(uint32_t), 0),
      tileBBuffer(kMaxTileDim * kMaxTileDim * sizeof(uint32_t), 0),
      tileCBuffer(kMaxTileDim * kMaxTileDim * sizeof(uint32_t), 0),
      computeBusy(false),
      pendingDescAddr(0),
      pendingMatrixA(0),
      pendingMatrixB(0),
      pendingResult(0),
      pendingFlagAddr(0),
      pendingSize(0),
      pendingDesc(),
      completionFlagValue(1),
      fetchDescCompleteEvent(
          [this] { onFetchDescComplete(); }, name() + ".fetch_desc"),
      computeDoneEvent([this] { processComputeDone(); }, name() + ".compute"),
      writeFlagCompleteEvent(
          [this] { onWriteFlagComplete(); }, name() + ".write_flag"),
      stats(this)
{
    fetchARowEvents.reserve(kMaxTileDim);
    fetchBRowEvents.reserve(kMaxTileDim);
    writeCRowEvents.reserve(kMaxTileDim);
    for (int i = 0; i < kMaxTileDim; ++i) {
        fetchARowEvents.emplace_back(
            [this] { onFetchARowComplete(); },
            name() + ".fetchA_row" + std::to_string(i));
        fetchBRowEvents.emplace_back(
            [this] { onFetchBRowComplete(); },
            name() + ".fetchB_row" + std::to_string(i));
        writeCRowEvents.emplace_back(
            [this] { onWriteCRowComplete(); },
            name() + ".writeC_row" + std::to_string(i));
    }

    DPRINTF(MatrixFlow,
            "Create MatrixFlowEngine: mac_array_size=%u, dma_chunk_size=%uB, "
            "compute_latency_per_op=%llu cycles, maxTile=%d\n",
            macArraySize,
            dmaChunkSize,
            static_cast<unsigned long long>(computeLatencyPerOp),
            kMaxTileDim);
}

MatrixFlowEngine::EngineStats::EngineStats(statistics::Group *parent)
    : statistics::Group(parent),
      ADD_STAT(totalDmaBytesRead, statistics::units::Byte::get(),
               "Total bytes read from HDM by MatrixFlow DMA"),
      ADD_STAT(totalDmaBytesWritten, statistics::units::Byte::get(),
               "Total bytes written to HDM by MatrixFlow DMA"),
      ADD_STAT(totalComputeCycles, statistics::units::Cycle::get(),
               "Total modeled MatrixFlow compute cycles")
{
}

Port &
MatrixFlowEngine::getPort(const std::string &if_name, PortID idx)
{
    if (if_name == "dma_port") {
        return dmaPort;
    }

    return ClockedObject::getPort(if_name, idx);
}

void
MatrixFlowEngine::init()
{
    panic_if(!dmaPort.isConnected(),
             "MatrixFlowEngine %s dma_port is not connected.\n", name());

    DPRINTF(MatrixFlow,
            "Init done. tileA=%zuB tileB=%zuB tileC=%zuB dmaLine=%lluB "
            "dmaChunk=%uB\n",
            tileABuffer.size(), tileBBuffer.size(), tileCBuffer.size(),
            static_cast<unsigned long long>(dmaPort.lineSize()),
            dmaChunkSize);
}

uint64_t
MatrixFlowEngine::estimateTileCycles(
    uint32_t tileM, uint32_t tileN, uint32_t tileK) const
{
    if (tileM == 0 || tileN == 0 || tileK == 0) {
        return 1;
    }

    const uint64_t totalOps =
        static_cast<uint64_t>(tileM) * tileN * tileK;

    const uint64_t arrayThroughput = std::max<uint64_t>(
        1, static_cast<uint64_t>(macArraySize) *
               static_cast<uint64_t>(macArraySize));

    const uint64_t opChunks = divCeil(totalOps, arrayThroughput);
    const uint64_t totalCycles =
        std::max<uint64_t>(
            1, opChunks * static_cast<uint64_t>(computeLatencyPerOp));

    return totalCycles;
}

void
MatrixFlowEngine::resetContext()
{
    phase = Phase::Idle;
    ctx = GemmContext();
    computeBusy = false;
    pendingDescAddr = 0;
    pendingMatrixA = 0;
    pendingMatrixB = 0;
    pendingResult = 0;
    pendingFlagAddr = 0;
    pendingSize = 0;
    pendingDesc = Descriptor();
    completionFlagValue = 1;
    reqsIssuedA = reqsCompletedA = targetReqsA = 0;
    reqsIssuedB = reqsCompletedB = targetReqsB = 0;
    reqsIssuedC = reqsCompletedC = targetReqsC = 0;
}

void
MatrixFlowEngine::issueFetchDescriptor()
{
    phase = Phase::FetchDesc;

    pendingDesc = Descriptor{};
    DPRINTF(MatrixFlow,
            "DMA FetchDesc addr=%#llx bytes=%zu\n",
            static_cast<unsigned long long>(pendingDescAddr),
            sizeof(pendingDesc));

    stats.totalDmaBytesRead += sizeof(pendingDesc);
    dmaPort.dmaAction(
        MemCmd::ReadReq, pendingDescAddr, sizeof(pendingDesc),
        &fetchDescCompleteEvent, dmaChunkSize,
        reinterpret_cast<uint8_t *>(&pendingDesc), 0);
}

void
MatrixFlowEngine::onFetchDescComplete()
{
    panic_if(pendingDesc.size == 0,
             "%s: descriptor fetched with invalid size=0 at %#llx\n",
             name(), static_cast<unsigned long long>(pendingDescAddr));

    ctx.baseA = pendingDesc.addrA;
    ctx.baseB = pendingDesc.addrB;
    ctx.baseC = pendingDesc.addrC;
    ctx.flagAddr = pendingDesc.flagAddr;
    ctx.size = pendingDesc.size;
    ctx.completionValue =
        pendingDesc.completionValue != 0 ? pendingDesc.completionValue : 1;
    ctx.elemBytes = sizeof(uint32_t);
    const uint32_t cap = static_cast<uint32_t>(kMaxTileDim);
    ctx.tileM = std::min(cap, ctx.size);
    ctx.tileN = std::min(cap, ctx.size);
    ctx.tileK = std::min(cap, ctx.size);
    ctx.i = 0;
    ctx.j = 0;
    ctx.k = 0;

    pendingMatrixA = ctx.baseA;
    pendingMatrixB = ctx.baseB;
    pendingResult = ctx.baseC;
    pendingFlagAddr = ctx.flagAddr;
    pendingSize = static_cast<int>(ctx.size);
    completionFlagValue = ctx.completionValue;

    DPRINTF(MatrixFlow,
            "Descriptor loaded: A=%#llx B=%#llx C=%#llx flag=%#llx size=%u "
            "completion=%#llx tile=(%u,%u,%u)\n",
            static_cast<unsigned long long>(ctx.baseA),
            static_cast<unsigned long long>(ctx.baseB),
            static_cast<unsigned long long>(ctx.baseC),
            static_cast<unsigned long long>(ctx.flagAddr),
            ctx.size,
            static_cast<unsigned long long>(ctx.completionValue),
            ctx.tileM, ctx.tileN, ctx.tileK);

    prepareOutputTile();
    issueFetchATile();
}

void
MatrixFlowEngine::prepareOutputTile()
{
    ctx.curTileM = std::min(ctx.tileM, ctx.size - ctx.i);
    ctx.curTileN = std::min(ctx.tileN, ctx.size - ctx.j);
    ctx.curTileK = std::min(ctx.tileK, ctx.size - ctx.k);

    if (ctx.k == 0) {
        std::fill(tileCBuffer.begin(), tileCBuffer.end(), 0);
    }
}

void
MatrixFlowEngine::issueFetchATile()
{
    phase = Phase::FetchA;

    panic_if(ctx.curTileM > static_cast<uint32_t>(kMaxTileDim),
             "%s: curTileM=%u exceeds max %d\n", name(), ctx.curTileM,
             kMaxTileDim);

    if (ctx.curTileM == 0) {
        issueFetchBTile();
        return;
    }

    reqsIssuedA = 0;
    reqsCompletedA = 0;
    targetReqsA = ctx.curTileM;
    trySendMoreA();
}

void
MatrixFlowEngine::trySendMoreA()
{
    const Addr rowBytes =
        static_cast<Addr>(ctx.curTileK) * ctx.elemBytes;

    while (reqsIssuedA < targetReqsA &&
           (reqsIssuedA - reqsCompletedA) < kMaxInFlight) {
        const uint32_t r = reqsIssuedA;
        const Addr rowAddr = ctx.baseA +
            ((static_cast<Addr>(ctx.i + r) * ctx.size + ctx.k) *
             ctx.elemBytes);
        auto *dst = tileABuffer.data() + r * rowBytes;

        DPRINTF(MatrixFlow,
                "DMA FetchA row=%u addr=%#llx bytes=%llu (tile i=%u j=%u k=%u)\n",
                r, static_cast<unsigned long long>(rowAddr),
                static_cast<unsigned long long>(rowBytes), ctx.i, ctx.j,
                ctx.k);

        stats.totalDmaBytesRead += rowBytes;
        dmaPort.dmaAction(MemCmd::ReadReq, rowAddr, rowBytes,
                          &fetchARowEvents[r], dmaChunkSize, dst, 0);
        ++reqsIssuedA;
    }
}

void
MatrixFlowEngine::issueFetchBTile()
{
    phase = Phase::FetchB;

    panic_if(ctx.curTileK > static_cast<uint32_t>(kMaxTileDim),
             "%s: curTileK=%u exceeds max %d\n", name(), ctx.curTileK,
             kMaxTileDim);

    if (ctx.curTileK == 0) {
        launchComputeTile();
        return;
    }

    reqsIssuedB = 0;
    reqsCompletedB = 0;
    targetReqsB = ctx.curTileK;
    trySendMoreB();
}

void
MatrixFlowEngine::trySendMoreB()
{
    const Addr rowBytes =
        static_cast<Addr>(ctx.curTileN) * ctx.elemBytes;

    while (reqsIssuedB < targetReqsB &&
           (reqsIssuedB - reqsCompletedB) < kMaxInFlight) {
        const uint32_t r = reqsIssuedB;
        const Addr rowAddr = ctx.baseB +
            ((static_cast<Addr>(ctx.k + r) * ctx.size + ctx.j) *
             ctx.elemBytes);
        auto *dst = tileBBuffer.data() + r * rowBytes;

        DPRINTF(MatrixFlow,
                "DMA FetchB row=%u addr=%#llx bytes=%llu (tile i=%u j=%u k=%u)\n",
                r, static_cast<unsigned long long>(rowAddr),
                static_cast<unsigned long long>(rowBytes), ctx.i, ctx.j,
                ctx.k);

        stats.totalDmaBytesRead += rowBytes;
        dmaPort.dmaAction(MemCmd::ReadReq, rowAddr, rowBytes,
                          &fetchBRowEvents[r], dmaChunkSize, dst, 0);
        ++reqsIssuedB;
    }
}

void
MatrixFlowEngine::issueWriteCTile()
{
    phase = Phase::WriteC;

    panic_if(ctx.curTileM > static_cast<uint32_t>(kMaxTileDim),
             "%s: issueWriteC curTileM=%u exceeds max %d\n", name(),
             ctx.curTileM, kMaxTileDim);

    if (ctx.curTileM == 0) {
        advanceTile();
        return;
    }

    reqsIssuedC = 0;
    reqsCompletedC = 0;
    targetReqsC = ctx.curTileM;
    trySendMoreC();
}

void
MatrixFlowEngine::trySendMoreC()
{
    const Addr rowBytes =
        static_cast<Addr>(ctx.curTileN) * ctx.elemBytes;

    while (reqsIssuedC < targetReqsC &&
           (reqsIssuedC - reqsCompletedC) < kMaxInFlight) {
        const uint32_t r = reqsIssuedC;
        const Addr rowAddr = ctx.baseC +
            ((static_cast<Addr>(ctx.i + r) * ctx.size + ctx.j) *
             ctx.elemBytes);
        auto *src = tileCBuffer.data() + r * rowBytes;

        DPRINTF(MatrixFlow,
                "DMA WriteC row=%u addr=%#llx bytes=%llu (tile i=%u j=%u k=%u)\n",
                r, static_cast<unsigned long long>(rowAddr),
                static_cast<unsigned long long>(rowBytes), ctx.i, ctx.j,
                ctx.k);

        stats.totalDmaBytesWritten += rowBytes;
        dmaPort.dmaAction(MemCmd::WriteReq, rowAddr, rowBytes,
                          &writeCRowEvents[r], dmaChunkSize, src, 0);
        ++reqsIssuedC;
    }
}

void
MatrixFlowEngine::issueWriteFlag()
{
    phase = Phase::WriteFlag;

    DPRINTF(MatrixFlow,
            "DMA WriteFlag addr=%#llx value=%llu\n",
            static_cast<unsigned long long>(ctx.flagAddr),
            static_cast<unsigned long long>(completionFlagValue));

    stats.totalDmaBytesWritten += sizeof(completionFlagValue);
    dmaPort.dmaAction(
        MemCmd::WriteReq, ctx.flagAddr, sizeof(completionFlagValue),
        &writeFlagCompleteEvent, dmaChunkSize,
        reinterpret_cast<uint8_t *>(&completionFlagValue), 0);
}

void
MatrixFlowEngine::onFetchARowComplete()
{
    panic_if(reqsCompletedA >= targetReqsA,
             "%s: onFetchARowComplete reqsCompletedA=%u >= targetReqsA=%u\n",
             name(), reqsCompletedA, targetReqsA);
    ++reqsCompletedA;
    if (reqsCompletedA == targetReqsA) {
        issueFetchBTile();
    } else {
        trySendMoreA();
    }
}

void
MatrixFlowEngine::onFetchBRowComplete()
{
    panic_if(reqsCompletedB >= targetReqsB,
             "%s: onFetchBRowComplete reqsCompletedB=%u >= targetReqsB=%u\n",
             name(), reqsCompletedB, targetReqsB);
    ++reqsCompletedB;
    if (reqsCompletedB == targetReqsB) {
        launchComputeTile();
    } else {
        trySendMoreB();
    }
}

void
MatrixFlowEngine::onWriteCRowComplete()
{
    panic_if(reqsCompletedC >= targetReqsC,
             "%s: onWriteCRowComplete reqsCompletedC=%u >= targetReqsC=%u\n",
             name(), reqsCompletedC, targetReqsC);
    ++reqsCompletedC;
    if (reqsCompletedC == targetReqsC) {
        advanceTile();
    } else {
        trySendMoreC();
    }
}

void
MatrixFlowEngine::launchComputeTile()
{
    phase = Phase::Compute;

    const uint64_t tileCycles = estimateTileCycles(
        ctx.curTileM, ctx.curTileN, ctx.curTileK);
    const Tick doneAt = curTick() + clockPeriod() * tileCycles;

    DPRINTF(MatrixFlow,
            "LaunchCompute tile i=%u j=%u k=%u dims=(%u,%u,%u) "
            "cycles=%llu done@%llu\n",
            ctx.i, ctx.j, ctx.k, ctx.curTileM, ctx.curTileN, ctx.curTileK,
            static_cast<unsigned long long>(tileCycles),
            static_cast<unsigned long long>(doneAt));

    stats.totalComputeCycles += tileCycles;
    schedule(computeDoneEvent, doneAt);
}

void
MatrixFlowEngine::accumulateCurrentTile()
{
    auto *tileA = reinterpret_cast<const uint32_t *>(tileABuffer.data());
    auto *tileB = reinterpret_cast<const uint32_t *>(tileBBuffer.data());
    auto *tileC = reinterpret_cast<uint32_t *>(tileCBuffer.data());

    for (uint32_t m = 0; m < ctx.curTileM; ++m) {
        for (uint32_t n = 0; n < ctx.curTileN; ++n) {
            uint64_t acc = tileC[m * ctx.curTileN + n];
            for (uint32_t kk = 0; kk < ctx.curTileK; ++kk) {
                acc += static_cast<uint64_t>(
                    tileA[m * ctx.curTileK + kk]) *
                    static_cast<uint64_t>(tileB[kk * ctx.curTileN + n]);
            }
            tileC[m * ctx.curTileN + n] = static_cast<uint32_t>(acc);
        }
    }
}

void
MatrixFlowEngine::advanceTile()
{
    ctx.k = 0;
    if (ctx.j + ctx.curTileN < ctx.size) {
        ctx.j += ctx.curTileN;
    } else {
        ctx.j = 0;
        ctx.i += ctx.curTileM;
    }

    if (ctx.i >= ctx.size) {
        DPRINTF(MatrixFlow,
                "MatrixFlow DMA complete: A=%#llx B=%#llx C=%#llx size=%u "
                "complete@%llu\n",
                static_cast<unsigned long long>(pendingMatrixA),
                static_cast<unsigned long long>(pendingMatrixB),
                static_cast<unsigned long long>(pendingResult),
                pendingSize,
                static_cast<unsigned long long>(curTick()));
        issueWriteFlag();
        return;
    }

    prepareOutputTile();
    issueFetchATile();
}

void
MatrixFlowEngine::startMatrixCompute(
    Addr descriptorAddr)
{
    if (computeBusy || computeDoneEvent.scheduled()) {
        warn("%s: compute already in flight, reject new request.\n", name());
        return;
    }

    computeBusy = true;
    pendingDescAddr = descriptorAddr;

    DPRINTF(MatrixFlow,
            "startMatrixCompute: descriptor=%#llx\n",
            static_cast<unsigned long long>(descriptorAddr));

    issueFetchDescriptor();
}

void
MatrixFlowEngine::processComputeDone()
{
    accumulateCurrentTile();

    DPRINTF(MatrixFlow,
            "processComputeDone: tile i=%u j=%u k=%u dims=(%u,%u,%u) "
            "A=%#llx B=%#llx C=%#llx now=%llu\n",
            ctx.i, ctx.j, ctx.k, ctx.curTileM, ctx.curTileN, ctx.curTileK,
            static_cast<unsigned long long>(pendingMatrixA),
            static_cast<unsigned long long>(pendingMatrixB),
            static_cast<unsigned long long>(pendingResult),
            static_cast<unsigned long long>(curTick()));

    if (ctx.k + ctx.curTileK < ctx.size) {
        ctx.k += ctx.curTileK;
        prepareOutputTile();
        issueFetchATile();
        return;
    }

    issueWriteCTile();
}

void
MatrixFlowEngine::onWriteFlagComplete()
{
    DPRINTF(MatrixFlow,
            "MatrixFlow completion flag written: flag=%#llx value=%llu "
            "dmaRead=%llu dmaWrite=%llu computeCycles=%llu\n",
            static_cast<unsigned long long>(pendingFlagAddr),
            static_cast<unsigned long long>(completionFlagValue),
            static_cast<unsigned long long>(stats.totalDmaBytesRead.value()),
            static_cast<unsigned long long>(stats.totalDmaBytesWritten.value()),
            static_cast<unsigned long long>(stats.totalComputeCycles.value()));
    resetContext();
}

void
MatrixFlowEngine::startup()
{
    // 脚手架已拆除：不再自触发，由 Host CPU 通过 CXL 控制器 GEMM 触发地址调用
    // DPRINTF(MatrixFlow, ">>> TEST: Auto-triggering compute in startup()...\n");
    // startMatrixCompute(0x1000, 0x2000, 0x3000, 1024);
}

} // namespace gem5
