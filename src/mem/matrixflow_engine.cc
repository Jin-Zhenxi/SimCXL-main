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

inline Addr
roundDownAddr(Addr value, Addr granularity)
{
    return value & ~(granularity - 1);
}

inline Addr
roundUpAddr(Addr value, Addr granularity)
{
    return (value + granularity - 1) & ~(granularity - 1);
}

} // namespace

MatrixFlowEngine::MatrixFlowEngine(const Params &p)
    : ClockedObject(p),
      dmaPort(*this, p.system),
      macArraySize(p.mac_array_size),
      computeLatencyPerOp(p.compute_latency_per_op),
      minReadRequestBytes(std::max<Addr>(1, p.min_read_request_bytes)),
      readBouncePitch(
          2 * std::max<Addr>(std::max<Addr>(1, p.min_read_request_bytes), 64)),
      nextPrefetchMode(p.next_prefetch_mode),
      nextPrefetchTrigger(p.next_prefetch_trigger),
      nextPrefetchRowsAConfig(p.next_prefetch_rows_a),
      nextPrefetchRowsBConfig(p.next_prefetch_rows_b),
      carryOverMaxRowsConfig(p.carry_over_max_rows),
      carryOverInheritInflight(p.carry_over_inherit_inflight),
      holeFillLeadRowsConfig(p.hole_fill_lead_rows),
      writeCOverlapBIssueBudgetRowsConfig(p.writec_overlap_b_issue_budget_rows),
      phase(Phase::Idle),
      tileABuffer(kMaxTileDim * kMaxTileDim * sizeof(uint32_t), 0),
      tileBBuffer(kMaxTileDim * kMaxTileDim * sizeof(uint32_t), 0),
      tileCBuffer(kMaxTileDim * kMaxTileDim * sizeof(uint32_t), 0),
      nextTileABuffer(kMaxTileDim * kMaxTileDim * sizeof(uint32_t), 0),
      nextTileBBuffer(kMaxTileDim * kMaxTileDim * sizeof(uint32_t), 0),
      nextOutputTileBBuffer(kMaxTileDim * kMaxTileDim * sizeof(uint32_t), 0),
      fetchABounceBuffer(kMaxTileDim * readBouncePitch, 0),
      fetchBBounceBuffer(kMaxTileDim * readBouncePitch, 0),
      nextFetchABounceBuffer(kMaxTileDim * readBouncePitch, 0),
      nextFetchBBounceBuffer(kMaxTileDim * readBouncePitch, 0),
      nextOutputFetchBBounceBuffer(kMaxTileDim * readBouncePitch, 0),
      fetchABounceReqAddr(kMaxTileDim, 0),
      fetchABounceReqBytes(kMaxTileDim, 0),
      fetchABounceRowBytes(kMaxTileDim, 0),
      fetchABounceOffset(kMaxTileDim, 0),
      fetchABounceActive(kMaxTileDim, false),
      fetchBBounceReqAddr(kMaxTileDim, 0),
      fetchBBounceReqBytes(kMaxTileDim, 0),
      fetchBBounceRowBytes(kMaxTileDim, 0),
      fetchBBounceOffset(kMaxTileDim, 0),
      fetchBBounceActive(kMaxTileDim, false),
      nextFetchABounceReqAddr(kMaxTileDim, 0),
      nextFetchABounceReqBytes(kMaxTileDim, 0),
      nextFetchABounceRowBytes(kMaxTileDim, 0),
      nextFetchABounceOffset(kMaxTileDim, 0),
      nextFetchABounceActive(kMaxTileDim, false),
      nextFetchBBounceReqAddr(kMaxTileDim, 0),
      nextFetchBBounceReqBytes(kMaxTileDim, 0),
      nextFetchBBounceRowBytes(kMaxTileDim, 0),
      nextFetchBBounceOffset(kMaxTileDim, 0),
      nextFetchBBounceActive(kMaxTileDim, false),
      nextOutputFetchBBounceReqAddr(kMaxTileDim, 0),
      nextOutputFetchBBounceReqBytes(kMaxTileDim, 0),
      nextOutputFetchBBounceRowBytes(kMaxTileDim, 0),
      nextOutputFetchBBounceOffset(kMaxTileDim, 0),
      nextOutputFetchBBounceActive(kMaxTileDim, false),
      nextFetchARowGeneration(kMaxTileDim, 0),
      nextFetchBRowGeneration(kMaxTileDim, 0),
      nextOutputFetchBRowGeneration(kMaxTileDim, 0),
      currentBRowState(kMaxTileDim, BRowState::Empty),
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
    nextFetchARowEvents.reserve(kMaxTileDim);
    nextFetchBRowEvents.reserve(kMaxTileDim);
    nextOutputFetchBRowEvents.reserve(kMaxTileDim);
    for (int i = 0; i < kMaxTileDim; ++i) {
        fetchARowEvents.emplace_back(
            [this, i] { onFetchARowComplete(i); },
            name() + ".fetchA_row" + std::to_string(i));
        fetchBRowEvents.emplace_back(
            [this, i] { onFetchBRowComplete(i); },
            name() + ".fetchB_row" + std::to_string(i));
        writeCRowEvents.emplace_back(
            [this, i] { onWriteCRowComplete(i); },
            name() + ".writeC_row" + std::to_string(i));
        nextFetchARowEvents.emplace_back(
            [this, i] { onNextFetchARowComplete(i); },
            name() + ".next_fetchA_row" + std::to_string(i));
        nextFetchBRowEvents.emplace_back(
            [this, i] { onNextFetchBRowComplete(i); },
            name() + ".next_fetchB_row" + std::to_string(i));
        nextOutputFetchBRowEvents.emplace_back(
            [this, i] { onNextOutputFetchBRowComplete(i); },
            name() + ".next_output_fetchB_row" + std::to_string(i));
    }

    DPRINTF(MatrixFlow,
            "Create MatrixFlowEngine: mac_array_size=%u, "
            "compute_latency_per_op=%llu cycles, maxTile=%d, "
            "minReadReq=%lluB nextPrefetchMode=%s trigger=%s rowsA=%u rowsB=%u\n",
            macArraySize,
            static_cast<unsigned long long>(computeLatencyPerOp),
            kMaxTileDim,
            static_cast<unsigned long long>(minReadRequestBytes),
            nextPrefetchMode.c_str(),
            nextPrefetchTrigger.c_str(),
            nextPrefetchRowsAConfig,
            nextPrefetchRowsBConfig);
}

MatrixFlowEngine::EngineStats::EngineStats(statistics::Group *parent)
    : statistics::Group(parent),
      ADD_STAT(totalDmaBytesRead, statistics::units::Byte::get(),
               "Total bytes read from HDM by MatrixFlow DMA"),
      ADD_STAT(totalDmaBytesWritten, statistics::units::Byte::get(),
               "Total bytes written to HDM by MatrixFlow DMA"),
      ADD_STAT(totalComputeCycles, statistics::units::Cycle::get(),
               "Total modeled MatrixFlow compute cycles"),
      ADD_STAT(nextPrefetchIssueCount, statistics::units::Count::get(),
               "Number of next-tile prefetch attempts issued"),
      ADD_STAT(nextKPrefetchIssueCount, statistics::units::Count::get(),
               "Number of next-k prefetch attempts issued"),
      ADD_STAT(nextOutputPrefetchIssueCount, statistics::units::Count::get(),
               "Number of next-output-tile prefetch attempts issued"),
      ADD_STAT(nextPrefetchHitCount, statistics::units::Count::get(),
               "Number of next-tile prefetch hits consumed"),
      ADD_STAT(nextOutputPrefetchHitCount, statistics::units::Count::get(),
               "Number of next-output-tile prefetch hits consumed"),
      ADD_STAT(nextPrefetchFallbackCount, statistics::units::Count::get(),
               "Number of tiles that fell back to the original fetch path"),
      ADD_STAT(nextOutputPrefetchFallbackCount, statistics::units::Count::get(),
               "Number of output-tile prefetch opportunities that fell back"),
      ADD_STAT(nextPrefetchLateCompletionCount, statistics::units::Count::get(),
               "Number of stale next-prefetch completions ignored by generation"),
      ADD_STAT(nextOutputPrefetchLateCompletionCount, statistics::units::Count::get(),
               "Number of stale next-output prefetch completions ignored by generation"),
      ADD_STAT(nextPrefetchDiscardCount, statistics::units::Count::get(),
               "Number of in-flight next-prefetch attempts discarded"),
      ADD_STAT(prefetchedBRowsConsumed, statistics::units::Count::get(),
               "Number of prefetched B rows consumed by the current tile"),
      ADD_STAT(fallbackBRowsFetched, statistics::units::Count::get(),
               "Number of B rows fetched by the original path after prefetch fallback"),
      ADD_STAT(nextOutputPrefetchRowsIssued, statistics::units::Count::get(),
               "Number of B rows issued for next-output prefetch"),
      ADD_STAT(nextKPrefetchRowsIssued, statistics::units::Count::get(),
               "Number of B rows issued for next-k prefetch"),
      ADD_STAT(nextOutputPrefetchDeferCount, statistics::units::Count::get(),
               "Number of arbitration points where next-output prefetch wanted rows but had no shared credits"),
      ADD_STAT(nextOutputHeadstartCycles, statistics::units::Cycle::get(),
               "Cycles of headstart between first next-output issue and boundary hit/fallback"),
      ADD_STAT(nextOutputFirstIssueToBoundaryCycles, statistics::units::Cycle::get(),
               "Cycles between first next-output issue and tile-boundary transition"),
      ADD_STAT(nextOutputRowsReadyAtBoundary, statistics::units::Count::get(),
               "How many next-output B rows were already ready at the tile boundary"),
      ADD_STAT(nextOutputConsumedBeforeFallbackRows, statistics::units::Count::get(),
               "How many next-output B rows were consumed before falling back to normal fetch"),
      ADD_STAT(carryOverRowsAtBoundary, statistics::units::Count::get(),
               "How many next-output B rows were already ready and carried over at tile boundary"),
      ADD_STAT(carryOverInflightRowsAtBoundary, statistics::units::Count::get(),
               "How many next-output B rows were still in flight and carried over at tile boundary"),
      ADD_STAT(carryOverRowsConsumedPostBoundary, statistics::units::Count::get(),
               "How many carried-over next-output B rows completed post-boundary and served the current tile"),
      ADD_STAT(normalFetchHoleRows, statistics::units::Count::get(),
               "How many B rows the normal path had to fetch as holes after carry-over"),
      ADD_STAT(duplicateBRowFetchAvoided, statistics::units::Count::get(),
               "How many duplicate current-tile B row fetches were avoided due to carry-over state"),
      ADD_STAT(duplicateBRowFetchDetected, statistics::units::Count::get(),
               "How many duplicate current-tile B row fetch attempts were detected"),
      ADD_STAT(carryOverLateCompletionCount, statistics::units::Count::get(),
               "How many carried-over next-output B completions arrived stale or after state teardown"),
      ADD_STAT(normalFetchDeferredByCarry, statistics::units::Count::get(),
               "How many times normal fetch stopped because hole-fill lead policy deferred more requests"),
      ADD_STAT(writeCOverlapCycles, statistics::units::Cycle::get(),
               "Cycles spent in WriteC tiles where overlap with later B/next-output traffic was enabled"),
      ADD_STAT(writeCOverlapEnabledCount, statistics::units::Count::get(),
               "How many WriteC tiles enabled overlap servicing for later B/next-output traffic"),
      ADD_STAT(writeCOverlapSuccessCount, statistics::units::Count::get(),
               "How many WriteC overlap service points observed actual B/next-output progress"),
      ADD_STAT(nextOutputProgressDuringWriteC, statistics::units::Count::get(),
               "How many next-output B rows completed while the engine was in WriteC overlap window"),
      ADD_STAT(bRowsIssuedDuringWriteC, statistics::units::Count::get(),
               "How many prefetch B rows were issued while the engine was in WriteC overlap window"),
      ADD_STAT(writeCBlockedBIssueCount, statistics::units::Count::get(),
               "How many WriteC overlap service points saw pending B work but no issue progress")
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
            "Init done. tileA=%zuB tileB=%zuB tileC=%zuB dmaLine=%lluB\n",
            tileABuffer.size(), tileBBuffer.size(), tileCBuffer.size(),
            static_cast<unsigned long long>(dmaPort.lineSize()));
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

std::tuple<Addr, Addr, Addr>
MatrixFlowEngine::planReadRequest(Addr rowAddr, Addr rowBytes) const
{
    if (rowBytes >= minReadRequestBytes) {
        return std::make_tuple(rowAddr, rowBytes, 0);
    }

    const Addr granularity = std::max<Addr>(
        minReadRequestBytes, std::max<Addr>(dmaPort.lineSize(), 1));
    const Addr reqAddr = roundDownAddr(rowAddr, granularity);
    const Addr reqEnd = roundUpAddr(rowAddr + rowBytes, granularity);
    const Addr reqBytes = std::max<Addr>(granularity, reqEnd - reqAddr);
    const Addr offset = rowAddr - reqAddr;

    panic_if(reqBytes > readBouncePitch,
             "%s: planned read reqBytes=%llu exceeds bounce pitch=%llu\n",
             name(), static_cast<unsigned long long>(reqBytes),
             static_cast<unsigned long long>(readBouncePitch));

    return std::make_tuple(reqAddr, reqBytes, offset);
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
    nextPrefetchBDrainActive = false;
    nextPrefetchBDrainGeneration = 0;
    nextPrefetchBDrainOutstanding = 0;
    nextOutputPrefetchBDrainActive = false;
    nextOutputPrefetchBDrainGeneration = 0;
    nextOutputPrefetchBDrainOutstanding = 0;
    currentPrefetchedBValid = false;
    currentPrefetchedBRows = 0;
    carryOverBActive = false;
    carryOverBGeneration = 0;
    carryOverBRowsIssued = 0;
    carryOverBRowsCompleted = 0;
    carryOverBRowsTarget = 0;
    writeCOverlapWindowActive = false;
    writeCOverlapMadeProgress = false;
    writeCStartTick = 0;
    writeCPrevPrefetchIssuedB = 0;
    writeCPrevNextOutputCompletedB = 0;
    writeCBRowsIssued = 0;
    std::fill(currentBRowState.begin(), currentBRowState.end(),
              BRowState::Empty);
    clearNextTilePrefetch();
    clearNextOutputTilePrefetch();
}

bool
MatrixFlowEngine::hasNextKTile() const
{
    return ctx.k + ctx.curTileK < ctx.size;
}

bool
MatrixFlowEngine::hasNextOutputTile() const
{
    return ctx.j + ctx.curTileN < ctx.size || ctx.i + ctx.curTileM < ctx.size;
}

bool
MatrixFlowEngine::prefetchEnabled() const
{
    return nextPrefetchMode != "none";
}

bool
MatrixFlowEngine::prefetchIncludesA() const
{
    return nextPrefetchMode == "a_b";
}

MatrixFlowEngine::GemmContext
MatrixFlowEngine::buildNextKContext() const
{
    GemmContext next = ctx;
    next.k += ctx.curTileK;
    next.curTileM = std::min(next.tileM, next.size - next.i);
    next.curTileN = std::min(next.tileN, next.size - next.j);
    next.curTileK = std::min(next.tileK, next.size - next.k);
    return next;
}

MatrixFlowEngine::GemmContext
MatrixFlowEngine::buildNextOutputContext() const
{
    GemmContext next = ctx;
    next.k = 0;
    if (ctx.j + ctx.curTileN < ctx.size) {
        next.j = ctx.j + ctx.curTileN;
        next.i = ctx.i;
    } else {
        next.j = 0;
        next.i = ctx.i + ctx.curTileM;
    }
    next.curTileM = std::min(next.tileM, next.size - next.i);
    next.curTileN = std::min(next.tileN, next.size - next.j);
    next.curTileK = std::min(next.tileK, next.size - next.k);
    return next;
}

void
MatrixFlowEngine::clearNextTilePrefetch()
{
    nextPrefetchIssuedA = false;
    nextPrefetchIssuedB = false;
    nextPrefetchReadyA = false;
    nextPrefetchReadyB = false;
    nextPrefetchValid = false;
    nextPrefetchIsOutputTile = false;
    nextCtx = GemmContext();
    nextPrefetchReqsIssuedA = nextPrefetchRowsACompleted = nextPrefetchTargetA = 0;
    nextPrefetchReqsIssuedB = nextPrefetchRowsBCompleted = nextPrefetchTargetB = 0;
    activeNextPrefetchGeneration = 0;
}

void
MatrixFlowEngine::clearNextOutputTilePrefetch()
{
    nextOutputPrefetchValid = false;
    nextOutputPrefetchReadyB = false;
    nextOutputCtx = GemmContext();
    nextOutputPrefetchReqsIssuedB = 0;
    nextOutputPrefetchRowsBCompleted = 0;
    nextOutputPrefetchTargetB = 0;
    activeNextOutputPrefetchGeneration = 0;
    nextOutputFirstIssueTick = 0;
}

bool
MatrixFlowEngine::nextPrefetchMatchesCurrentTile() const
{
    return nextPrefetchValid &&
           nextCtx.i == ctx.i &&
           nextCtx.j == ctx.j &&
           nextCtx.k == ctx.k;
}

bool
MatrixFlowEngine::nextOutputPrefetchMatchesCurrentTile() const
{
    return nextOutputPrefetchValid &&
           nextOutputCtx.i == ctx.i &&
           nextOutputCtx.j == ctx.j &&
           nextOutputCtx.k == ctx.k;
}

void
MatrixFlowEngine::applyPrefetchedNextTile()
{
    const uint32_t prefetched_rows =
        std::min(nextPrefetchRowsBCompleted, nextCtx.curTileK);
    std::swap(tileBBuffer, nextTileBBuffer);
    currentPrefetchedBValid = prefetched_rows > 0;
    currentPrefetchedBRows = prefetched_rows;
    stats.nextPrefetchHitCount++;
    stats.prefetchedBRowsConsumed += prefetched_rows;
    if (nextPrefetchIsOutputTile) {
        stats.nextOutputPrefetchHitCount++;
    }
    DPRINTF(MatrixFlow,
            "next-prefetch hit (%s): tile i=%u j=%u k=%u gen=%llu "
            "prefetchedBRows=%u/%u\n",
            nextPrefetchIsOutputTile ? "next_output" : "next_k",
            ctx.i, ctx.j, ctx.k,
            static_cast<unsigned long long>(activeNextPrefetchGeneration),
            prefetched_rows, nextCtx.curTileK);
    clearNextTilePrefetch();
}

void
MatrixFlowEngine::applyPrefetchedNextOutputTile()
{
    const uint32_t prefetched_rows =
        std::min(nextOutputPrefetchRowsBCompleted, nextOutputCtx.curTileK);
    applyPrefetchedNextOutputRows(prefetched_rows, true);
    clearNextOutputTilePrefetch();
}

void
MatrixFlowEngine::applyPrefetchedNextOutputRows(uint32_t prefetched_rows,
                                                bool full_hit)
{
    std::swap(tileBBuffer, nextOutputTileBBuffer);
    currentPrefetchedBValid = prefetched_rows > 0;
    currentPrefetchedBRows = prefetched_rows;
    stats.prefetchedBRowsConsumed += prefetched_rows;
    if (full_hit) {
        stats.nextPrefetchHitCount++;
        stats.nextOutputPrefetchHitCount++;
        DPRINTF(MatrixFlow,
                "next-output prefetch hit: tile i=%u j=%u k=%u gen=%llu "
                "prefetchedBRows=%u/%u\n",
                ctx.i, ctx.j, ctx.k,
                static_cast<unsigned long long>(activeNextOutputPrefetchGeneration),
                prefetched_rows, nextOutputCtx.curTileK);
    } else {
        DPRINTF(MatrixFlow,
                "next-output prefetch partial headstart: tile i=%u j=%u k=%u "
                "gen=%llu prefetchedBRows=%u/%u\n",
                ctx.i, ctx.j, ctx.k,
                static_cast<unsigned long long>(activeNextOutputPrefetchGeneration),
                prefetched_rows, nextOutputCtx.curTileK);
    }
}

void
MatrixFlowEngine::discardNextTilePrefetch()
{
    if (nextPrefetchValid) {
        const uint32_t outstanding_b =
            nextPrefetchReqsIssuedB - nextPrefetchRowsBCompleted;
        stats.nextPrefetchDiscardCount++;
        DPRINTF(MatrixFlow,
                "discard next-prefetch (%s): tile i=%u j=%u k=%u gen=%llu "
                "readyA=%d readyB=%d completedA=%u/%u completedB=%u/%u "
                "outstandingB=%u\n",
                nextPrefetchIsOutputTile ? "next_output" : "next_k",
                nextCtx.i, nextCtx.j, nextCtx.k,
                static_cast<unsigned long long>(activeNextPrefetchGeneration),
                nextPrefetchReadyA, nextPrefetchReadyB,
                nextPrefetchRowsACompleted, nextPrefetchTargetA,
                nextPrefetchRowsBCompleted, nextPrefetchTargetB,
                outstanding_b);
        if (outstanding_b > 0) {
            nextPrefetchBDrainActive = true;
            nextPrefetchBDrainGeneration = activeNextPrefetchGeneration;
            nextPrefetchBDrainOutstanding = outstanding_b;
        }
    }
    clearNextTilePrefetch();
}

void
MatrixFlowEngine::discardNextOutputTilePrefetch()
{
    if (nextOutputPrefetchValid) {
        const uint32_t outstanding_b =
            nextOutputPrefetchReqsIssuedB - nextOutputPrefetchRowsBCompleted;
        stats.nextPrefetchDiscardCount++;
        DPRINTF(MatrixFlow,
                "discard next-output prefetch: tile i=%u j=%u k=%u gen=%llu "
                "readyB=%d completedB=%u/%u outstandingB=%u\n",
                nextOutputCtx.i, nextOutputCtx.j, nextOutputCtx.k,
                static_cast<unsigned long long>(activeNextOutputPrefetchGeneration),
                nextOutputPrefetchReadyB,
                nextOutputPrefetchRowsBCompleted, nextOutputPrefetchTargetB,
                outstanding_b);
        if (outstanding_b > 0) {
            nextOutputPrefetchBDrainActive = true;
            nextOutputPrefetchBDrainGeneration =
                activeNextOutputPrefetchGeneration;
            nextOutputPrefetchBDrainOutstanding = outstanding_b;
        }
    }
    clearNextOutputTilePrefetch();
}

void
MatrixFlowEngine::startPrefetchNextTile(const GemmContext &prefetchCtx,
                                        bool isOutputTile)
{
    discardNextTilePrefetch();

    nextCtx = prefetchCtx;
    nextPrefetchValid = true;
    nextPrefetchIsOutputTile = isOutputTile;
    activeNextPrefetchGeneration = ++nextPrefetchGeneration;
    nextPrefetchReqsIssuedA = 0;
    nextPrefetchRowsACompleted = 0;
    nextPrefetchTargetA = prefetchIncludesA()
        ? (nextPrefetchRowsAConfig == 0
            ? nextCtx.curTileM
            : std::min(nextCtx.curTileM, nextPrefetchRowsAConfig))
        : 0;
    nextPrefetchReqsIssuedB = 0;
    nextPrefetchRowsBCompleted = 0;
    nextPrefetchTargetB = nextPrefetchRowsBConfig == 0
        ? nextCtx.curTileK
        : std::min(nextCtx.curTileK, nextPrefetchRowsBConfig);
    nextPrefetchIssuedA = nextPrefetchTargetA > 0;
    nextPrefetchIssuedB = nextPrefetchTargetB > 0;
    nextPrefetchReadyA = true;
    nextPrefetchReadyB = nextPrefetchTargetB == 0;
    stats.nextPrefetchIssueCount++;
    if (isOutputTile) {
        stats.nextOutputPrefetchIssueCount++;
    } else {
        stats.nextKPrefetchIssueCount++;
    }

    DPRINTF(MatrixFlow,
            "issue next-prefetch (%s): tile i=%u j=%u k=%u dims=(%u,%u,%u) gen=%llu "
            "mode=%s trigger=%s rowsA=%u rowsB=%u\n",
            isOutputTile ? "next_output" : "next_k",
            nextCtx.i, nextCtx.j, nextCtx.k, nextCtx.curTileM,
            nextCtx.curTileN, nextCtx.curTileK,
            static_cast<unsigned long long>(activeNextPrefetchGeneration),
            nextPrefetchMode.c_str(), nextPrefetchTrigger.c_str(),
            nextPrefetchTargetA, nextPrefetchTargetB);

    if (nextPrefetchTargetA > 0) {
        trySendMoreNextA();
    }
    arbitratePrefetchBIssues();
}

void
MatrixFlowEngine::startOutputPrefetchTile(const GemmContext &prefetchCtx)
{
    discardNextOutputTilePrefetch();

    nextOutputCtx = prefetchCtx;
    nextOutputPrefetchValid = true;
    activeNextOutputPrefetchGeneration = ++nextPrefetchGeneration;
    nextOutputPrefetchReqsIssuedB = 0;
    nextOutputPrefetchRowsBCompleted = 0;
    nextOutputPrefetchTargetB = nextPrefetchRowsBConfig == 0
        ? nextOutputCtx.curTileK
        : std::min(nextOutputCtx.curTileK, nextPrefetchRowsBConfig);
    nextOutputPrefetchReadyB = nextOutputPrefetchTargetB == 0;
    nextOutputFirstIssueTick = 0;
    stats.nextPrefetchIssueCount++;
    stats.nextOutputPrefetchIssueCount++;

    DPRINTF(MatrixFlow,
            "issue next-output prefetch: tile i=%u j=%u k=%u dims=(%u,%u,%u) "
            "gen=%llu trigger=%s rowsB=%u\n",
            nextOutputCtx.i, nextOutputCtx.j, nextOutputCtx.k,
            nextOutputCtx.curTileM, nextOutputCtx.curTileN,
            nextOutputCtx.curTileK,
            static_cast<unsigned long long>(activeNextOutputPrefetchGeneration),
            nextPrefetchTrigger.c_str(), nextOutputPrefetchTargetB);

    arbitratePrefetchBIssues();
}

void
MatrixFlowEngine::maybePrefetchNextTile()
{
    if (!prefetchEnabled() || nextPrefetchValid || nextPrefetchBDrainActive) {
        return;
    }
    if (hasNextKTile()) {
        startPrefetchNextTile(buildNextKContext(), false);
    }
}

void
MatrixFlowEngine::maybePrefetchNextOutputTile()
{
    if (!prefetchEnabled() || nextOutputPrefetchValid ||
        nextOutputPrefetchBDrainActive || !hasNextOutputTile() ||
        hasNextKTile()) {
        return;
    }

    startOutputPrefetchTile(buildNextOutputContext());
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
        &fetchDescCompleteEvent,
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

    DPRINTF(MatrixFlow,
            "Descriptor loaded: A=%#llx B=%#llx C=%#llx flag=%#llx size=%u "
            "tile=(%u,%u,%u)\n",
            static_cast<unsigned long long>(ctx.baseA),
            static_cast<unsigned long long>(ctx.baseB),
            static_cast<unsigned long long>(ctx.baseC),
            static_cast<unsigned long long>(ctx.flagAddr),
            ctx.size, ctx.tileM, ctx.tileN, ctx.tileK);

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

    reqsIssuedA = 0;
    reqsCompletedA = 0;
    targetReqsA = ctx.curTileM;

    if (targetReqsA > 0) {
        trySendMoreA();
    }

    // Start B fetch setup immediately so A/B DMA can overlap.
    issueFetchBTile();
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
        auto [reqAddr, reqBytes, reqOffset] = planReadRequest(rowAddr, rowBytes);
        uint8_t *dmaDst = dst;
        if (reqBytes != rowBytes || reqOffset != 0) {
            fetchABounceActive[r] = true;
            fetchABounceReqAddr[r] = reqAddr;
            fetchABounceReqBytes[r] = reqBytes;
            fetchABounceRowBytes[r] = rowBytes;
            fetchABounceOffset[r] = reqOffset;
            dmaDst = fetchABounceBuffer.data() + r * readBouncePitch;
        } else {
            fetchABounceActive[r] = false;
        }

        DPRINTF(MatrixFlow,
                "DMA FetchA row=%u addr=%#llx bytes=%llu reqAddr=%#llx "
                "reqBytes=%llu reqOffset=%llu (tile i=%u j=%u k=%u)\n",
                r,
                static_cast<unsigned long long>(rowAddr),
                static_cast<unsigned long long>(rowBytes),
                static_cast<unsigned long long>(reqAddr),
                static_cast<unsigned long long>(reqBytes),
                static_cast<unsigned long long>(reqOffset), ctx.i, ctx.j,
                ctx.k);

        stats.totalDmaBytesRead += reqBytes;
        dmaPort.dmaAction(MemCmd::ReadReq, reqAddr, reqBytes,
                          &fetchARowEvents[r], dmaDst, 0);
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

    std::fill(currentBRowState.begin(), currentBRowState.end(),
              BRowState::Empty);

    uint32_t prefetched_rows = currentPrefetchedBValid
        ? std::min(currentPrefetchedBRows, ctx.curTileK)
        : 0;
    uint32_t inflight_rows = 0;
    if (carryOverBActive) {
        prefetched_rows =
            std::min(carryOverBRowsCompleted, ctx.curTileK);
        const uint32_t covered_rows =
            std::min(carryOverBRowsIssued, ctx.curTileK);
        inflight_rows = covered_rows > prefetched_rows
            ? (covered_rows - prefetched_rows)
            : 0;
        stats.duplicateBRowFetchAvoided += covered_rows;
        for (uint32_t r = 0; r < prefetched_rows; ++r) {
            currentBRowState[r] = BRowState::ReadyFromNextOutput;
        }
        for (uint32_t r = prefetched_rows; r < covered_rows; ++r) {
            currentBRowState[r] = BRowState::InflightFromNextOutput;
        }
    } else if (prefetched_rows > 0) {
        for (uint32_t r = 0; r < prefetched_rows; ++r) {
            currentBRowState[r] = BRowState::ReadyFromNextOutput;
        }
    }

    reqsIssuedB = prefetched_rows + inflight_rows;
    reqsCompletedB = prefetched_rows;
    targetReqsB = ctx.curTileK;
    nextNormalBRowCursor = 0;
    currentPrefetchedBValid = false;
    currentPrefetchedBRows = 0;

    if (targetReqsB > reqsIssuedB) {
        trySendMoreB();
    }

    if (!computeDoneEvent.scheduled() &&
        reqsCompletedA == targetReqsA &&
        reqsCompletedB == targetReqsB) {
        launchComputeTile();
    }
}

void
MatrixFlowEngine::trySendMoreB()
{
    const Addr rowBytes =
        static_cast<Addr>(ctx.curTileN) * ctx.elemBytes;
    const bool carryOverInflightOutstanding =
        carryOverBActive && (carryOverBRowsCompleted < carryOverBRowsIssued);
    const uint32_t normalIssueLimit =
        (carryOverInflightOutstanding && holeFillLeadRowsConfig > 0) ?
        std::min(targetReqsB, carryOverBRowsIssued + holeFillLeadRowsConfig) :
        targetReqsB;

    while (reqsIssuedB < targetReqsB &&
           (reqsIssuedB - reqsCompletedB) < kMaxInFlight) {
        if (nextNormalBRowCursor >= normalIssueLimit) {
            if (normalIssueLimit < targetReqsB) {
                stats.normalFetchDeferredByCarry++;
            }
            break;
        }

        while (nextNormalBRowCursor < targetReqsB &&
               currentBRowState[nextNormalBRowCursor] != BRowState::Empty) {
            ++nextNormalBRowCursor;
            if (nextNormalBRowCursor >= normalIssueLimit) {
                break;
            }
        }
        if (nextNormalBRowCursor >= targetReqsB) {
            break;
        }
        if (nextNormalBRowCursor >= normalIssueLimit) {
            if (normalIssueLimit < targetReqsB) {
                stats.normalFetchDeferredByCarry++;
            }
            break;
        }

        const uint32_t r = nextNormalBRowCursor;
        const Addr rowAddr = ctx.baseB +
            ((static_cast<Addr>(ctx.k + r) * ctx.size + ctx.j) *
             ctx.elemBytes);
        auto *dst = tileBBuffer.data() + r * rowBytes;
        auto [reqAddr, reqBytes, reqOffset] = planReadRequest(rowAddr, rowBytes);
        uint8_t *dmaDst = dst;
        if (reqBytes != rowBytes || reqOffset != 0) {
            fetchBBounceActive[r] = true;
            fetchBBounceReqAddr[r] = reqAddr;
            fetchBBounceReqBytes[r] = reqBytes;
            fetchBBounceRowBytes[r] = rowBytes;
            fetchBBounceOffset[r] = reqOffset;
            dmaDst = fetchBBounceBuffer.data() + r * readBouncePitch;
        } else {
            fetchBBounceActive[r] = false;
        }

        DPRINTF(MatrixFlow,
                "DMA FetchB row=%u addr=%#llx bytes=%llu reqAddr=%#llx "
                "reqBytes=%llu reqOffset=%llu (tile i=%u j=%u k=%u)\n",
                r,
                static_cast<unsigned long long>(rowAddr),
                static_cast<unsigned long long>(rowBytes),
                static_cast<unsigned long long>(reqAddr),
                static_cast<unsigned long long>(reqBytes),
                static_cast<unsigned long long>(reqOffset), ctx.i, ctx.j,
                ctx.k);

        stats.totalDmaBytesRead += reqBytes;
        stats.fallbackBRowsFetched++;
        stats.normalFetchHoleRows++;
        dmaPort.dmaAction(MemCmd::ReadReq, reqAddr, reqBytes,
                          &fetchBRowEvents[r], dmaDst, 0);
        currentBRowState[r] = BRowState::InflightFromNormal;
        ++reqsIssuedB;
        ++nextNormalBRowCursor;
    }
}

void
MatrixFlowEngine::trySendMoreNextA()
{
    const Addr rowBytes =
        static_cast<Addr>(nextCtx.curTileK) * nextCtx.elemBytes;

    while (nextPrefetchReqsIssuedA < nextPrefetchTargetA &&
           (nextPrefetchReqsIssuedA - nextPrefetchRowsACompleted) < kMaxInFlight) {
        const uint32_t r = nextPrefetchReqsIssuedA;
        const Addr rowAddr = nextCtx.baseA +
            ((static_cast<Addr>(nextCtx.i + r) * nextCtx.size + nextCtx.k) *
             nextCtx.elemBytes);
        auto *dst = nextTileABuffer.data() + r * rowBytes;
        auto [reqAddr, reqBytes, reqOffset] = planReadRequest(rowAddr, rowBytes);
        uint8_t *dmaDst = dst;
        if (reqBytes != rowBytes || reqOffset != 0) {
            nextFetchABounceActive[r] = true;
            nextFetchABounceReqAddr[r] = reqAddr;
            nextFetchABounceReqBytes[r] = reqBytes;
            nextFetchABounceRowBytes[r] = rowBytes;
            nextFetchABounceOffset[r] = reqOffset;
            dmaDst = nextFetchABounceBuffer.data() + r * readBouncePitch;
        } else {
            nextFetchABounceActive[r] = false;
        }

        stats.totalDmaBytesRead += reqBytes;
        nextFetchARowGeneration[r] = activeNextPrefetchGeneration;
        dmaPort.dmaAction(MemCmd::ReadReq, reqAddr, reqBytes,
                          &nextFetchARowEvents[r], dmaDst, 0);
        ++nextPrefetchReqsIssuedA;
    }
}

void
MatrixFlowEngine::trySendMoreNextB()
{
    trySendMoreNextKBBudgeted();
}

uint32_t
MatrixFlowEngine::sharedPrefetchBOutstanding() const
{
    const uint32_t next_k_outstanding =
        nextPrefetchReqsIssuedB - nextPrefetchRowsBCompleted;
    const uint32_t next_output_outstanding =
        nextOutputPrefetchReqsIssuedB - nextOutputPrefetchRowsBCompleted;
    return next_k_outstanding + next_output_outstanding;
}

uint32_t
MatrixFlowEngine::sharedPrefetchBCredits() const
{
    const uint32_t outstanding = sharedPrefetchBOutstanding();
    return outstanding >= kMaxInFlight ? 0 : (kMaxInFlight - outstanding);
}

void
MatrixFlowEngine::trySendMoreNextKBBudgeted()
{
    const Addr rowBytes =
        static_cast<Addr>(nextCtx.curTileN) * nextCtx.elemBytes;

    while (nextPrefetchReqsIssuedB < nextPrefetchTargetB &&
           sharedPrefetchBCredits() > 0) {
        if (phase == Phase::WriteC && writeCOverlapWindowActive &&
            writeCBRowsIssued >= writeCOverlapBIssueBudgetRowsConfig) {
            break;
        }
        const uint32_t r = nextPrefetchReqsIssuedB;
        const Addr rowAddr = nextCtx.baseB +
            ((static_cast<Addr>(nextCtx.k + r) * nextCtx.size + nextCtx.j) *
             nextCtx.elemBytes);
        auto *dst = nextTileBBuffer.data() + r * rowBytes;
        auto [reqAddr, reqBytes, reqOffset] = planReadRequest(rowAddr, rowBytes);
        uint8_t *dmaDst = dst;
        if (reqBytes != rowBytes || reqOffset != 0) {
            nextFetchBBounceActive[r] = true;
            nextFetchBBounceReqAddr[r] = reqAddr;
            nextFetchBBounceReqBytes[r] = reqBytes;
            nextFetchBBounceRowBytes[r] = rowBytes;
            nextFetchBBounceOffset[r] = reqOffset;
            dmaDst = nextFetchBBounceBuffer.data() + r * readBouncePitch;
        } else {
            nextFetchBBounceActive[r] = false;
        }

        stats.totalDmaBytesRead += reqBytes;
        nextFetchBRowGeneration[r] = activeNextPrefetchGeneration;
        dmaPort.dmaAction(MemCmd::ReadReq, reqAddr, reqBytes,
                          &nextFetchBRowEvents[r], dmaDst, 0);
        ++nextPrefetchReqsIssuedB;
        stats.nextKPrefetchRowsIssued++;
        if (phase == Phase::WriteC && writeCOverlapWindowActive) {
            ++writeCBRowsIssued;
            stats.bRowsIssuedDuringWriteC++;
        }
    }
}

void
MatrixFlowEngine::trySendMoreNextOutputB()
{
    const Addr rowBytes =
        static_cast<Addr>(nextOutputCtx.curTileN) * nextOutputCtx.elemBytes;

    while (nextOutputPrefetchReqsIssuedB < nextOutputPrefetchTargetB &&
           sharedPrefetchBCredits() > 0) {
        if (phase == Phase::WriteC && writeCOverlapWindowActive &&
            writeCBRowsIssued >= writeCOverlapBIssueBudgetRowsConfig) {
            break;
        }
        if (nextOutputFirstIssueTick == 0) {
            nextOutputFirstIssueTick = curTick();
        }
        const uint32_t r = nextOutputPrefetchReqsIssuedB;
        const Addr rowAddr = nextOutputCtx.baseB +
            ((static_cast<Addr>(nextOutputCtx.k + r) * nextOutputCtx.size +
              nextOutputCtx.j) * nextOutputCtx.elemBytes);
        auto *dst = nextOutputTileBBuffer.data() + r * rowBytes;
        auto [reqAddr, reqBytes, reqOffset] = planReadRequest(rowAddr, rowBytes);
        uint8_t *dmaDst = dst;
        if (reqBytes != rowBytes || reqOffset != 0) {
            nextOutputFetchBBounceActive[r] = true;
            nextOutputFetchBBounceReqAddr[r] = reqAddr;
            nextOutputFetchBBounceReqBytes[r] = reqBytes;
            nextOutputFetchBBounceRowBytes[r] = rowBytes;
            nextOutputFetchBBounceOffset[r] = reqOffset;
            dmaDst = nextOutputFetchBBounceBuffer.data() + r * readBouncePitch;
        } else {
            nextOutputFetchBBounceActive[r] = false;
        }

        stats.totalDmaBytesRead += reqBytes;
        nextOutputFetchBRowGeneration[r] = activeNextOutputPrefetchGeneration;
        dmaPort.dmaAction(MemCmd::ReadReq, reqAddr, reqBytes,
                          &nextOutputFetchBRowEvents[r], dmaDst, 0);
        ++nextOutputPrefetchReqsIssuedB;
        stats.nextOutputPrefetchRowsIssued++;
        if (phase == Phase::WriteC && writeCOverlapWindowActive) {
            ++writeCBRowsIssued;
            stats.bRowsIssuedDuringWriteC++;
        }
    }
}

void
MatrixFlowEngine::arbitratePrefetchBIssues()
{
    if (phase == Phase::WriteC && writeCOverlapWindowActive) {
        const bool next_output_needs_new_rows =
            nextOutputPrefetchValid &&
            !nextOutputPrefetchReadyB &&
            nextOutputPrefetchReqsIssuedB < nextOutputPrefetchTargetB;
        const bool next_k_needs_new_rows =
            nextPrefetchValid &&
            !nextPrefetchIsOutputTile &&
            !nextPrefetchReadyB &&
            nextPrefetchReqsIssuedB < nextPrefetchTargetB;
        const bool budget_exhausted =
            writeCOverlapBIssueBudgetRowsConfig == 0 ||
            writeCBRowsIssued >= writeCOverlapBIssueBudgetRowsConfig;

        if (budget_exhausted &&
            (next_output_needs_new_rows || next_k_needs_new_rows)) {
            stats.writeCBlockedBIssueCount++;
            DPRINTF(MatrixFlow,
                    "WriteC conservative overlap blocked new B issue: "
                    "tile i=%u j=%u k=%u next_output=%u/%u next_k=%u/%u "
                    "budget=%u issued=%u\n",
                    ctx.i, ctx.j, ctx.k,
                    nextOutputPrefetchReqsIssuedB, nextOutputPrefetchTargetB,
                    nextPrefetchReqsIssuedB, nextPrefetchTargetB,
                    writeCOverlapBIssueBudgetRowsConfig, writeCBRowsIssued);
            return;
        }
    }

    const bool next_output_needs_rows =
        nextOutputPrefetchValid &&
        !nextOutputPrefetchReadyB &&
        nextOutputPrefetchReqsIssuedB < nextOutputPrefetchTargetB;

    if (next_output_needs_rows) {
        if (sharedPrefetchBCredits() == 0) {
            stats.nextOutputPrefetchDeferCount++;
            return;
        }
        trySendMoreNextOutputB();
    }

    const bool next_k_needs_rows =
        nextPrefetchValid &&
        !nextPrefetchIsOutputTile &&
        !nextPrefetchReadyB &&
        nextPrefetchReqsIssuedB < nextPrefetchTargetB;

    if (sharedPrefetchBCredits() > 0 && next_k_needs_rows) {
        trySendMoreNextKBBudgeted();
    }
}

void
MatrixFlowEngine::serviceWriteCOverlap()
{
    if (phase != Phase::WriteC || !writeCOverlapWindowActive ||
        !prefetchEnabled()) {
        return;
    }

    const bool continuation_active =
        (nextOutputPrefetchValid &&
         nextOutputPrefetchReqsIssuedB > nextOutputPrefetchRowsBCompleted) ||
        (nextPrefetchValid &&
         nextPrefetchReqsIssuedB > nextPrefetchRowsBCompleted) ||
        (carryOverBActive &&
         carryOverBRowsIssued > carryOverBRowsCompleted);

    const bool pending_prefetch_b =
        (nextOutputPrefetchValid &&
         !nextOutputPrefetchReadyB &&
         nextOutputPrefetchReqsIssuedB < nextOutputPrefetchTargetB) ||
        (nextPrefetchValid &&
         !nextPrefetchIsOutputTile &&
         !nextPrefetchReadyB &&
         nextPrefetchReqsIssuedB < nextPrefetchTargetB);

    if (continuation_active) {
        DPRINTF(MatrixFlow,
                "WriteC conservative overlap keeps continuation alive: "
                "tile i=%u j=%u k=%u next_output_outstanding=%u "
                "next_k_outstanding=%u carry_over_outstanding=%u\n",
                ctx.i, ctx.j, ctx.k,
                nextOutputPrefetchReqsIssuedB - nextOutputPrefetchRowsBCompleted,
                nextPrefetchReqsIssuedB - nextPrefetchRowsBCompleted,
                carryOverBRowsIssued - carryOverBRowsCompleted);
    } else if (pending_prefetch_b) {
        stats.writeCBlockedBIssueCount++;
        DPRINTF(MatrixFlow,
                "WriteC conservative overlap deferred expansion: "
                "tile i=%u j=%u k=%u next_output=%u/%u next_k=%u/%u\n",
                ctx.i, ctx.j, ctx.k,
                nextOutputPrefetchReqsIssuedB, nextOutputPrefetchTargetB,
                nextPrefetchReqsIssuedB, nextPrefetchTargetB);
    } else {
        DPRINTF(MatrixFlow,
                "WriteC conservative overlap idle: tile i=%u j=%u k=%u\n",
                ctx.i, ctx.j, ctx.k);
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
    writeCOverlapWindowActive =
        writeCOverlapBIssueBudgetRowsConfig > 0 &&
        prefetchEnabled() &&
        (carryOverBActive || nextOutputPrefetchValid || nextPrefetchValid);
    writeCOverlapMadeProgress = false;
    writeCStartTick = curTick();
    writeCPrevPrefetchIssuedB =
        nextPrefetchReqsIssuedB + nextOutputPrefetchReqsIssuedB;
    writeCPrevNextOutputCompletedB = nextOutputPrefetchRowsBCompleted;
    writeCBRowsIssued = 0;
    if (writeCOverlapWindowActive) {
        stats.writeCOverlapEnabledCount++;
        serviceWriteCOverlap();
    }
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
                          &writeCRowEvents[r], src, 0);
        ++reqsIssuedC;
    }

    serviceWriteCOverlap();
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
        &writeFlagCompleteEvent,
        reinterpret_cast<uint8_t *>(&completionFlagValue), 0);
}

void
MatrixFlowEngine::onFetchARowComplete(uint32_t rowIdx)
{
    if (fetchABounceActive[rowIdx]) {
        uint8_t *src = fetchABounceBuffer.data() + rowIdx * readBouncePitch +
                       fetchABounceOffset[rowIdx];
        uint8_t *dst = tileABuffer.data() + rowIdx * fetchABounceRowBytes[rowIdx];
        std::memcpy(dst, src, fetchABounceRowBytes[rowIdx]);
        fetchABounceActive[rowIdx] = false;
    }

    panic_if(reqsCompletedA >= targetReqsA,
             "%s: onFetchARowComplete reqsCompletedA=%u >= targetReqsA=%u\n",
             name(), reqsCompletedA, targetReqsA);
    ++reqsCompletedA;
    if (reqsCompletedA < targetReqsA) {
        trySendMoreA();
    }

    if (reqsCompletedA == targetReqsA &&
        nextPrefetchTrigger == "a_ready") {
        maybePrefetchNextTile();
        maybePrefetchNextOutputTile();
    }

    if (!computeDoneEvent.scheduled() &&
        reqsCompletedA == targetReqsA &&
        reqsCompletedB == targetReqsB) {
        launchComputeTile();
    }
}

void
MatrixFlowEngine::onFetchBRowComplete(uint32_t rowIdx)
{
    if (fetchBBounceActive[rowIdx]) {
        uint8_t *src = fetchBBounceBuffer.data() + rowIdx * readBouncePitch +
                       fetchBBounceOffset[rowIdx];
        uint8_t *dst = tileBBuffer.data() + rowIdx * fetchBBounceRowBytes[rowIdx];
        std::memcpy(dst, src, fetchBBounceRowBytes[rowIdx]);
        fetchBBounceActive[rowIdx] = false;
    }

    panic_if(reqsCompletedB >= targetReqsB,
             "%s: onFetchBRowComplete reqsCompletedB=%u >= targetReqsB=%u\n",
             name(), reqsCompletedB, targetReqsB);
    currentBRowState[rowIdx] = BRowState::ReadyFromNormal;
    ++reqsCompletedB;
    if (reqsCompletedB < targetReqsB) {
        trySendMoreB();
    }

    // Earliest safe launch for next-output is when the current chunk is the
    // final-k chunk and its B rows are fully available. This gives next-output
    // more runway without making it correctness-critical.
    if (reqsCompletedB == targetReqsB && !hasNextKTile()) {
        maybePrefetchNextOutputTile();
    }

    if (reqsCompletedB == targetReqsB &&
        nextPrefetchTrigger == "b_ready") {
        maybePrefetchNextTile();
    }

    if (!computeDoneEvent.scheduled() &&
        reqsCompletedA == targetReqsA &&
        reqsCompletedB == targetReqsB) {
        launchComputeTile();
    }
}

void
MatrixFlowEngine::onWriteCRowComplete(uint32_t rowIdx)
{
    panic_if(reqsCompletedC >= targetReqsC,
             "%s: onWriteCRowComplete reqsCompletedC=%u >= targetReqsC=%u\n",
             name(), reqsCompletedC, targetReqsC);
    ++reqsCompletedC;
    serviceWriteCOverlap();
    if (reqsCompletedC == targetReqsC) {
        if (writeCOverlapWindowActive && writeCOverlapMadeProgress &&
            curTick() >= writeCStartTick) {
            stats.writeCOverlapCycles +=
                (curTick() - writeCStartTick) / clockPeriod();
        }
        writeCOverlapWindowActive = false;
        writeCOverlapMadeProgress = false;
        writeCStartTick = 0;
        writeCPrevPrefetchIssuedB = 0;
        writeCPrevNextOutputCompletedB = 0;
        writeCBRowsIssued = 0;
        advanceTile();
    } else {
        trySendMoreC();
    }
}

void
MatrixFlowEngine::onNextFetchARowComplete(uint32_t rowIdx)
{
    const uint64_t rowGen = nextFetchARowGeneration[rowIdx];
    if (!nextPrefetchValid || rowGen == 0 ||
        rowGen != activeNextPrefetchGeneration) {
        stats.nextPrefetchLateCompletionCount++;
        DPRINTF(MatrixFlow,
                "ignore stale next-fetchA completion: row=%u rowGen=%llu "
                "activeGen=%llu valid=%d\n",
                rowIdx, static_cast<unsigned long long>(rowGen),
                static_cast<unsigned long long>(activeNextPrefetchGeneration),
                nextPrefetchValid);
        return;
    }

    if (nextFetchABounceActive[rowIdx]) {
        uint8_t *src = nextFetchABounceBuffer.data() + rowIdx * readBouncePitch +
                       nextFetchABounceOffset[rowIdx];
        uint8_t *dst =
            nextTileABuffer.data() + rowIdx * nextFetchABounceRowBytes[rowIdx];
        std::memcpy(dst, src, nextFetchABounceRowBytes[rowIdx]);
        nextFetchABounceActive[rowIdx] = false;
    }

    panic_if(nextPrefetchRowsACompleted >= nextPrefetchTargetA,
             "%s: onNextFetchARowComplete nextPrefetchRowsACompleted=%u >= "
             "nextPrefetchTargetA=%u\n",
             name(), nextPrefetchRowsACompleted, nextPrefetchTargetA);
    ++nextPrefetchRowsACompleted;
    if (nextPrefetchRowsACompleted < nextPrefetchTargetA) {
        trySendMoreNextA();
    }

    if (nextPrefetchRowsACompleted == nextPrefetchTargetA) {
        nextPrefetchReadyA = true;
        DPRINTF(MatrixFlow,
                "next-fetchA ready: tile i=%u j=%u k=%u gen=%llu\n",
                nextCtx.i, nextCtx.j, nextCtx.k,
                static_cast<unsigned long long>(activeNextPrefetchGeneration));
    }
}

void
MatrixFlowEngine::onNextFetchBRowComplete(uint32_t rowIdx)
{
    const uint64_t rowGen = nextFetchBRowGeneration[rowIdx];
    if (nextPrefetchBDrainActive && rowGen == nextPrefetchBDrainGeneration) {
        panic_if(nextPrefetchBDrainOutstanding == 0,
                 "%s: stale-drain completion with zero outstanding for gen=%llu\n",
                 name(),
                 static_cast<unsigned long long>(nextPrefetchBDrainGeneration));
        --nextPrefetchBDrainOutstanding;
        stats.nextPrefetchLateCompletionCount++;
        DPRINTF(MatrixFlow,
                "drain stale next-fetchB completion: row=%u rowGen=%llu "
                "remaining=%u\n",
                rowIdx, static_cast<unsigned long long>(rowGen),
                nextPrefetchBDrainOutstanding);
        if (nextPrefetchBDrainOutstanding == 0) {
            nextPrefetchBDrainActive = false;
            nextPrefetchBDrainGeneration = 0;
        }
        arbitratePrefetchBIssues();
        return;
    }
    if (!nextPrefetchValid || rowGen == 0 ||
        rowGen != activeNextPrefetchGeneration) {
        stats.nextPrefetchLateCompletionCount++;
        DPRINTF(MatrixFlow,
                "ignore stale next-fetchB completion: row=%u rowGen=%llu "
                "activeGen=%llu valid=%d\n",
                rowIdx, static_cast<unsigned long long>(rowGen),
                static_cast<unsigned long long>(activeNextPrefetchGeneration),
                nextPrefetchValid);
        arbitratePrefetchBIssues();
        return;
    }

    if (nextFetchBBounceActive[rowIdx]) {
        uint8_t *src = nextFetchBBounceBuffer.data() + rowIdx * readBouncePitch +
                       nextFetchBBounceOffset[rowIdx];
        uint8_t *dst =
            nextTileBBuffer.data() + rowIdx * nextFetchBBounceRowBytes[rowIdx];
        std::memcpy(dst, src, nextFetchBBounceRowBytes[rowIdx]);
        nextFetchBBounceActive[rowIdx] = false;
    }

    panic_if(nextPrefetchRowsBCompleted >= nextPrefetchTargetB,
             "%s: onNextFetchBRowComplete nextPrefetchRowsBCompleted=%u >= "
             "nextPrefetchTargetB=%u\n",
             name(), nextPrefetchRowsBCompleted, nextPrefetchTargetB);
    ++nextPrefetchRowsBCompleted;
    if (nextPrefetchRowsBCompleted < nextPrefetchTargetB) {
        arbitratePrefetchBIssues();
    }

    if (nextPrefetchRowsBCompleted == nextPrefetchTargetB) {
        nextPrefetchReadyB = true;
        DPRINTF(MatrixFlow,
                "next-fetchB ready: tile i=%u j=%u k=%u gen=%llu\n",
                nextCtx.i, nextCtx.j, nextCtx.k,
                static_cast<unsigned long long>(activeNextPrefetchGeneration));
    }
    arbitratePrefetchBIssues();
}

void
MatrixFlowEngine::onNextOutputFetchBRowComplete(uint32_t rowIdx)
{
    const uint64_t rowGen = nextOutputFetchBRowGeneration[rowIdx];
    if (carryOverBActive && rowGen == carryOverBGeneration &&
        rowIdx < carryOverBRowsIssued) {
        if (carryOverBRowsCompleted >= carryOverBRowsIssued ||
            currentBRowState[rowIdx] == BRowState::ReadyFromNextOutput ||
            currentBRowState[rowIdx] == BRowState::Consumed) {
            stats.nextPrefetchLateCompletionCount++;
            stats.nextOutputPrefetchLateCompletionCount++;
            stats.carryOverLateCompletionCount++;
            DPRINTF(MatrixFlow,
                    "ignore duplicate/late carry-over completion: row=%u "
                    "rowGen=%llu completed=%u issued=%u state=%d\n",
                    rowIdx, static_cast<unsigned long long>(rowGen),
                    carryOverBRowsCompleted, carryOverBRowsIssued,
                    static_cast<int>(currentBRowState[rowIdx]));
            arbitratePrefetchBIssues();
            return;
        }
        if (nextOutputFetchBBounceActive[rowIdx]) {
            uint8_t *src = nextOutputFetchBBounceBuffer.data() +
                           rowIdx * readBouncePitch +
                           nextOutputFetchBBounceOffset[rowIdx];
            uint8_t *dst = tileBBuffer.data() +
                           rowIdx * nextOutputFetchBBounceRowBytes[rowIdx];
            std::memcpy(dst, src, nextOutputFetchBBounceRowBytes[rowIdx]);
            nextOutputFetchBBounceActive[rowIdx] = false;
        }

        ++carryOverBRowsCompleted;
        currentBRowState[rowIdx] = BRowState::ReadyFromNextOutput;
        stats.carryOverRowsConsumedPostBoundary++;
        if (phase == Phase::WriteC && writeCOverlapWindowActive) {
            writeCOverlapMadeProgress = true;
            stats.writeCOverlapSuccessCount++;
            stats.nextOutputProgressDuringWriteC++;
        }
        panic_if(reqsCompletedB >= targetReqsB,
                 "%s: carry-over completion reqsCompletedB=%u >= targetReqsB=%u\n",
                 name(), reqsCompletedB, targetReqsB);
        ++reqsCompletedB;
        if (carryOverBRowsCompleted == carryOverBRowsIssued) {
            carryOverBActive = false;
            carryOverBGeneration = 0;
            carryOverBRowsIssued = 0;
            carryOverBRowsCompleted = 0;
            carryOverBRowsTarget = 0;
        }
        if (reqsCompletedB < targetReqsB) {
            trySendMoreB();
        }
        if (!computeDoneEvent.scheduled() &&
            reqsCompletedA == targetReqsA &&
            reqsCompletedB == targetReqsB) {
            launchComputeTile();
        }
        return;
    }
    if (nextOutputPrefetchBDrainActive &&
        rowGen == nextOutputPrefetchBDrainGeneration) {
        panic_if(nextOutputPrefetchBDrainOutstanding == 0,
                 "%s: output stale-drain completion with zero outstanding "
                 "for gen=%llu\n",
                 name(),
                 static_cast<unsigned long long>(
                     nextOutputPrefetchBDrainGeneration));
        --nextOutputPrefetchBDrainOutstanding;
        stats.nextPrefetchLateCompletionCount++;
        stats.nextOutputPrefetchLateCompletionCount++;
        if (nextOutputPrefetchBDrainOutstanding == 0) {
            nextOutputPrefetchBDrainActive = false;
            nextOutputPrefetchBDrainGeneration = 0;
        }
        arbitratePrefetchBIssues();
        return;
    }

    if (!nextOutputPrefetchValid || rowGen == 0 ||
        rowGen != activeNextOutputPrefetchGeneration) {
        stats.nextPrefetchLateCompletionCount++;
        stats.nextOutputPrefetchLateCompletionCount++;
        if (carryOverBGeneration != 0 && rowGen == carryOverBGeneration) {
            stats.carryOverLateCompletionCount++;
        }
        DPRINTF(MatrixFlow,
                "ignore stale next-output fetchB completion: row=%u rowGen=%llu "
                "activeGen=%llu valid=%d\n",
                rowIdx, static_cast<unsigned long long>(rowGen),
                static_cast<unsigned long long>(
                    activeNextOutputPrefetchGeneration),
                nextOutputPrefetchValid);
        arbitratePrefetchBIssues();
        return;
    }

    if (nextOutputFetchBBounceActive[rowIdx]) {
        uint8_t *src = nextOutputFetchBBounceBuffer.data() +
                       rowIdx * readBouncePitch +
                       nextOutputFetchBBounceOffset[rowIdx];
        uint8_t *dst = nextOutputTileBBuffer.data() +
                       rowIdx * nextOutputFetchBBounceRowBytes[rowIdx];
        std::memcpy(dst, src, nextOutputFetchBBounceRowBytes[rowIdx]);
        nextOutputFetchBBounceActive[rowIdx] = false;
    }

    panic_if(nextOutputPrefetchRowsBCompleted >= nextOutputPrefetchTargetB,
             "%s: onNextOutputFetchBRowComplete completed=%u >= target=%u\n",
             name(), nextOutputPrefetchRowsBCompleted,
             nextOutputPrefetchTargetB);
    ++nextOutputPrefetchRowsBCompleted;
    if (phase == Phase::WriteC && writeCOverlapWindowActive) {
        writeCOverlapMadeProgress = true;
        stats.writeCOverlapSuccessCount++;
        stats.nextOutputProgressDuringWriteC++;
    }
    if (nextOutputPrefetchRowsBCompleted < nextOutputPrefetchTargetB) {
        arbitratePrefetchBIssues();
    }

    if (nextOutputPrefetchRowsBCompleted == nextOutputPrefetchTargetB) {
        nextOutputPrefetchReadyB = true;
        DPRINTF(MatrixFlow,
                "next-output fetchB ready: tile i=%u j=%u k=%u gen=%llu\n",
                nextOutputCtx.i, nextOutputCtx.j, nextOutputCtx.k,
                static_cast<unsigned long long>(
                    activeNextOutputPrefetchGeneration));
    }
    arbitratePrefetchBIssues();
}

void
MatrixFlowEngine::launchComputeTile()
{
    phase = Phase::Compute;

    for (uint32_t r = 0; r < ctx.curTileK; ++r) {
        currentBRowState[r] = BRowState::Consumed;
    }

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
    if (nextPrefetchTrigger == "compute_launch") {
        maybePrefetchNextTile();
    }
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
    if (nextOutputPrefetchValid && nextOutputFirstIssueTick != 0) {
        const Tick boundary_delta = curTick() - nextOutputFirstIssueTick;
        stats.nextOutputFirstIssueToBoundaryCycles +=
            boundary_delta / clockPeriod();
    }
    if (nextOutputPrefetchValid) {
        const uint32_t rows_ready =
            std::min(nextOutputPrefetchRowsBCompleted, nextOutputCtx.curTileK);
        stats.nextOutputRowsReadyAtBoundary += rows_ready;
    }
    if (nextOutputPrefetchMatchesCurrentTile() && nextOutputPrefetchReadyB) {
        if (nextOutputFirstIssueTick != 0) {
            const Tick headstart_delta = curTick() - nextOutputFirstIssueTick;
            stats.nextOutputHeadstartCycles +=
                headstart_delta / clockPeriod();
        }
        applyPrefetchedNextOutputTile();
    } else if (nextOutputPrefetchMatchesCurrentTile() &&
               nextOutputPrefetchRowsBCompleted > 0) {
        const uint32_t rows_cap = carryOverMaxRowsConfig == 0
            ? nextOutputCtx.curTileK
            : std::min(nextOutputCtx.curTileK, carryOverMaxRowsConfig);
        const uint32_t rows_consumed_before_fallback =
            std::min(
                std::min(nextOutputPrefetchRowsBCompleted, nextOutputCtx.curTileK),
                rows_cap);
        const uint32_t issued_rows_total =
            std::min(nextOutputPrefetchReqsIssuedB, nextOutputCtx.curTileK);
        const uint32_t inherited_rows_issued = carryOverInheritInflight
            ? std::min(issued_rows_total, rows_cap)
            : rows_consumed_before_fallback;
        const uint32_t inflight_rows_at_boundary =
            inherited_rows_issued > rows_consumed_before_fallback
                ? (inherited_rows_issued - rows_consumed_before_fallback)
                : 0;
        stats.nextPrefetchFallbackCount++;
        stats.nextOutputPrefetchFallbackCount++;
        stats.nextOutputConsumedBeforeFallbackRows +=
            rows_consumed_before_fallback;
        stats.carryOverRowsAtBoundary += rows_consumed_before_fallback;
        stats.carryOverInflightRowsAtBoundary += inflight_rows_at_boundary;
        if (nextOutputFirstIssueTick != 0) {
            const Tick headstart_delta = curTick() - nextOutputFirstIssueTick;
            stats.nextOutputHeadstartCycles +=
                headstart_delta / clockPeriod();
        }
        std::swap(tileBBuffer, nextOutputTileBBuffer);
        currentPrefetchedBValid = false;
        currentPrefetchedBRows = 0;
        carryOverBActive = inherited_rows_issued > 0;
        carryOverBGeneration = activeNextOutputPrefetchGeneration;
        carryOverBRowsIssued = inherited_rows_issued;
        carryOverBRowsCompleted = rows_consumed_before_fallback;
        carryOverBRowsTarget = nextOutputCtx.curTileK;
        clearNextOutputTilePrefetch();
        DPRINTF(MatrixFlow,
                "next-output carry-over at tile advance: tile i=%u j=%u "
                "k=%u readyB=%u inflightB=%u inheritedIssued=%u cap=%u "
                "inheritInflight=%d gen=%llu\n",
                ctx.i, ctx.j, ctx.k,
                rows_consumed_before_fallback, inflight_rows_at_boundary,
                inherited_rows_issued, rows_cap, carryOverInheritInflight,
                static_cast<unsigned long long>(carryOverBGeneration));
    } else if (nextOutputPrefetchValid) {
        stats.nextPrefetchFallbackCount++;
        stats.nextOutputPrefetchFallbackCount++;
        const uint32_t rows_consumed_before_fallback =
            std::min(nextOutputPrefetchRowsBCompleted, nextOutputCtx.curTileK);
        stats.nextOutputConsumedBeforeFallbackRows +=
            rows_consumed_before_fallback;
        if (nextOutputFirstIssueTick != 0) {
            const Tick headstart_delta = curTick() - nextOutputFirstIssueTick;
            stats.nextOutputHeadstartCycles +=
                headstart_delta / clockPeriod();
        }
        DPRINTF(MatrixFlow,
                "next-output prefetch fallback at tile advance: tile i=%u j=%u "
                "k=%u readyB=%d completedB=%u/%u gen=%llu\n",
                nextOutputCtx.i, nextOutputCtx.j, nextOutputCtx.k,
                nextOutputPrefetchReadyB,
                nextOutputPrefetchRowsBCompleted, nextOutputPrefetchTargetB,
                static_cast<unsigned long long>(
                    activeNextOutputPrefetchGeneration));
        discardNextOutputTilePrefetch();
    }
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
        if (nextPrefetchMatchesCurrentTile() && nextPrefetchReadyB) {
            applyPrefetchedNextTile();
        } else {
            if (nextPrefetchValid) {
                stats.nextPrefetchFallbackCount++;
                DPRINTF(MatrixFlow,
                        "next-prefetch fallback (%s): tile i=%u j=%u k=%u "
                        "readyA=%d readyB=%d gen=%llu\n",
                        nextPrefetchIsOutputTile ? "next_output" : "next_k",
                        nextCtx.i, nextCtx.j, nextCtx.k,
                        nextPrefetchReadyA, nextPrefetchReadyB,
                        static_cast<unsigned long long>(
                            activeNextPrefetchGeneration));
                discardNextTilePrefetch();
            }
        }
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
