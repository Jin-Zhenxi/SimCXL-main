#ifndef __MEM_MATRIXFLOW_ENGINE_HH__
#define __MEM_MATRIXFLOW_ENGINE_HH__

#include <cstdint>
#include <tuple>
#include <vector>

#include "base/statistics.hh"
#include "base/types.hh"
#include "dev/dma_device.hh"
#include "params/MatrixFlowEngine.hh"
#include "sim/clocked_object.hh"
#include "sim/eventq.hh"

namespace gem5
{

class MatrixFlowEngine : public ClockedObject
{
  private:
    /** Max tile dimension; buffer size and concurrent row DMA event pool. */
    static constexpr int kMaxTileDim = 128;
    /** Max in-flight DMA requests (sliding window to avoid bus deadlock). */
    static constexpr uint32_t kMaxInFlight = 64;

    class EngineDmaPort : public DmaPort
    {
      public:
        EngineDmaPort(MatrixFlowEngine &owner, System *sys)
            : DmaPort(&owner, sys)
        {
        }

        Addr lineSize() const
        {
            return cacheLineSize;
        }
    };

    EngineDmaPort dmaPort;

    enum class Phase
    {
        Idle,
        FetchDesc,
        FetchA,
        FetchB,
        Compute,
        WriteC,
        WriteFlag,
    };

    enum class BRowState : uint8_t
    {
        Empty = 0,
        ReadyFromNextOutput,
        InflightFromNextOutput,
        ReadyFromNormal,
        InflightFromNormal,
        Consumed,
    };

    struct Descriptor
    {
        uint64_t addrA = 0;
        uint64_t addrB = 0;
        uint64_t addrC = 0;
        uint64_t flagAddr = 0;
        uint32_t size = 0;
    };

    static_assert(sizeof(Descriptor) == 40,
                  "MatrixFlow descriptor layout must stay stable");

    struct GemmContext
    {
        Addr baseA = 0;
        Addr baseB = 0;
        Addr baseC = 0;
        Addr flagAddr = 0;
        uint32_t size = 0;
        uint32_t elemBytes = sizeof(uint32_t);
        uint32_t tileM = kMaxTileDim;
        uint32_t tileN = kMaxTileDim;
        uint32_t tileK = kMaxTileDim;
        uint32_t i = 0;
        uint32_t j = 0;
        uint32_t k = 0;
        uint32_t curTileM = 0;
        uint32_t curTileN = 0;
        uint32_t curTileK = 0;
    };

    struct EngineStats : public statistics::Group
    {
        statistics::Scalar totalDmaBytesRead;
        statistics::Scalar totalDmaBytesWritten;
        statistics::Scalar totalComputeCycles;
        statistics::Scalar nextPrefetchIssueCount;
        statistics::Scalar nextKPrefetchIssueCount;
        statistics::Scalar nextOutputPrefetchIssueCount;
        statistics::Scalar nextPrefetchHitCount;
        statistics::Scalar nextOutputPrefetchHitCount;
        statistics::Scalar nextPrefetchFallbackCount;
        statistics::Scalar nextOutputPrefetchFallbackCount;
        statistics::Scalar nextPrefetchLateCompletionCount;
        statistics::Scalar nextOutputPrefetchLateCompletionCount;
        statistics::Scalar nextPrefetchDiscardCount;
        statistics::Scalar prefetchedBRowsConsumed;
        statistics::Scalar fallbackBRowsFetched;
        statistics::Scalar nextOutputPrefetchRowsIssued;
        statistics::Scalar nextKPrefetchRowsIssued;
        statistics::Scalar nextOutputPrefetchDeferCount;
        statistics::Scalar nextOutputHeadstartCycles;
        statistics::Scalar nextOutputFirstIssueToBoundaryCycles;
        statistics::Scalar nextOutputRowsReadyAtBoundary;
        statistics::Scalar nextOutputConsumedBeforeFallbackRows;
        statistics::Scalar carryOverRowsAtBoundary;
        statistics::Scalar carryOverInflightRowsAtBoundary;
        statistics::Scalar carryOverRowsConsumedPostBoundary;
        statistics::Scalar normalFetchHoleRows;
        statistics::Scalar duplicateBRowFetchAvoided;
        statistics::Scalar duplicateBRowFetchDetected;
        statistics::Scalar carryOverLateCompletionCount;
        statistics::Scalar normalFetchDeferredByCarry;
        statistics::Scalar writeCOverlapCycles;
        statistics::Scalar writeCOverlapEnabledCount;
        statistics::Scalar writeCOverlapSuccessCount;
        statistics::Scalar nextOutputProgressDuringWriteC;
        statistics::Scalar bRowsIssuedDuringWriteC;
        statistics::Scalar writeCBlockedBIssueCount;

        explicit EngineStats(statistics::Group *parent);
    };

    const unsigned macArraySize;
    const Cycles computeLatencyPerOp;
    const Addr minReadRequestBytes;
    const Addr readBouncePitch;
    const std::string nextPrefetchMode;
    const std::string nextPrefetchTrigger;
    const uint32_t nextPrefetchRowsAConfig;
    const uint32_t nextPrefetchRowsBConfig;
    const uint32_t carryOverMaxRowsConfig;
    const bool carryOverInheritInflight;
    const uint32_t holeFillLeadRowsConfig;
    const uint32_t writeCOverlapBIssueBudgetRowsConfig;

    Phase phase;
    GemmContext ctx;

    /** Tile buffers: up to kMaxTileDim^2 uint32_t elements each. */
    std::vector<uint8_t> tileABuffer;
    std::vector<uint8_t> tileBBuffer;
    std::vector<uint8_t> tileCBuffer;
    std::vector<uint8_t> nextTileABuffer;
    std::vector<uint8_t> nextTileBBuffer;
    std::vector<uint8_t> nextOutputTileBBuffer;
    std::vector<uint8_t> fetchABounceBuffer;
    std::vector<uint8_t> fetchBBounceBuffer;
    std::vector<uint8_t> nextFetchABounceBuffer;
    std::vector<uint8_t> nextFetchBBounceBuffer;
    std::vector<uint8_t> nextOutputFetchBBounceBuffer;
    std::vector<Addr> fetchABounceReqAddr;
    std::vector<Addr> fetchABounceReqBytes;
    std::vector<Addr> fetchABounceRowBytes;
    std::vector<Addr> fetchABounceOffset;
    std::vector<bool> fetchABounceActive;
    std::vector<Addr> fetchBBounceReqAddr;
    std::vector<Addr> fetchBBounceReqBytes;
    std::vector<Addr> fetchBBounceRowBytes;
    std::vector<Addr> fetchBBounceOffset;
    std::vector<bool> fetchBBounceActive;
    std::vector<Addr> nextFetchABounceReqAddr;
    std::vector<Addr> nextFetchABounceReqBytes;
    std::vector<Addr> nextFetchABounceRowBytes;
    std::vector<Addr> nextFetchABounceOffset;
    std::vector<bool> nextFetchABounceActive;
    std::vector<Addr> nextFetchBBounceReqAddr;
    std::vector<Addr> nextFetchBBounceReqBytes;
    std::vector<Addr> nextFetchBBounceRowBytes;
    std::vector<Addr> nextFetchBBounceOffset;
    std::vector<bool> nextFetchBBounceActive;
    std::vector<Addr> nextOutputFetchBBounceReqAddr;
    std::vector<Addr> nextOutputFetchBBounceReqBytes;
    std::vector<Addr> nextOutputFetchBBounceRowBytes;
    std::vector<Addr> nextOutputFetchBBounceOffset;
    std::vector<bool> nextOutputFetchBBounceActive;
    std::vector<uint64_t> nextFetchARowGeneration;
    std::vector<uint64_t> nextFetchBRowGeneration;
    std::vector<uint64_t> nextOutputFetchBRowGeneration;
    std::vector<BRowState> currentBRowState;

    bool computeBusy;
    Addr pendingDescAddr;
    Addr pendingMatrixA;
    Addr pendingMatrixB;
    Addr pendingResult;
    Addr pendingFlagAddr;
    int pendingSize;
    Descriptor pendingDesc;
    uint64_t completionFlagValue;

    /** Sliding window state: issued vs completed per phase. */
    uint32_t reqsIssuedA = 0;
    uint32_t reqsCompletedA = 0;
    uint32_t targetReqsA = 0;
    uint32_t reqsIssuedB = 0;
    uint32_t reqsCompletedB = 0;
    uint32_t targetReqsB = 0;
    uint32_t nextNormalBRowCursor = 0;
    uint32_t reqsIssuedC = 0;
    uint32_t reqsCompletedC = 0;
    uint32_t targetReqsC = 0;

    /** Opportunistic next-k prefetch state; never required for correctness. */
    bool nextPrefetchIssuedA = false;
    bool nextPrefetchIssuedB = false;
    bool nextPrefetchReadyA = false;
    bool nextPrefetchReadyB = false;
    bool nextPrefetchValid = false;
    bool nextPrefetchIsOutputTile = false;
    GemmContext nextCtx;
    uint32_t nextPrefetchReqsIssuedA = 0;
    uint32_t nextPrefetchRowsACompleted = 0;
    uint32_t nextPrefetchTargetA = 0;
    uint32_t nextPrefetchReqsIssuedB = 0;
    uint32_t nextPrefetchRowsBCompleted = 0;
    uint32_t nextPrefetchTargetB = 0;
    uint64_t nextPrefetchGeneration = 0;
    uint64_t activeNextPrefetchGeneration = 0;
    bool nextPrefetchBDrainActive = false;
    uint64_t nextPrefetchBDrainGeneration = 0;
    uint32_t nextPrefetchBDrainOutstanding = 0;
    bool nextOutputPrefetchValid = false;
    bool nextOutputPrefetchReadyB = false;
    GemmContext nextOutputCtx;
    uint32_t nextOutputPrefetchReqsIssuedB = 0;
    uint32_t nextOutputPrefetchRowsBCompleted = 0;
    uint32_t nextOutputPrefetchTargetB = 0;
    uint64_t nextOutputPrefetchGeneration = 0;
    uint64_t activeNextOutputPrefetchGeneration = 0;
    bool nextOutputPrefetchBDrainActive = false;
    uint64_t nextOutputPrefetchBDrainGeneration = 0;
    uint32_t nextOutputPrefetchBDrainOutstanding = 0;
    Tick nextOutputFirstIssueTick = 0;
    bool currentPrefetchedBValid = false;
    uint32_t currentPrefetchedBRows = 0;
    bool carryOverBActive = false;
    uint64_t carryOverBGeneration = 0;
    uint32_t carryOverBRowsIssued = 0;
    uint32_t carryOverBRowsCompleted = 0;
    uint32_t carryOverBRowsTarget = 0;
    bool writeCOverlapWindowActive = false;
    bool writeCOverlapMadeProgress = false;
    Tick writeCStartTick = 0;
    uint32_t writeCPrevPrefetchIssuedB = 0;
    uint32_t writeCPrevNextOutputCompletedB = 0;
    uint32_t writeCBRowsIssued = 0;

    EventFunctionWrapper fetchDescCompleteEvent;
    EventFunctionWrapper computeDoneEvent;
    EventFunctionWrapper writeFlagCompleteEvent;

    /** One completion event per row slot (avoids "Event already scheduled"). */
    std::vector<EventFunctionWrapper> fetchARowEvents;
    std::vector<EventFunctionWrapper> fetchBRowEvents;
    std::vector<EventFunctionWrapper> writeCRowEvents;
    std::vector<EventFunctionWrapper> nextFetchARowEvents;
    std::vector<EventFunctionWrapper> nextFetchBRowEvents;
    std::vector<EventFunctionWrapper> nextOutputFetchBRowEvents;

    EngineStats stats;

    uint64_t estimateTileCycles(uint32_t tileM, uint32_t tileN,
                                uint32_t tileK) const;
    std::tuple<Addr, Addr, Addr> planReadRequest(
        Addr rowAddr, Addr rowBytes) const;
    void resetContext();
    void issueFetchDescriptor();
    void prepareOutputTile();
    void issueFetchATile();
    void issueFetchBTile();
    void issueWriteCTile();
    void trySendMoreA();
    void trySendMoreB();
    void trySendMoreC();
    void trySendMoreNextA();
    void trySendMoreNextB();
    void issueWriteFlag();
    void onFetchDescComplete();
    void onFetchARowComplete(uint32_t rowIdx);
    void onFetchBRowComplete(uint32_t rowIdx);
    void onWriteCRowComplete(uint32_t rowIdx);
    void onNextFetchARowComplete(uint32_t rowIdx);
    void onNextFetchBRowComplete(uint32_t rowIdx);
    void onNextOutputFetchBRowComplete(uint32_t rowIdx);
    void onWriteFlagComplete();
    void launchComputeTile();
    void accumulateCurrentTile();
    void advanceTile();
    void processComputeDone();
    bool hasNextKTile() const;
    bool hasNextOutputTile() const;
    GemmContext buildNextKContext() const;
    GemmContext buildNextOutputContext() const;
    void maybePrefetchNextTile();
    void startPrefetchNextTile(const GemmContext &prefetchCtx,
                               bool isOutputTile);
    void maybePrefetchNextOutputTile();
    void startOutputPrefetchTile(const GemmContext &prefetchCtx);
    void clearNextTilePrefetch();
    void discardNextTilePrefetch();
    void clearNextOutputTilePrefetch();
    void discardNextOutputTilePrefetch();
    bool nextPrefetchMatchesCurrentTile() const;
    bool nextOutputPrefetchMatchesCurrentTile() const;
    void applyPrefetchedNextTile();
    void applyPrefetchedNextOutputTile();
    void applyPrefetchedNextOutputRows(uint32_t prefetched_rows,
                                       bool full_hit);
    bool prefetchEnabled() const;
    bool prefetchIncludesA() const;
    void trySendMoreNextOutputB();
    void trySendMoreNextKBBudgeted();
    uint32_t sharedPrefetchBOutstanding() const;
    uint32_t sharedPrefetchBCredits() const;
    void arbitratePrefetchBIssues();
    void serviceWriteCOverlap();

  public:
    typedef MatrixFlowEngineParams Params;
    MatrixFlowEngine(const Params &p);

    Port &getPort(const std::string &if_name,
                  PortID idx = InvalidPortID) override;

    void init() override;

    void startup() override;

    void startMatrixCompute(Addr descriptorAddr);
};

} // namespace gem5

#endif // __MEM_MATRIXFLOW_ENGINE_HH__
