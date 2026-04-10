#ifndef __MEM_MATRIXFLOW_ENGINE_HH__
#define __MEM_MATRIXFLOW_ENGINE_HH__

#include <cstdint>
#include <tuple>
#include <unordered_map>
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
    /** Max bytes in one logical B row. */
    static constexpr size_t kMaxTileRowBytes =
        kMaxTileDim * sizeof(uint32_t);
    /** Max in-flight DMA requests (sliding window to avoid bus deadlock). */
    static constexpr uint32_t kMaxInFlight = 64;
    /** Fixed first-version quota for consecutive current-protected-B wins. */
    static constexpr uint32_t kCurrentProtectedBQuotaRows = 4;
    /** Conservative gather budget for the no-starvation sidecar. */
    static constexpr uint32_t kCoverageGatherNoStarvationBudgetRows = 16;
    /** Do not let gather monopolize B issue arbitration. */
    static constexpr uint32_t kCoverageGatherNoStarvationMaxConsecutiveIssues = 2;
    /** Give mainline a short grace window before gather can compete. */
    static constexpr uint32_t kCoverageGatherMainlineGraceCycles = 64;

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

    enum class ABSchedulerMode : uint8_t
    {
        Baseline = 0,
        Lightweight,
        FullScore,
        ProtectedB,
        HierarchicalProtectedB,
        HierarchicalClaimHoleFilling,
        HierarchicalClaimVIPPool,
        VIPOracleGuidedSingleRun,
        VIPOracleGuidedMainlineOnly,
        VIPRescueBufferSingleRun,
        BVIPRescueAntiDeadBlockAdmission,
        BVIPRescueRecurrenceAwareV2,
        BVIPRescueRecurrenceAwareV3,
        BMHotMainlineDefault,
        BMHotMainlineEnhanced,
        BMHotRuntimeFirstCut,
        BMHotGapAwareNextCut,
        BMHotCoverageBlindspotCandidateFirstCut,
        BMHotCoverageBlindspotCandidateV2,
        BCoverageShadowControllerFirstCut,
        BCoverageShadowControllerV2,
        BCoverage2DGatherFirstCut,
        BCoverage2DGatherV2,
        BCoverage2DGatherNoStarvationV3,
        BCoverage2DGatherMinGuaranteeFirstCut,
        BCoverage2DGatherPingPongFirstCut,
        ABSmartPatternPrefetchFirstCut,
        VIPABRescueBufferSingleRun,
        DualRXABHierarchicalVIP,
    };

    enum class BProtectionClass : uint8_t
    {
        None = 0,
        CarryOver,
        NextOutput,
        HoleFilling,
        ComputeWindow,
        FutureHoleFilling,
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

    enum class VipSourceClass : uint8_t
    {
        None = 0,
        NextOutput,
        Claim,
        CarryOver,
        CurrentWindow,
        Normal,
        Rescue,
        RescueA,
    };

    enum class VipAdmitClass : uint8_t
    {
        Reject = 0,
        Weak,
        Strong,
    };

    enum class CoverageShadowInsertSource : uint8_t
    {
        NextK = 0,
        NextOutput,
        Gather,
    };

    enum class OracleSourceClass : uint8_t
    {
        NextOutput = 0,
        Claim,
        CarryOver,
        CurrentWindow,
        Normal,
        NumClasses,
    };

    enum class OracleDistanceBucket : uint8_t
    {
        Immediate = 0,
        Near,
        Far,
        NumBuckets,
    };

    enum class OracleAReuseBucket : uint8_t
    {
        One = 0,
        Multi,
        NumBuckets,
    };

    enum class OracleARecurrenceBucket : uint8_t
    {
        First = 0,
        Repeat,
        NumBuckets,
    };

    enum class FallbackAutopsyBucket : uint8_t
    {
        Timeliness = 0,
        Churn,
        Coverage,
        NumBuckets,
    };

    struct VipRowSlot
    {
        bool valid = false;
        Addr rowAddr = 0;
        Addr rowBytes = 0;
        VipSourceClass source = VipSourceClass::None;
        BProtectionClass protectionClass = BProtectionClass::None;
        VipAdmitClass admitClass = VipAdmitClass::Reject;
        uint32_t rescueRepeatVictimCount = 0;
        uint32_t rescueFutureReuseCount = 0;
        bool rescueCritical = false;
        bool rescueShortNextUse = false;
        bool rescueNearOpportunity = false;
        bool coverageBlindspot = false;
        uint64_t epoch = 0;
    };

    struct BFallbackAutopsyMeta
    {
        bool nearbyInserted = false;
        bool nearbyEvicted = false;
        VipSourceClass lastSource = VipSourceClass::None;
        uint64_t lastInsertTick = 0;
        uint64_t lastEvictTick = 0;
    };

    struct CoverageBlindspotMeta
    {
        uint32_t seenCount = 0;
        uint64_t lastFallbackTick = 0;
        uint64_t lastPromoteTick = 0;
        bool promoted = false;
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
        statistics::Scalar aRowsIssued;
        statistics::Scalar aRowsReadyBeforeCompute;
        statistics::Scalar aRowsInflightPeak;
        statistics::Scalar bRowsInflightPeak;
        statistics::Scalar abParallelFetchOverlapCycles;
        statistics::Scalar aFetchProgressDuringBFetch;
        statistics::Scalar bFetchProgressDuringAFetch;
        statistics::Scalar aPathStallWaitingForB;
        statistics::Scalar bPathStallWaitingForA;
        statistics::Scalar aCreditFloorHits;
        statistics::Scalar bBiasWins;
        statistics::Scalar urgencyPriorityWins;
        statistics::Scalar deficitPriorityWins;
        statistics::Scalar reusePriorityWins;
        statistics::Scalar fallbackRiskPriorityWins;
        statistics::Scalar scoreTieBreakCount;
        statistics::Scalar avgScoreA;
        statistics::Scalar avgScoreB;
        statistics::Scalar maxScoreA;
        statistics::Scalar maxScoreB;
        statistics::Scalar protectedBIssueCount;
        statistics::Scalar protectedBReadyCount;
        statistics::Scalar protectedBPriorityWins;
        statistics::Scalar protectedBBlocksACount;
        statistics::Scalar protectedBBlocksNormalBCount;
        statistics::Scalar protectedBFromCarryOverCount;
        statistics::Scalar protectedBFromNextOutputCount;
        statistics::Scalar protectedBFromHoleFillingCount;
        statistics::Scalar protectedBFromComputeWindowCount;
        statistics::Scalar futureProtectedBIssueCount;
        statistics::Scalar futureProtectedBPriorityWins;
        statistics::Scalar futureProtectedBBlocksACount;
        statistics::Scalar futureProtectedBBlocksCurrentBCount;
        statistics::Scalar futureProtectedBFromNextOutputCount;
        statistics::Scalar futureProtectedBFromFutureHoleFillingCount;
        statistics::Scalar futureProtectedBReadyAtBoundaryCount;
        statistics::Scalar currentProtectedBIssueCount;
        statistics::Scalar currentProtectedBPriorityWins;
        statistics::Scalar currentProtectedBBlocksACount;
        statistics::Scalar currentProtectedBQuotaExhaustCount;
        statistics::Scalar currentProtectedBFromComputeWindowCount;
        statistics::Scalar currentProtectedBFromCurrentHoleFillingCount;
        statistics::Scalar bRowsClaimedByFuture;
        statistics::Scalar futureClaimSetCount;
        statistics::Scalar futureClaimClearedCount;
        statistics::Scalar futureClaimBlockedNormalFetchCount;
        statistics::Scalar futureClaimExpiredCount;
        statistics::Scalar futureClaimConsumedSuccessCount;
        statistics::Scalar futureClaimInvalidatedCount;
        statistics::Scalar vipPoolCapacity;
        statistics::Scalar vipPoolOccupancyPeak;
        statistics::Scalar vipInsertCount;
        statistics::Scalar vipHitCount;
        statistics::Scalar vipMissCount;
        statistics::Scalar vipEvictionCount;
        statistics::Scalar vipHitOnNextOutputCount;
        statistics::Scalar vipHitOnNextOutputImmediateCount;
        statistics::Scalar vipHitOnNextOutputNearCount;
        statistics::Scalar vipHitOnClaimedFutureBCount;
        statistics::Scalar vipHitOnCarryOverBCount;
        statistics::Scalar vipInsertFromNextOutputCount;
        statistics::Scalar vipInsertFromClaimCount;
        statistics::Scalar vipInsertFromCarryOverCount;
        statistics::Scalar vipMaterializeToCurrentCount;
        statistics::Scalar vipBRowsServedToCompute;
        statistics::Scalar vipBRowsPreventedFallbackCount;
        statistics::Scalar vipStrongAdmitCount;
        statistics::Scalar vipWeakAdmitCount;
        statistics::Scalar vipEvictLowPriorityCount;
        statistics::Scalar vipEvictWeakAdmitCount;
        statistics::Scalar vipEvictNormalCount;
        statistics::Scalar vipAdmitNextOutputImmediateCount;
        statistics::Scalar vipAdmitNextOutputNearCount;
        statistics::Scalar vipAdmitNextOutputFarCount;
        statistics::Scalar vipAdmitClaimCount;
        statistics::Scalar vipAdmitCarryOverCount;
        statistics::Scalar vipAdmitCurrentWindowImmediateCount;
        statistics::Scalar vipRejectNormalNearCount;
        statistics::Scalar vipRejectNormalFarCount;
        statistics::Scalar vipRejectOtherCount;
        statistics::Scalar oracleSelectedBRowsCount;
        statistics::Scalar oracleSelectedRowsServedByVipCount;
        statistics::Scalar oracleSelectedRowsMissedByVipCount;
        statistics::Scalar vipRescueInsertCount;
        statistics::Scalar vipRescueInsertStrongCount;
        statistics::Scalar vipRescueInsertWeakCount;
        statistics::Scalar vipRescueHitCount;
        statistics::Scalar vipRescueMissCount;
        statistics::Scalar vipRescueEvictionCount;
        statistics::Scalar vipRescueServedToComputeCount;
        statistics::Scalar vipRescuePreventedFallbackCount;
        statistics::Scalar vipRescueInsertAfterFallbackCount;
        statistics::Scalar vipRescueInsertShortNextUseCount;
        statistics::Scalar vipRescueInsertMultiFutureUseCount;
        statistics::Scalar vipRescueReusedCount;
        statistics::Scalar vipRescueRejectNonFallbackCount;
        statistics::Scalar vipRescueRejectNotShortUseCount;
        statistics::Scalar vipRescueRejectNotCriticalCount;
        statistics::Scalar vipRescueRejectNotRescueCriticalCount;
        statistics::Scalar vipRescueRejectSingleUseCount;
        statistics::Scalar vipRescueRejectNonRepeatCount;
        statistics::Scalar repeatVictimCountTotal;
        statistics::Scalar repeatVictimPromotedCount;
        statistics::Scalar criticalWindowVictimCount;
        statistics::Scalar criticalWindowVipInsertCount;
        statistics::Scalar weakRescuePromotedOnFirstFallbackCount;
        statistics::Scalar strongRescuePromotedOnRecurrenceCount;
        statistics::Scalar rescueCriticalityHighCount;
        statistics::Scalar nextRescueOpportunityNearCount;
        statistics::Scalar eligibleRescueVictimsFirstSeenCount;
        statistics::Scalar eligibleRescueVictimsPromotedEarlyCount;
        statistics::Scalar eligibleRescueVictimsHitAfterFirstPromotionCount;
        statistics::Scalar vipRescueAInsertCount;
        statistics::Scalar vipRescueAHitCount;
        statistics::Scalar vipRescueAMissCount;
        statistics::Scalar vipRescueAEvictionCount;
        statistics::Scalar vipRescueAServedToComputeCount;
        statistics::Scalar vipRescueAPreventedRemoteFetchCount;
        statistics::Scalar vipRescueAInsertShortNextUseCount;
        statistics::Scalar vipRescueAInsertMultiFutureUseCount;
        statistics::Scalar vipRescueAReusedCount;
        statistics::Scalar mhotPoolCapacity;
        statistics::Scalar mhotOccupancyPeak;
        statistics::Scalar mhotInsertCount;
        statistics::Scalar mhotInsertDefaultCount;
        statistics::Scalar mhotInsertEnhancedCount;
        statistics::Scalar mhotInsertNextOutputImmediateCount;
        statistics::Scalar mhotInsertNextOutputNearCount;
        statistics::Scalar mhotInsertNextOutputFarCount;
        statistics::Scalar mhotInsertClaimCount;
        statistics::Scalar mhotInsertCarryOverCount;
        statistics::Scalar mhotInsertNormalNearCount;
        statistics::Scalar mhotInsertCurrentWindowFarCount;
        statistics::Scalar mhotInsertCoverageBlindspotCount;
        statistics::Scalar mhotHitCount;
        statistics::Scalar mhotHitOnNextOutputImmediateCount;
        statistics::Scalar mhotHitOnNextOutputNearCount;
        statistics::Scalar mhotHitOnNextOutputFarCount;
        statistics::Scalar mhotHitOnClaimCount;
        statistics::Scalar mhotHitOnCarryOverCount;
        statistics::Scalar mhotHitOnNormalNearCount;
        statistics::Scalar mhotHitOnCurrentWindowFarCount;
        statistics::Scalar mhotHitOnCoverageBlindspotCount;
        statistics::Scalar mhotRowsServedToComputeCount;
        statistics::Scalar mhotMaterializeToCurrentCount;
        statistics::Scalar mhotEvictionCount;
        statistics::Scalar mhotEvictionCoverageBlindspotCount;
        statistics::Scalar mhotReuseHitCount;
        statistics::Scalar mhotMainlinePreventedRemoteCount;
        statistics::Scalar mhotRowsServedToComputeFromCoverageBlindspotCount;
        statistics::Scalar mhotMainlinePreventedRemoteFromCoverageBlindspotCount;
        statistics::Scalar mhotCheckedOnFallbackCount;
        statistics::Scalar mhotMissThenRemoteCount;
        statistics::Scalar mhotHitBeforeRemoteCount;
        statistics::Scalar coverageShadowPoolCapacity;
        statistics::Scalar coverageShadowOccupancyPeak;
        statistics::Scalar coverageShadowInsertCount;
        statistics::Scalar coverageShadowInsertFromNextKCount;
        statistics::Scalar coverageShadowInsertFromNextOutputCount;
        statistics::Scalar coverageShadowInsertFromGatherCount;
        statistics::Scalar coverageShadowHitCount;
        statistics::Scalar coverageShadowRowsServedToComputeCount;
        statistics::Scalar coverageShadowPreventedRemoteCount;
        statistics::Scalar coverageShadowEvictionCount;
        statistics::Scalar coverageShadowCheckedOnFallbackCount;
        statistics::Scalar coverageShadowMissThenRemoteCount;
        statistics::Scalar coverageShadowHitBeforeRemoteCount;
        statistics::Scalar coverageShadowPrimeCheckCount;
        statistics::Scalar coverageShadowPrimeHitCount;
        statistics::Scalar coverageShadowPrimeMissCount;
        statistics::Scalar coverageShadowPrimeRowsMaterializedCount;
        statistics::Scalar coverageShadowFallbackLookupSkippedCount;
        statistics::Scalar coverageGatherIssueCount;
        statistics::Scalar coverageGatherCompletionCount;
        statistics::Scalar coverageGatherLateCompletionCount;
        statistics::Scalar coverageGatherTargetRows;
        statistics::Scalar coverageGatherCompletionWhileRowEmptyCount;
        statistics::Scalar coverageGatherCompletionWhileRowInflightCount;
        statistics::Scalar coverageGatherCompletionWhileRowReadyCount;
        statistics::Scalar coverageGatherCompletionWhileRowConsumedCount;
        statistics::Scalar coverageGatherCompletionDowngradedCount;
        statistics::Scalar coverageGatherCompletionInsertedLowPriorityCount;
        statistics::Scalar coverageGatherCompletionDiscardedCount;
        statistics::Scalar coverageShadowLowPriorityOccupancyPeak;
        statistics::Scalar gatherMinBudgetReservedCount;
        statistics::Scalar gatherBudgetGrantedCount;
        statistics::Scalar gatherDeferredByMainlineCount;
        statistics::Scalar mainlineCreditReservedCount;
        statistics::Scalar gatherBlockedByMainlineGraceWindowCount;
        statistics::Scalar gatherDeferredByVipCount;
        statistics::Scalar vipDeferredByGatherCount;
        statistics::Scalar vipRescueModeActiveCount;
        statistics::Scalar gatherBudgetExhaustedCount;
        statistics::Scalar gatherMaxConsecutiveIssueHitsCount;
        statistics::Scalar gatherCandidateAgedUpCount;
        statistics::Scalar gatherDroppedDueToDeadlineCount;
        statistics::Scalar mhotCoverageBlindspotSeenCount;
        statistics::Scalar mhotCoverageBlindspotRepeatCount;
        statistics::Scalar mhotCoverageBlindspotPromotedCount;
        statistics::Scalar mhotCoverageBlindspotRejectSingleCount;
        statistics::Scalar mhotCoverageBlindspotRejectLongDistanceCount;
        statistics::Scalar mhotCoverageBlindspotRejectLowReuseCount;
        statistics::Scalar repeatCoverageVictimCountTotal;
        statistics::Scalar repeatCoverageVictimPromotedCount;
        statistics::Scalar coverageBlindspotShortNextUseCount;
        statistics::Scalar coverageBlindspotFutureReuseGt1Count;
        statistics::Scalar coverageBlindspotFirstSeenMissCount;
        statistics::Scalar coverageBlindspotRepeatMissCount;
        statistics::Scalar coverageBlindspotDistinctPatternCount;
        statistics::Scalar coverageBlindspotTargetMissCount;
        statistics::Scalar coverageBlindspotTargetFirstSeenMissCount;
        statistics::Scalar coverageBlindspotTargetRepeatMissCount;
        statistics::Scalar coverageBlindspotTargetDistinctPatternCount;
        statistics::Scalar smartPrefetchAIssueCount;
        statistics::Scalar smartPrefetchARepeatPriorityCount;
        statistics::Scalar smartPrefetchANearMultiPriorityCount;
        statistics::Scalar smartPrefetchBNextOutputPriorityIssueCount;
        statistics::Scalar smartPrefetchBNextOutputCoverageTargetIssueCount;
        statistics::Scalar smartPrefetchBNextKPriorityIssueCount;
        statistics::Vector fallbackAutopsyCoverageNormalByDistance;
        statistics::Vector coverageBlindspotSeenByDistance;
        statistics::Vector coverageBlindspotPromotedByDistance;
        statistics::Vector mhotHitOnCoverageBlindspotByDistance;
        statistics::Vector coverageBlindspotSeenByReuseBucket;
        statistics::Vector coverageBlindspotPromotedByReuseBucket;
        statistics::Vector coverageBlindspotSeenByTileBand;
        statistics::Vector coverageBlindspotPromotedByTileBand;
        statistics::Scalar rxAIssueCount;
        statistics::Scalar rxBIssueCount;
        statistics::Scalar rxAReadyCount;
        statistics::Scalar rxBReadyCount;
        statistics::Scalar rxAQueueOccupancyPeak;
        statistics::Scalar rxBQueueOccupancyPeak;
        statistics::Scalar rxAStallCycles;
        statistics::Scalar rxBStallCycles;
        statistics::Scalar rxAPriorityWins;
        statistics::Scalar rxBPriorityWins;
        statistics::Scalar rxBDeficitWins;
        statistics::Scalar rxADeficitWins;
        statistics::Scalar rxBDeadlineWins;
        statistics::Scalar rxADeadlineWins;
        statistics::Vector2d oracleLocalRows;
        statistics::Vector2d oracleFallbackRows;
        statistics::Vector2d oracleLocalReuseWeight;
        statistics::Vector2d oracleFallbackReuseWeight;
        statistics::Vector oracleALocalByDistance;
        statistics::Vector oracleARemoteByDistance;
        statistics::Vector oracleALocalByReuse;
        statistics::Vector oracleARemoteByReuse;
        statistics::Vector oracleALocalByRecurrence;
        statistics::Vector oracleARemoteByRecurrence;
        statistics::Scalar fallbackAutopsyTimelinessCount;
        statistics::Scalar fallbackAutopsyChurnCount;
        statistics::Scalar fallbackAutopsyCoverageCount;
        statistics::Vector2d fallbackAutopsyRows;

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
    const uint32_t vipBRowsCapacityConfig;
    const uint32_t mhotBRowsCapacityConfig;
    const uint32_t coverageShadowRowsCapacityConfig;
    const uint32_t coverageGatherMinIssueBudgetConfig;
    const ABSchedulerMode abSchedulerMode;
    const uint32_t abAMinCreditRowsConfig;
    const int abBiasBConfig;
    const int abWeightUrgencyConfig;
    const int abWeightDeficitConfig;
    const int abWeightReuseConfig;
    const int abWeightFallbackRiskConfig;
    const uint32_t abMinLaunchRowsAConfig;
    const uint32_t abMinLaunchRowsBConfig;

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
    std::vector<uint8_t> coverageGatherBBounceBuffer;
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
    std::vector<Addr> coverageGatherBBounceReqAddr;
    std::vector<Addr> coverageGatherBBounceReqBytes;
    std::vector<Addr> coverageGatherBBounceRowBytes;
    std::vector<Addr> coverageGatherBBounceOffset;
    std::vector<bool> coverageGatherBBounceActive;
    std::vector<uint64_t> nextFetchARowGeneration;
    std::vector<uint64_t> nextFetchBRowGeneration;
    std::vector<uint64_t> nextOutputFetchBRowGeneration;
    std::vector<uint64_t> coverageGatherBRowGeneration;
    std::vector<BRowState> currentBRowState;
    std::vector<BProtectionClass> currentBProtectionClass;
    std::vector<bool> nextOutputFutureClaimed;
    std::vector<BProtectionClass> nextOutputFutureClaimClass;
    std::vector<bool> nextOutputPrefetchFromMHot;
    std::vector<VipSourceClass> nextOutputPrefetchMHotSource;
    std::vector<bool> stagedCurrentBFutureClaimed;
    std::vector<BProtectionClass> stagedCurrentBFutureClaimClass;
    std::vector<bool> currentBFutureClaimed;
    std::vector<BProtectionClass> currentBFutureClaimClass;
    std::vector<uint8_t> vipBBuffer;
    std::vector<VipRowSlot> vipBSlots;
    std::vector<uint8_t> mhotBBuffer;
    std::vector<VipRowSlot> mhotBSlots;
    std::vector<uint8_t> coverageShadowBBuffer;
    std::vector<uint8_t> coverageShadowFillBBuffer;
    std::vector<VipRowSlot> coverageShadowBSlots;
    std::vector<VipRowSlot> coverageShadowFillBSlots;
    std::vector<bool> coverageShadowLowPrioritySlot;
    std::vector<bool> stagedCurrentBVipBacked;
    std::vector<bool> stagedCurrentBOracleVipSelected;
    std::vector<VipSourceClass> stagedCurrentBVipSource;
    std::vector<BProtectionClass> stagedCurrentBVipProtectionClass;
    std::vector<VipAdmitClass> stagedCurrentBVipAdmitClass;
    std::vector<bool> stagedCurrentBVipPrefetchedCounted;
    std::vector<bool> currentBVipBacked;
    std::vector<bool> currentBOracleVipSelected;
    std::vector<bool> currentBServedFromVip;
    std::vector<VipSourceClass> currentBVipSource;
    std::vector<VipAdmitClass> currentBVipAdmitClass;
    std::vector<bool> currentBVipPrefetchedCounted;
    std::vector<OracleSourceClass> currentBOracleSource;
    std::vector<bool> currentBOracleOutcomeRecorded;
    std::vector<bool> currentBCoverageShadowScanned;
    std::vector<bool> currentBCoverageShadowKnownMiss;
    std::vector<bool> currentARowIssued;
    std::vector<bool> nextPrefetchARowIssued;
    std::vector<bool> nextPrefetchBRowIssued;
    std::vector<bool> nextOutputPrefetchBRowIssued;
    std::vector<bool> coverageGatherBRowIssued;
    std::vector<uint8_t> coverageGatherAgeScore;
    std::unordered_map<Addr, BFallbackAutopsyMeta> bFallbackAutopsy;

    bool computeBusy;
    Addr pendingDescAddr;
    Addr pendingMatrixA;
    Addr pendingMatrixB;
    Addr pendingResult;
    Addr pendingFlagAddr;
    int pendingSize;
    Descriptor pendingDesc;
    uint64_t completionFlagValue;
    uint64_t hierarchicalDebugComputeDeferCount = 0;

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
    GemmContext coverageGatherCtx;
    GemmContext coverageShadowActiveCtx;
    GemmContext coverageShadowFillCtx;
    uint32_t nextOutputPrefetchReqsIssuedB = 0;
    uint32_t nextOutputPrefetchRowsBCompleted = 0;
    uint32_t nextOutputPrefetchTargetB = 0;
    uint64_t nextOutputPrefetchGeneration = 0;
    uint64_t activeNextOutputPrefetchGeneration = 0;
    bool coverageGatherValid = false;
    bool coverageShadowActiveValid = false;
    bool coverageShadowFillValid = false;
    uint32_t coverageGatherReqsIssuedB = 0;
    uint32_t coverageGatherRowsBCompleted = 0;
    uint32_t coverageGatherTargetB = 0;
    uint64_t coverageGatherGeneration = 0;
    uint64_t activeCoverageGatherGeneration = 0;
    uint32_t coverageGatherBudgetUsedThisTile = 0;
    uint32_t coverageGatherConsecutiveIssueStreak = 0;
    uint32_t coverageGatherMinBudgetGrantedThisTile = 0;
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
    bool lastFetchIssueWasA = false;
    bool abParallelOverlapActive = false;
    Tick abParallelOverlapStartTick = 0;
    bool aWaitedForBThisTile = false;
    bool bWaitedForAThisTile = false;
    uint32_t aInflightPeakObserved = 0;
    uint32_t bInflightPeakObserved = 0;
    uint64_t scoreSamplesAObserved = 0;
    uint64_t scoreSamplesBObserved = 0;
    int64_t scoreSumAObserved = 0;
    int64_t scoreSumBObserved = 0;
    int64_t maxScoreAObserved = 0;
    int64_t maxScoreBObserved = 0;
    uint32_t currentProtectedBWinStreak = 0;
    uint64_t stagedCurrentBFutureClaimGeneration = 0;
    uint64_t currentBFutureClaimGeneration = 0;
    uint32_t currentBFutureClaimActiveCount = 0;
    uint32_t peakFutureClaimRowsObserved = 0;
    uint32_t vipPoolOccupancy = 0;
    uint32_t vipPoolOccupancyPeakObserved = 0;
    uint64_t vipInsertEpoch = 0;
    uint32_t mhotPoolOccupancy = 0;
    uint32_t mhotPoolOccupancyPeakObserved = 0;
    uint64_t mhotInsertEpoch = 0;
    uint32_t coverageShadowPoolOccupancy = 0;
    uint32_t coverageShadowPoolOccupancyPeakObserved = 0;
    uint32_t coverageShadowLowPriorityOccupancy = 0;
    uint32_t coverageShadowLowPriorityOccupancyPeakObserved = 0;
    uint32_t coverageShadowFillPoolOccupancy = 0;
    uint64_t coverageShadowInsertEpoch = 0;
    uint32_t rxAQueuePeakObserved = 0;
    uint32_t rxBQueuePeakObserved = 0;
    std::unordered_map<Addr, uint32_t> aRemoteVictimCount;
    std::unordered_map<Addr, uint32_t> bFallbackVictimCount;
    std::unordered_map<uint64_t, CoverageBlindspotMeta> bCoverageBlindspotPattern;
    std::unordered_map<uint64_t, uint32_t> bCoverageBlindspotMissPatternCount;

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
    std::vector<EventFunctionWrapper> coverageGatherBRowEvents;

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
    bool issueOneFetchA();
    bool issueOneFetchB();
    void serviceParallelFetchAB(bool allowLaunchCompute = true);
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
    void onCoverageGatherBRowComplete(uint32_t rowIdx);
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
    void startCoverageGatherForCurrentTile();
    void clearCoverageGatherForCurrentTile();
    void trySendMoreCoverageGatherB();
    void ageUpCoverageGatherCandidates();
    uint32_t sharedPrefetchBOutstanding() const;
    uint32_t sharedPrefetchBCredits() const;
    void arbitratePrefetchBIssues();
    void serviceWriteCOverlap();
    bool baselineSchedulerMode() const;
    bool parallelABSchedulerMode() const;
    bool protectedBSchedulerMode() const;
    bool hierarchicalProtectedBSchedulerMode() const;
    bool claimBasedHoleFillingMode() const;
    bool vipPoolMode() const;
    bool vipPoolEnabled() const;
    bool mhotPoolMode() const;
    bool mhotPoolEnabled() const;
    bool mhotDefaultMode() const;
    bool mhotEnhancedMode() const;
    bool mhotRuntimeFirstCutMode() const;
    bool mhotGapAwareNextCutMode() const;
    bool mhotCoverageBlindspotCandidateMode() const;
    bool mhotCoverageBlindspotCandidateV2Mode() const;
    bool coverageShadowMode() const;
    bool coverageShadowV2Mode() const;
    bool coverage2DGatherMode() const;
    bool coverage2DGatherV2Mode() const;
    bool coverage2DGatherNoStarvationV3Mode() const;
    bool coverage2DGatherMinGuaranteeFirstCutMode() const;
    bool coverage2DGatherPingPongFirstCutMode() const;
    bool coverageShadowPoolEnabled() const;
    bool coverageShadowUsesPrefetchFeed() const;
    bool coverageGatherMainlineGraceActive() const;
    bool smartPatternPrefetchMode() const;
    bool mhotAllowsMainlineBypass() const;
    bool oracleGuidedVipMode() const;
    bool vipForcedDiversionMode() const;
    bool pureMainlineVipMode() const;
    bool vipRescueMode() const;
    bool vipRescueAntiDeadBlockMode() const;
    bool vipRescueRecurrenceAwareV2Mode() const;
    bool vipRescueRecurrenceAwareV3Mode() const;
    bool vipABRescueMode() const;
    bool dualRxSchedulerMode() const;
    uint32_t computeLaunchWindowA() const;
    uint32_t computeLaunchWindowB() const;
    bool minimumInputsReadyForCompute() const;
    bool fullInputsReadyForAccumulate() const;
    bool inputsReadyForCompute() const;
    bool canIssueA() const;
    uint32_t normalIssueLimit() const;
    int nextIssuableBRow() const;
    bool canIssueB() const;
    bool isClaimedByFuture(uint32_t rowIdx) const;
    int nextClaimBlockedBRow() const;
    uint32_t protectedBFrontier() const;
    bool isProtectedComputeWindowRow(uint32_t rowIdx) const;
    bool isProtectedHoleFillingRow(uint32_t rowIdx) const;
    int nextProtectedBRow() const;
    bool hasProtectedBPressure() const;
    bool hasNormalBPressure() const;
    bool protectedNextOutputPending() const;
    bool nextOutputPendingRowCanUseMHot() const;
    bool hasFutureProtectedBPressure() const;
    bool hasCurrentProtectedBPressure() const;
    bool currentProtectedBQuotaAvailable() const;
    int nextCompetitiveBRow() const;
    bool canIssueCompetitiveB() const;
    bool issueOneCompetitiveB();
    void maybeSeedFutureProtectedB();
    bool issueOneFutureProtectedB();
    bool issueOneCurrentProtectedB();
    uint8_t urgencyBucketA() const;
    uint8_t urgencyBucketB() const;
    uint8_t deficitBucketA() const;
    uint8_t deficitBucketB() const;
    uint8_t reuseBucketA() const;
    uint8_t reuseBucketB() const;
    uint8_t fallbackRiskBucketA() const;
    uint8_t fallbackRiskBucketB() const;
    void updateScoreStats(int aScore, int bScore);
    void recordProtectedBReady(BProtectionClass cls);
    void recordProtectedBIssue(BProtectionClass cls);
    void setNextOutputFutureClaim(uint32_t rowIdx, BProtectionClass cls);
    void clearNextOutputFutureClaims();
    void clearStagedCurrentFutureClaims();
    void clearCurrentFutureClaims(bool invalidated = false);
    void clearVipPool();
    void clearMHotPool();
    void clearCoverageShadowPool();
    void clearCoverageShadowActiveBank();
    void clearCoverageShadowFillBank();
    void clearCoverageShadowLowPrioritySlots();
    void clearStagedCurrentVipBacked();
    void clearCurrentVipBacked(bool invalidate = false);
    void clearStagedCurrentOracleVipSelected();
    void clearCurrentOracleVipSelected();
    void clearCurrentOracleTracking();
    void clearFallbackAutopsyTracking();
    void stageCurrentFutureClaimsFromNextOutput(uint32_t firstClaimedRow,
                                                uint32_t issuedRows,
                                                uint64_t generation);
    void applyStagedCurrentFutureClaims();
    void stageCurrentVipBackedRow(uint32_t rowIdx, VipSourceClass source,
                                  BProtectionClass protectionClass,
                                  VipAdmitClass admitClass,
                                  bool prefetchedCounted);
    void stageCurrentOracleVipSelectedRow(uint32_t rowIdx);
    void applyStagedCurrentVipBackedRows();
    void applyStagedCurrentOracleVipSelectedRows();
    void clearCurrentFutureClaimRow(uint32_t rowIdx, bool expired,
                                    bool consumedSuccess,
                                    bool invalidated);
    bool expireCurrentFutureClaimsIfBlocked();
    OracleSourceClass oracleSourceFromVipSource(VipSourceClass src) const;
    OracleSourceClass oracleSourceFromProtectionClass(BProtectionClass cls) const;
    OracleDistanceBucket oracleDistanceBucket(uint32_t rowIdx,
                                              uint32_t totalRows) const;
    OracleDistanceBucket oracleADistanceBucket() const;
    OracleAReuseBucket oracleAReuseBucket() const;
    OracleARecurrenceBucket oracleARecurrenceBucket(Addr rowAddr) const;
    FallbackAutopsyBucket classifyBFallbackAutopsy(uint32_t rowIdx,
                                                   BProtectionClass cls,
                                                   Addr rowAddr) const;
    void setCurrentOracleSource(uint32_t rowIdx, OracleSourceClass src);
    void recordCurrentBOracleOutcome(uint32_t rowIdx, bool local);
    void recordCurrentAOracleOutcome(Addr rowAddr, bool local);
    void recordBFallbackAutopsy(uint32_t rowIdx, BProtectionClass cls,
                                Addr rowAddr);
    void noteBFallbackAutopsyInsert(Addr rowAddr, VipSourceClass source);
    void noteBFallbackAutopsyEvict(Addr rowAddr, VipSourceClass source);
    bool issueCurrentBRow(uint32_t rowIdx, BProtectionClass cls);
    VipSourceClass vipSourceFromProtectionClass(BProtectionClass cls) const;
    int vipSourcePriority(VipSourceClass src) const;
    int mhotSourcePriority(VipSourceClass src) const;
    uint32_t vipEmergencyReserveRows(uint32_t totalRows) const;
    uint32_t vipNearUseWindowRows(uint32_t totalRows) const;
    VipAdmitClass classifyVipAdmission(uint32_t rowIdx, uint32_t totalRows,
                                       VipSourceClass source,
                                       BProtectionClass protectionClass) const;
    VipAdmitClass classifyMHotAdmission(uint32_t rowIdx, uint32_t totalRows,
                                        VipSourceClass source,
                                        BProtectionClass protectionClass) const;
    void recordVipOracleGuidedAdmit(uint32_t rowIdx, uint32_t totalRows,
                                    VipSourceClass source);
    void recordMHotInsertStats(VipSourceClass source,
                               OracleDistanceBucket dist,
                               VipAdmitClass admitClass,
                               bool coverageBlindspot = false);
    void recordMHotHitStats(VipSourceClass source,
                            OracleDistanceBucket dist,
                            bool coverageBlindspot = false);
    void recordMHotMaterializeCurrent(VipSourceClass source,
                                      uint32_t rowIdx,
                                      uint32_t totalRows,
                                      bool coverageBlindspot = false);
    void recordVipOracleGuidedReject(OracleSourceClass source,
                                     OracleDistanceBucket dist);
    void recordOracleSelectedRowMiss(uint32_t rowIdx);
    uint32_t rescueFutureITileReuseCount() const;
    uint32_t rescueFutureJTileReuseCount() const;
    uint32_t rescueTileStepsToNextUse() const;
    uint32_t rescueATileStepsToNextUse() const;
    bool rescueNextUseIsShort() const;
    bool rescueANextUseIsShort() const;
    bool nextRescueOpportunityNear(uint32_t rowIdx) const;
    uint32_t repeatVictimCount(Addr rowAddr) const;
    bool criticalRowOrWindow(uint32_t rowIdx) const;
    bool rescueCriticalityHigh(uint32_t rowIdx, Addr rowAddr) const;
    VipAdmitClass classifyVipRescueAdmission(uint32_t rowIdx,
                                             Addr rowAddr) const;
    bool shouldAdmitVipRescueRow(uint32_t rowIdx) const;
    bool shouldAdmitVipRescueARow(uint32_t rowIdx) const;
    int findVipSlot(Addr rowAddr, Addr rowBytes) const;
    int findMHotSlot(Addr rowAddr, Addr rowBytes) const;
    int findCoverageShadowSlot(Addr rowAddr, Addr rowBytes) const;
    int findCoverageShadowSlotInBank(const std::vector<VipRowSlot> &slots,
                                     Addr rowAddr, Addr rowBytes) const;
    int rescueRetentionScore(uint32_t repeatVictimCount,
                             bool criticalRow,
                             bool shortNextUse,
                             bool nearOpportunity,
                             uint32_t futureReuseCount,
                             VipAdmitClass admitClass) const;
    int selectVipVictimSlot(VipSourceClass incoming,
                            VipAdmitClass admitClass,
                            uint32_t rescueRepeatVictimCount = 0,
                            bool rescueCritical = false,
                            bool rescueShortNextUse = false,
                            bool rescueNearOpportunity = false,
                            uint32_t rescueFutureReuseCount = 0) const;
    int selectMHotVictimSlot(VipSourceClass incoming,
                             VipAdmitClass admitClass,
                             bool coverageBlindspotCandidate) const;
    int selectCoverageShadowVictimSlot() const;
    int selectCoverageShadowVictimSlotInBank(
        const std::vector<VipRowSlot> &slots) const;
    bool tryInsertVipRow(Addr rowAddr, Addr rowBytes, const uint8_t *src,
                         VipSourceClass source,
                         BProtectionClass protectionClass,
                         VipAdmitClass admitClass,
                         uint32_t rescueRepeatVictimCount = 0,
                         bool rescueCritical = false,
                         bool rescueShortNextUse = false,
                         bool rescueNearOpportunity = false,
                         uint32_t rescueFutureReuseCount = 0);
    bool tryInsertMHotRow(Addr rowAddr, Addr rowBytes, const uint8_t *src,
                          VipSourceClass source,
                          BProtectionClass protectionClass,
                          VipAdmitClass admitClass,
                          OracleDistanceBucket dist,
                          bool coverageBlindspotCandidate = false);
    bool tryInsertCoverageShadowRow(Addr rowAddr, Addr rowBytes,
                                    const uint8_t *src,
                                    CoverageShadowInsertSource source);
    bool tryInsertCoverageShadowRowLowPriority(Addr rowAddr, Addr rowBytes,
                                               const uint8_t *src,
                                               CoverageShadowInsertSource source);
    bool tryInsertCoverageShadowRowInBank(std::vector<uint8_t> &buffer,
                                          std::vector<VipRowSlot> &slots,
                                          uint32_t &occupancy,
                                          Addr rowAddr, Addr rowBytes,
                                          const uint8_t *src,
                                          CoverageShadowInsertSource source);
    uint32_t mhotCoverageBlindspotQuotaRows() const;
    uint32_t futureITileReuseCount(const GemmContext &gctx) const;
    uint32_t futureJTileReuseCount(const GemmContext &gctx) const;
    uint32_t tileStepsToNextUse(const GemmContext &gctx) const;
    bool isSmartCoverageTargetForContext(const GemmContext &gctx,
                                         uint32_t rowIdx) const;
    bool isSmartGapTargetForContext(const GemmContext &gctx,
                                    uint32_t rowIdx) const;
    bool isSmartACandidateForContext(const GemmContext &gctx,
                                     uint32_t rowIdx) const;
    int selectSmartCurrentARow() const;
    int selectSmartNextPrefetchARow() const;
    int selectSmartNextPrefetchBRow() const;
    int selectSmartNextOutputPrefetchBRow() const;
    bool isCoverageBlindspotNormalCandidate(uint32_t rowIdx,
                                            BProtectionClass cls,
                                            Addr rowAddr) const;
    uint8_t coverageBlindspotFutureReuseBucket(uint32_t futureReuse) const;
    uint8_t coverageBlindspotTileStepBand(uint32_t tileSteps) const;
    uint64_t coverageBlindspotPatternKey(uint32_t rowIdx,
                                         BProtectionClass cls,
                                         Addr rowAddr) const;
    bool isCoverageBlindspotTargetPattern(uint32_t rowIdx,
                                          BProtectionClass cls,
                                          Addr rowAddr) const;
    bool coverageShadowContextMatches(const GemmContext &a,
                                      const GemmContext &b) const;
    bool buildCoverageGatherFillContext(GemmContext &gctx) const;
    void maybeActivateCoverageShadowBankForCurrentTile();
    bool issueCoverageGatherRow(uint32_t rowIdx, const GemmContext &gctx);
    bool isCoverageBlindspotPromoted(uint32_t rowIdx,
                                     BProtectionClass cls,
                                     Addr rowAddr) const;
    bool tryServeCurrentARowFromVip(uint32_t rowIdx, bool countMiss);
    bool tryServeCurrentBRowFromVip(uint32_t rowIdx, bool countMiss);
    bool tryServeCurrentBRowFromMHot(uint32_t rowIdx, BProtectionClass cls);
    bool tryPrimeCurrentBRowFromCoverageShadow(uint32_t rowIdx,
                                               BProtectionClass cls);
    bool tryServeCurrentBRowFromCoverageShadow(uint32_t rowIdx,
                                               BProtectionClass cls);
    bool tryServeNextOutputRowFromMHot(uint32_t rowIdx,
                                       BProtectionClass cls);
    void primeCurrentBRowsFromVip();
    void primeCurrentBRowsFromCoverageShadow();
    bool issueOneNextOutputProtectedB();
    bool issueOneProtectedB();
    bool chooseHierarchicalIssueA();
    bool issueOneDualRxB();
    bool chooseDualRxIssueA();
    uint32_t dualRxADeadlineScore() const;
    uint32_t dualRxBDeadlineScore() const;
    uint32_t dualRxADeficitScore() const;
    uint32_t dualRxBDeficitScore() const;
    void updateDualRxQueueStats();
    void recordFactorWin(bool choseA, int urgencyDiff, int deficitDiff,
                         int reuseDiff, int fallbackRiskDiff,
                         int biasDiff);
    bool chooseLightweightIssueA();
    bool chooseFullScoreIssueA();
    void tryLaunchComputeTile();
    void updateABParallelOverlapTracking();
    void closeABParallelOverlapTracking();

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
