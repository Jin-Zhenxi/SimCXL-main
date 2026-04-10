#include "mem/matrixflow_engine.hh"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
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

inline uint32_t
satOutstanding(uint32_t issued, uint32_t completed)
{
    return issued > completed ? (issued - completed) : 0;
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
      vipBRowsCapacityConfig(p.vip_b_rows_capacity),
      mhotBRowsCapacityConfig(p.mhot_b_rows_capacity),
      coverageShadowRowsCapacityConfig(p.coverage_shadow_rows_capacity),
      coverageGatherMinIssueBudgetConfig(p.coverage_gather_min_issue_budget),
      abSchedulerMode([&p]() {
          if (p.ab_scheduler_mode == "baseline") {
              return ABSchedulerMode::Baseline;
          }
          if (p.ab_scheduler_mode == "lightweight") {
              return ABSchedulerMode::Lightweight;
          }
          if (p.ab_scheduler_mode == "fullscore") {
              return ABSchedulerMode::FullScore;
          }
          if (p.ab_scheduler_mode == "protected_b") {
              return ABSchedulerMode::ProtectedB;
          }
          if (p.ab_scheduler_mode == "hierarchical_protected_b") {
              return ABSchedulerMode::HierarchicalProtectedB;
          }
          if (p.ab_scheduler_mode == "hierarchical_claim_hole_filling") {
              return ABSchedulerMode::HierarchicalClaimHoleFilling;
          }
          if (p.ab_scheduler_mode == "hierarchical_claim_vip_pool") {
              return ABSchedulerMode::HierarchicalClaimVIPPool;
          }
          if (p.ab_scheduler_mode == "vip_oracle_guided_single_run") {
              return ABSchedulerMode::VIPOracleGuidedSingleRun;
          }
          if (p.ab_scheduler_mode == "vip_oracle_guided_mainline_only") {
              return ABSchedulerMode::VIPOracleGuidedMainlineOnly;
          }
          if (p.ab_scheduler_mode == "vip_rescue_buffer_single_run") {
              return ABSchedulerMode::VIPRescueBufferSingleRun;
          }
          if (p.ab_scheduler_mode ==
              "b_vip_rescue_anti_dead_block_admission") {
              return ABSchedulerMode::BVIPRescueAntiDeadBlockAdmission;
          }
          if (p.ab_scheduler_mode ==
              "b_vip_rescue_recurrence_aware_v2") {
              return ABSchedulerMode::BVIPRescueRecurrenceAwareV2;
          }
          if (p.ab_scheduler_mode ==
              "b_vip_rescue_recurrence_aware_v3") {
              return ABSchedulerMode::BVIPRescueRecurrenceAwareV3;
          }
          if (p.ab_scheduler_mode == "b_mhot_mainline_default") {
              return ABSchedulerMode::BMHotMainlineDefault;
          }
          if (p.ab_scheduler_mode == "b_mhot_mainline_enhanced") {
              return ABSchedulerMode::BMHotMainlineEnhanced;
          }
          if (p.ab_scheduler_mode == "b_mhot_runtime_first_cut") {
              return ABSchedulerMode::BMHotRuntimeFirstCut;
          }
          if (p.ab_scheduler_mode == "b_mhot_gap_aware_next_cut") {
              return ABSchedulerMode::BMHotGapAwareNextCut;
          }
          if (p.ab_scheduler_mode ==
              "b_mhot_coverage_blindspot_candidate_first_cut") {
              return ABSchedulerMode::BMHotCoverageBlindspotCandidateFirstCut;
          }
          if (p.ab_scheduler_mode ==
              "b_mhot_coverage_blindspot_candidate_v2") {
              return ABSchedulerMode::BMHotCoverageBlindspotCandidateV2;
          }
          if (p.ab_scheduler_mode ==
              "b_coverage_shadow_controller_first_cut") {
              return ABSchedulerMode::BCoverageShadowControllerFirstCut;
          }
          if (p.ab_scheduler_mode ==
              "b_coverage_shadow_controller_v2") {
              return ABSchedulerMode::BCoverageShadowControllerV2;
          }
          if (p.ab_scheduler_mode ==
              "b_coverage_2d_gather_first_cut") {
              return ABSchedulerMode::BCoverage2DGatherFirstCut;
          }
          if (p.ab_scheduler_mode ==
              "b_coverage_2d_gather_v2") {
              return ABSchedulerMode::BCoverage2DGatherV2;
          }
          if (p.ab_scheduler_mode ==
              "b_coverage_2d_gather_no_starvation_v3") {
              return ABSchedulerMode::BCoverage2DGatherNoStarvationV3;
          }
          if (p.ab_scheduler_mode ==
              "b_coverage_2d_gather_min_guarantee_first_cut") {
              return ABSchedulerMode::BCoverage2DGatherMinGuaranteeFirstCut;
          }
          if (p.ab_scheduler_mode ==
              "b_coverage_2d_gather_pingpong_first_cut") {
              return ABSchedulerMode::BCoverage2DGatherPingPongFirstCut;
          }
          if (p.ab_scheduler_mode ==
              "ab_smart_pattern_prefetch_first_cut") {
              return ABSchedulerMode::ABSmartPatternPrefetchFirstCut;
          }
          if (p.ab_scheduler_mode == "vip_ab_rescue_buffer_single_run") {
              return ABSchedulerMode::VIPABRescueBufferSingleRun;
          }
          if (p.ab_scheduler_mode == "dual_rx_ab_hierarchical_vip") {
              return ABSchedulerMode::DualRXABHierarchicalVIP;
          }
          panic("Unsupported A/B scheduler mode: %s",
                p.ab_scheduler_mode);
      }()),
      abAMinCreditRowsConfig(p.ab_a_min_credit_rows),
      abBiasBConfig(p.ab_bias_b),
      abWeightUrgencyConfig(p.ab_weight_urgency),
      abWeightDeficitConfig(p.ab_weight_deficit),
      abWeightReuseConfig(p.ab_weight_reuse),
      abWeightFallbackRiskConfig(p.ab_weight_fallback_risk),
      abMinLaunchRowsAConfig(p.ab_min_launch_rows_a),
      abMinLaunchRowsBConfig(p.ab_min_launch_rows_b),
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
      coverageGatherBBounceBuffer(kMaxTileDim * readBouncePitch, 0),
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
      coverageGatherBBounceReqAddr(kMaxTileDim, 0),
      coverageGatherBBounceReqBytes(kMaxTileDim, 0),
      coverageGatherBBounceRowBytes(kMaxTileDim, 0),
      coverageGatherBBounceOffset(kMaxTileDim, 0),
      coverageGatherBBounceActive(kMaxTileDim, false),
      nextFetchARowGeneration(kMaxTileDim, 0),
      nextFetchBRowGeneration(kMaxTileDim, 0),
      nextOutputFetchBRowGeneration(kMaxTileDim, 0),
      coverageGatherBRowGeneration(kMaxTileDim, 0),
      currentBRowState(kMaxTileDim, BRowState::Empty),
      currentBProtectionClass(kMaxTileDim, BProtectionClass::None),
      nextOutputFutureClaimed(kMaxTileDim, false),
      nextOutputFutureClaimClass(kMaxTileDim, BProtectionClass::None),
      nextOutputPrefetchFromMHot(kMaxTileDim, false),
      nextOutputPrefetchMHotSource(kMaxTileDim, VipSourceClass::None),
      stagedCurrentBFutureClaimed(kMaxTileDim, false),
      stagedCurrentBFutureClaimClass(kMaxTileDim, BProtectionClass::None),
      currentBFutureClaimed(kMaxTileDim, false),
      currentBFutureClaimClass(kMaxTileDim, BProtectionClass::None),
      vipBBuffer(p.vip_b_rows_capacity * kMaxTileRowBytes, 0),
      vipBSlots(p.vip_b_rows_capacity),
      mhotBBuffer(p.mhot_b_rows_capacity * kMaxTileRowBytes, 0),
      mhotBSlots(p.mhot_b_rows_capacity),
      coverageShadowBBuffer(p.coverage_shadow_rows_capacity * kMaxTileRowBytes, 0),
      coverageShadowFillBBuffer(
          p.coverage_shadow_rows_capacity * kMaxTileRowBytes, 0),
      coverageShadowBSlots(p.coverage_shadow_rows_capacity),
      coverageShadowFillBSlots(p.coverage_shadow_rows_capacity),
      coverageShadowLowPrioritySlot(p.coverage_shadow_rows_capacity, false),
      stagedCurrentBVipBacked(kMaxTileDim, false),
      stagedCurrentBOracleVipSelected(kMaxTileDim, false),
      stagedCurrentBVipSource(kMaxTileDim, VipSourceClass::None),
      stagedCurrentBVipProtectionClass(kMaxTileDim,
                                       BProtectionClass::None),
      stagedCurrentBVipAdmitClass(kMaxTileDim, VipAdmitClass::Reject),
      stagedCurrentBVipPrefetchedCounted(kMaxTileDim, false),
      currentBVipBacked(kMaxTileDim, false),
      currentBOracleVipSelected(kMaxTileDim, false),
      currentBServedFromVip(kMaxTileDim, false),
      currentBVipSource(kMaxTileDim, VipSourceClass::None),
      currentBVipAdmitClass(kMaxTileDim, VipAdmitClass::Reject),
      currentBVipPrefetchedCounted(kMaxTileDim, false),
      currentBOracleSource(kMaxTileDim, OracleSourceClass::Normal),
      currentBOracleOutcomeRecorded(kMaxTileDim, false),
      currentBCoverageShadowScanned(kMaxTileDim, false),
      currentBCoverageShadowKnownMiss(kMaxTileDim, false),
      currentARowIssued(kMaxTileDim, false),
      nextPrefetchARowIssued(kMaxTileDim, false),
      nextPrefetchBRowIssued(kMaxTileDim, false),
      nextOutputPrefetchBRowIssued(kMaxTileDim, false),
      coverageGatherBRowIssued(kMaxTileDim, false),
      coverageGatherAgeScore(kMaxTileDim, 0),
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
    coverageGatherBRowEvents.reserve(kMaxTileDim);
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
        coverageGatherBRowEvents.emplace_back(
            [this, i] { onCoverageGatherBRowComplete(i); },
            name() + ".coverage_gatherB_row" + std::to_string(i));
    }

    DPRINTF(MatrixFlow,
            "Create MatrixFlowEngine: mac_array_size=%u, "
            "compute_latency_per_op=%llu cycles, maxTile=%d, "
            "minReadReq=%lluB nextPrefetchMode=%s trigger=%s rowsA=%u "
            "rowsB=%u abScheduler=%s launchWinA=%u launchWinB=%u\n",
            macArraySize,
            static_cast<unsigned long long>(computeLatencyPerOp),
            kMaxTileDim,
            static_cast<unsigned long long>(minReadRequestBytes),
            nextPrefetchMode.c_str(),
            nextPrefetchTrigger.c_str(),
            nextPrefetchRowsAConfig,
            nextPrefetchRowsBConfig,
            p.ab_scheduler_mode.c_str(),
            abMinLaunchRowsAConfig,
            abMinLaunchRowsBConfig);
    stats.vipPoolCapacity = vipBRowsCapacityConfig;
    stats.mhotPoolCapacity = mhotBRowsCapacityConfig;
    stats.coverageShadowPoolCapacity = p.coverage_shadow_rows_capacity;
    using namespace statistics;
    constexpr size_t oracleSources = static_cast<size_t>(
        OracleSourceClass::NumClasses);
    constexpr size_t oracleDistances = static_cast<size_t>(
        OracleDistanceBucket::NumBuckets);
    // Do not use nozero on Vector2d here. In gem5, Vector2d::zero() only
    // checks the first element, which can suppress the whole matrix if
    // next_output/immediate happens to stay zero even when other buckets
    // are populated.
    stats.oracleLocalRows.init(oracleSources, oracleDistances)
        .flags(total);
    stats.oracleFallbackRows.init(oracleSources, oracleDistances)
        .flags(total);
    stats.oracleLocalReuseWeight.init(oracleSources, oracleDistances)
        .flags(total);
    stats.oracleFallbackReuseWeight.init(oracleSources, oracleDistances)
        .flags(total);
    constexpr size_t oracleAReuseBuckets = static_cast<size_t>(
        OracleAReuseBucket::NumBuckets);
    constexpr size_t oracleARecurrenceBuckets = static_cast<size_t>(
        OracleARecurrenceBucket::NumBuckets);
    constexpr size_t coverageBlindspotReuseBuckets = 4;
    constexpr size_t coverageBlindspotTileBands = 4;
    stats.oracleALocalByDistance.init(oracleDistances).flags(total);
    stats.oracleARemoteByDistance.init(oracleDistances).flags(total);
    stats.oracleALocalByReuse.init(oracleAReuseBuckets).flags(total);
    stats.oracleARemoteByReuse.init(oracleAReuseBuckets).flags(total);
    stats.oracleALocalByRecurrence.init(oracleARecurrenceBuckets).flags(total);
    stats.oracleARemoteByRecurrence.init(oracleARecurrenceBuckets).flags(total);
    stats.fallbackAutopsyCoverageNormalByDistance.init(oracleDistances)
        .flags(total);
    stats.coverageBlindspotSeenByDistance.init(oracleDistances).flags(total);
    stats.coverageBlindspotPromotedByDistance.init(oracleDistances)
        .flags(total);
    stats.mhotHitOnCoverageBlindspotByDistance.init(oracleDistances)
        .flags(total);
    stats.coverageBlindspotSeenByReuseBucket.init(coverageBlindspotReuseBuckets)
        .flags(total);
    stats.coverageBlindspotPromotedByReuseBucket
        .init(coverageBlindspotReuseBuckets)
        .flags(total);
    stats.coverageBlindspotSeenByTileBand.init(coverageBlindspotTileBands)
        .flags(total);
    stats.coverageBlindspotPromotedByTileBand.init(coverageBlindspotTileBands)
        .flags(total);
    constexpr size_t fallbackAutopsyBuckets = static_cast<size_t>(
        FallbackAutopsyBucket::NumBuckets);
    stats.fallbackAutopsyRows.init(oracleSources, fallbackAutopsyBuckets)
        .flags(total);
    const auto oracleSourceName = [](OracleSourceClass src) -> const char * {
        switch (src) {
          case OracleSourceClass::NextOutput:
            return "next_output";
          case OracleSourceClass::Claim:
            return "claim";
          case OracleSourceClass::CarryOver:
            return "carry_over";
          case OracleSourceClass::CurrentWindow:
            return "current_window";
          case OracleSourceClass::Normal:
          case OracleSourceClass::NumClasses:
          default:
            return "normal";
        }
    };
    const auto oracleDistanceName =
        [](OracleDistanceBucket bucket) -> const char * {
            switch (bucket) {
              case OracleDistanceBucket::Immediate:
                return "immediate";
              case OracleDistanceBucket::Near:
                return "near";
              case OracleDistanceBucket::Far:
              case OracleDistanceBucket::NumBuckets:
              default:
                return "far";
            }
        };
    const auto oracleAReuseName = [](OracleAReuseBucket bucket) -> const char * {
        switch (bucket) {
          case OracleAReuseBucket::One:
            return "one";
          case OracleAReuseBucket::Multi:
          case OracleAReuseBucket::NumBuckets:
          default:
            return "multi";
        }
    };
    const auto oracleARecurrenceName =
        [](OracleARecurrenceBucket bucket) -> const char * {
            switch (bucket) {
              case OracleARecurrenceBucket::First:
                return "first";
              case OracleARecurrenceBucket::Repeat:
              case OracleARecurrenceBucket::NumBuckets:
              default:
                return "repeat";
            }
        };
    const auto coverageBlindspotReuseName = [](size_t bucket) -> const char * {
        switch (bucket) {
          case 0:
            return "none";
          case 1:
            return "one";
          case 2:
            return "two_to_three";
          case 3:
          default:
            return "four_plus";
        }
    };
    const auto coverageBlindspotTileBandName = [](size_t bucket) -> const char * {
        switch (bucket) {
          case 0:
            return "lte4";
          case 1:
            return "lte12";
          case 2:
            return "lte24";
          case 3:
          default:
            return "gt24";
        }
    };
    const auto fallbackAutopsyName =
        [](FallbackAutopsyBucket bucket) -> const char * {
            switch (bucket) {
              case FallbackAutopsyBucket::Timeliness:
                return "timeliness";
              case FallbackAutopsyBucket::Churn:
                return "churn";
              case FallbackAutopsyBucket::Coverage:
              case FallbackAutopsyBucket::NumBuckets:
              default:
                return "coverage";
            }
        };
    for (size_t i = 0; i < oracleSources; ++i) {
        const auto src = static_cast<OracleSourceClass>(i);
        stats.oracleLocalRows.subname(i, oracleSourceName(src));
        stats.oracleFallbackRows.subname(i, oracleSourceName(src));
        stats.oracleLocalReuseWeight.subname(i, oracleSourceName(src));
        stats.oracleFallbackReuseWeight.subname(i, oracleSourceName(src));
        stats.fallbackAutopsyRows.subname(i, oracleSourceName(src));
        for (size_t j = 0; j < oracleDistances; ++j) {
            const auto dist = static_cast<OracleDistanceBucket>(j);
            stats.oracleLocalRows.ysubname(j, oracleDistanceName(dist));
            stats.oracleFallbackRows.ysubname(j, oracleDistanceName(dist));
            stats.oracleLocalReuseWeight.ysubname(
                j, oracleDistanceName(dist));
            stats.oracleFallbackReuseWeight.ysubname(
                j, oracleDistanceName(dist));
            stats.oracleALocalByDistance.subname(j, oracleDistanceName(dist));
            stats.oracleARemoteByDistance.subname(j, oracleDistanceName(dist));
            stats.fallbackAutopsyCoverageNormalByDistance.subname(
                j, oracleDistanceName(dist));
            stats.coverageBlindspotSeenByDistance.subname(
                j, oracleDistanceName(dist));
            stats.coverageBlindspotPromotedByDistance.subname(
                j, oracleDistanceName(dist));
            stats.mhotHitOnCoverageBlindspotByDistance.subname(
                j, oracleDistanceName(dist));
        }
        for (size_t j = 0; j < fallbackAutopsyBuckets; ++j) {
            const auto bucket = static_cast<FallbackAutopsyBucket>(j);
            stats.fallbackAutopsyRows.ysubname(j, fallbackAutopsyName(bucket));
        }
    }
    for (size_t i = 0; i < oracleAReuseBuckets; ++i) {
        const auto bucket = static_cast<OracleAReuseBucket>(i);
        stats.oracleALocalByReuse.subname(i, oracleAReuseName(bucket));
        stats.oracleARemoteByReuse.subname(i, oracleAReuseName(bucket));
    }
    for (size_t i = 0; i < oracleARecurrenceBuckets; ++i) {
        const auto bucket = static_cast<OracleARecurrenceBucket>(i);
        stats.oracleALocalByRecurrence.subname(i,
                                               oracleARecurrenceName(bucket));
        stats.oracleARemoteByRecurrence.subname(
            i, oracleARecurrenceName(bucket));
    }
    for (size_t i = 0; i < coverageBlindspotReuseBuckets; ++i) {
        stats.coverageBlindspotSeenByReuseBucket.subname(
            i, coverageBlindspotReuseName(i));
        stats.coverageBlindspotPromotedByReuseBucket.subname(
            i, coverageBlindspotReuseName(i));
    }
    for (size_t i = 0; i < coverageBlindspotTileBands; ++i) {
        stats.coverageBlindspotSeenByTileBand.subname(
            i, coverageBlindspotTileBandName(i));
        stats.coverageBlindspotPromotedByTileBand.subname(
            i, coverageBlindspotTileBandName(i));
    }
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
               "How many WriteC overlap service points saw pending B work but no issue progress"),
      ADD_STAT(aRowsIssued, statistics::units::Count::get(),
               "How many A rows were issued by the current-tile fetch path"),
      ADD_STAT(aRowsReadyBeforeCompute, statistics::units::Count::get(),
               "How many A rows were ready when compute launched"),
      ADD_STAT(aRowsInflightPeak, statistics::units::Count::get(),
               "Peak number of in-flight A rows observed during the run"),
      ADD_STAT(bRowsInflightPeak, statistics::units::Count::get(),
               "Peak number of in-flight B rows observed during the run"),
      ADD_STAT(abParallelFetchOverlapCycles, statistics::units::Cycle::get(),
               "Cycles where current-tile A and B fetches had overlap in flight"),
      ADD_STAT(aFetchProgressDuringBFetch, statistics::units::Count::get(),
               "How many A row completions happened while B fetch was still active"),
      ADD_STAT(bFetchProgressDuringAFetch, statistics::units::Count::get(),
               "How many B row completions happened while A fetch was still active"),
      ADD_STAT(aPathStallWaitingForB, statistics::units::Count::get(),
               "How many tiles saw A finish before B became compute-ready"),
      ADD_STAT(bPathStallWaitingForA, statistics::units::Count::get(),
               "How many tiles saw B finish before A became compute-ready"),
      ADD_STAT(aCreditFloorHits, statistics::units::Count::get(),
               "How many scheduler decisions were forced to A by the minimum credit floor"),
      ADD_STAT(bBiasWins, statistics::units::Count::get(),
               "How many scheduler decisions went to B because the configured bias broke a tie"),
      ADD_STAT(urgencyPriorityWins, statistics::units::Count::get(),
               "How many scheduler decisions were decided by urgency"),
      ADD_STAT(deficitPriorityWins, statistics::units::Count::get(),
               "How many scheduler decisions were decided by deficit"),
      ADD_STAT(reusePriorityWins, statistics::units::Count::get(),
               "How many scheduler decisions were decided by reuse"),
      ADD_STAT(fallbackRiskPriorityWins, statistics::units::Count::get(),
               "How many scheduler decisions were decided by fallback-risk pressure"),
      ADD_STAT(scoreTieBreakCount, statistics::units::Count::get(),
               "How many scheduler decisions ended in a tie break"),
      ADD_STAT(avgScoreA, statistics::units::Count::get(),
               "Running average score assigned to A in fullscore mode"),
      ADD_STAT(avgScoreB, statistics::units::Count::get(),
               "Running average score assigned to B in fullscore mode"),
      ADD_STAT(maxScoreA, statistics::units::Count::get(),
               "Maximum score assigned to A in fullscore mode"),
      ADD_STAT(maxScoreB, statistics::units::Count::get(),
               "Maximum score assigned to B in fullscore mode"),
      ADD_STAT(protectedBIssueCount, statistics::units::Count::get(),
               "How many protected-B requests were issued ahead of ordinary scheduler competition"),
      ADD_STAT(protectedBReadyCount, statistics::units::Count::get(),
               "How many protected-B rows became ready for use"),
      ADD_STAT(protectedBPriorityWins, statistics::units::Count::get(),
               "How many times the protected-B layer won before A/normal-B scoring"),
      ADD_STAT(protectedBBlocksACount, statistics::units::Count::get(),
               "How many times protected-B priority blocked an A issue opportunity"),
      ADD_STAT(protectedBBlocksNormalBCount, statistics::units::Count::get(),
               "How many times protected-B priority blocked a normal-B issue opportunity"),
      ADD_STAT(protectedBFromCarryOverCount, statistics::units::Count::get(),
               "How many protected-B rows came from carry-over"),
      ADD_STAT(protectedBFromNextOutputCount, statistics::units::Count::get(),
               "How many protected-B rows came from next-output prefetch"),
      ADD_STAT(protectedBFromHoleFillingCount, statistics::units::Count::get(),
               "How many protected-B rows came from hole-filling"),
      ADD_STAT(protectedBFromComputeWindowCount, statistics::units::Count::get(),
               "How many protected-B rows came from compute-window critical B requests"),
      ADD_STAT(futureProtectedBIssueCount, statistics::units::Count::get(),
               "How many future-protected-B requests were issued"),
      ADD_STAT(futureProtectedBPriorityWins, statistics::units::Count::get(),
               "How many times future-protected-B won the top priority layer"),
      ADD_STAT(futureProtectedBBlocksACount, statistics::units::Count::get(),
               "How many times future-protected-B blocked an A issue opportunity"),
      ADD_STAT(futureProtectedBBlocksCurrentBCount,
               statistics::units::Count::get(),
               "How many times future-protected-B blocked current-protected-B"),
      ADD_STAT(futureProtectedBFromNextOutputCount,
               statistics::units::Count::get(),
               "How many future-protected-B rows came from next-output"),
      ADD_STAT(futureProtectedBFromFutureHoleFillingCount,
               statistics::units::Count::get(),
               "How many future-protected-B rows came from future hole-filling"),
      ADD_STAT(futureProtectedBReadyAtBoundaryCount,
               statistics::units::Count::get(),
               "How many future-protected-B rows were already ready at boundary"),
      ADD_STAT(currentProtectedBIssueCount, statistics::units::Count::get(),
               "How many current-protected-B requests were issued"),
      ADD_STAT(currentProtectedBPriorityWins,
               statistics::units::Count::get(),
               "How many times current-protected-B won the second priority layer"),
      ADD_STAT(currentProtectedBBlocksACount,
               statistics::units::Count::get(),
               "How many times current-protected-B blocked an A issue opportunity"),
      ADD_STAT(currentProtectedBQuotaExhaustCount,
               statistics::units::Count::get(),
               "How many arbitration points saw current-protected-B quota exhausted"),
      ADD_STAT(currentProtectedBFromComputeWindowCount,
               statistics::units::Count::get(),
               "How many current-protected-B rows came from compute-window pressure"),
      ADD_STAT(currentProtectedBFromCurrentHoleFillingCount,
               statistics::units::Count::get(),
               "How many current-protected-B rows came from current hole-filling"),
      ADD_STAT(bRowsClaimedByFuture, statistics::units::Count::get(),
               "Peak number of current-tile B rows simultaneously claimed by future-B"),
      ADD_STAT(futureClaimSetCount, statistics::units::Count::get(),
               "How many future-B row claims were established"),
      ADD_STAT(futureClaimClearedCount, statistics::units::Count::get(),
               "How many future-B row claims were cleared"),
      ADD_STAT(futureClaimBlockedNormalFetchCount,
               statistics::units::Count::get(),
               "How many arbitration points saw a future claim block normal B issue"),
      ADD_STAT(futureClaimExpiredCount, statistics::units::Count::get(),
               "How many future-B row claims expired and released fallback"),
      ADD_STAT(futureClaimConsumedSuccessCount,
               statistics::units::Count::get(),
               "How many future-B row claims were fulfilled by future completions"),
      ADD_STAT(futureClaimInvalidatedCount,
               statistics::units::Count::get(),
               "How many future-B row claims were invalidated before success"),
      ADD_STAT(vipPoolCapacity, statistics::units::Count::get(),
               "Configured capacity of the guaranteed-local VIP B-row pool"),
      ADD_STAT(vipPoolOccupancyPeak, statistics::units::Count::get(),
               "Peak number of B rows resident in the VIP pool"),
      ADD_STAT(vipInsertCount, statistics::units::Count::get(),
               "How many B rows were inserted into the VIP pool"),
      ADD_STAT(vipHitCount, statistics::units::Count::get(),
               "How many current-tile B rows were served from the VIP pool"),
      ADD_STAT(vipMissCount, statistics::units::Count::get(),
               "How many VIP lookups found no matching retained B row"),
      ADD_STAT(vipEvictionCount, statistics::units::Count::get(),
               "How many VIP entries were evicted to admit newer rows"),
      ADD_STAT(vipHitOnNextOutputCount, statistics::units::Count::get(),
               "How many VIP hits came from next-output retained rows"),
      ADD_STAT(vipHitOnNextOutputImmediateCount,
               statistics::units::Count::get(),
               "How many VIP hits came from next-output/immediate rows"),
      ADD_STAT(vipHitOnNextOutputNearCount,
               statistics::units::Count::get(),
               "How many VIP hits came from next-output/near rows"),
      ADD_STAT(vipHitOnClaimedFutureBCount, statistics::units::Count::get(),
               "How many VIP hits came from claimed future-B rows"),
      ADD_STAT(vipHitOnCarryOverBCount, statistics::units::Count::get(),
               "How many VIP hits came from carry-over retained rows"),
      ADD_STAT(vipInsertFromNextOutputCount, statistics::units::Count::get(),
               "How many VIP inserts originated from next-output rows"),
      ADD_STAT(vipInsertFromClaimCount, statistics::units::Count::get(),
               "How many VIP inserts originated from claimed future-B rows"),
      ADD_STAT(vipInsertFromCarryOverCount, statistics::units::Count::get(),
               "How many VIP inserts originated from carry-over rows"),
      ADD_STAT(vipMaterializeToCurrentCount,
               statistics::units::Count::get(),
               "How many VIP rows were materialized into current-tile state"),
      ADD_STAT(vipBRowsServedToCompute, statistics::units::Count::get(),
               "How many B rows were supplied to compute via the VIP pool"),
      ADD_STAT(vipBRowsPreventedFallbackCount,
               statistics::units::Count::get(),
               "How many VIP hits avoided a normal remote B fallback fetch"),
      ADD_STAT(vipStrongAdmitCount, statistics::units::Count::get(),
               "How many rows were strongly admitted into the VIP pool"),
      ADD_STAT(vipWeakAdmitCount, statistics::units::Count::get(),
               "How many rows were weakly admitted into the VIP pool"),
      ADD_STAT(vipEvictLowPriorityCount,
               statistics::units::Count::get(),
               "How many VIP evictions removed lower-priority rows"),
      ADD_STAT(vipEvictWeakAdmitCount,
               statistics::units::Count::get(),
               "How many VIP evictions removed weak-admit rows"),
      ADD_STAT(vipEvictNormalCount, statistics::units::Count::get(),
               "How many VIP evictions removed normal-value rows"),
      ADD_STAT(vipAdmitNextOutputImmediateCount,
               statistics::units::Count::get(),
               "How many VIP admits came from next-output/immediate rows"),
      ADD_STAT(vipAdmitNextOutputNearCount,
               statistics::units::Count::get(),
               "How many VIP admits came from next-output/near rows"),
      ADD_STAT(vipAdmitNextOutputFarCount,
               statistics::units::Count::get(),
               "How many VIP admits came from next-output/far rows"),
      ADD_STAT(vipAdmitClaimCount, statistics::units::Count::get(),
               "How many VIP admits came from claimed future-B rows"),
      ADD_STAT(vipAdmitCarryOverCount, statistics::units::Count::get(),
               "How many VIP admits came from carry-over rows"),
      ADD_STAT(vipAdmitCurrentWindowImmediateCount,
               statistics::units::Count::get(),
               "How many VIP admits used the small current-window emergency reserve"),
      ADD_STAT(vipRejectNormalNearCount,
               statistics::units::Count::get(),
               "How many normal/near rows were rejected by the Oracle-guided VIP rules"),
      ADD_STAT(vipRejectNormalFarCount,
               statistics::units::Count::get(),
               "How many normal/far rows were rejected by the Oracle-guided VIP rules"),
      ADD_STAT(vipRejectOtherCount, statistics::units::Count::get(),
               "How many non-target rows were rejected by the Oracle-guided VIP rules"),
      ADD_STAT(oracleSelectedBRowsCount, statistics::units::Count::get(),
               "How many rows fell into Oracle-guided VIP-selected buckets"),
      ADD_STAT(oracleSelectedRowsServedByVipCount,
               statistics::units::Count::get(),
               "How many Oracle-selected rows were ultimately served by VIP"),
      ADD_STAT(oracleSelectedRowsMissedByVipCount,
               statistics::units::Count::get(),
               "How many Oracle-selected rows were not captured by VIP"),
      ADD_STAT(vipRescueInsertCount, statistics::units::Count::get(),
               "How many fallback-victim rows were admitted into the VIP rescue buffer"),
      ADD_STAT(vipRescueInsertStrongCount, statistics::units::Count::get(),
               "How many rescue-buffer admits satisfied the strong anti-dead-block rule"),
      ADD_STAT(vipRescueInsertWeakCount, statistics::units::Count::get(),
               "How many rescue-buffer admits satisfied the weak anti-dead-block rule"),
      ADD_STAT(vipRescueHitCount, statistics::units::Count::get(),
               "How many fallback-time VIP rescue lookups hit"),
      ADD_STAT(vipRescueMissCount, statistics::units::Count::get(),
               "How many fallback-time VIP rescue lookups missed"),
      ADD_STAT(vipRescueEvictionCount, statistics::units::Count::get(),
               "How many rescue-buffer admissions evicted an older rescue row"),
      ADD_STAT(vipRescueServedToComputeCount, statistics::units::Count::get(),
               "How many B rows were supplied to compute from the rescue buffer"),
      ADD_STAT(vipRescuePreventedFallbackCount,
               statistics::units::Count::get(),
               "How many remote fallback fetches were avoided by the rescue buffer"),
      ADD_STAT(vipRescueInsertAfterFallbackCount,
               statistics::units::Count::get(),
               "How many rescue-buffer inserts happened immediately after a fallback fetch"),
      ADD_STAT(vipRescueInsertShortNextUseCount,
               statistics::units::Count::get(),
               "How many rescue-buffer inserts had short estimated next-use distance"),
      ADD_STAT(vipRescueInsertMultiFutureUseCount,
               statistics::units::Count::get(),
               "How many rescue-buffer inserts had multiple future tile reuses remaining"),
      ADD_STAT(vipRescueReusedCount, statistics::units::Count::get(),
               "How many rescue-buffer rows were reused by a later fallback rescue hit"),
      ADD_STAT(vipRescueRejectNonFallbackCount,
               statistics::units::Count::get(),
               "How many rescue candidates were rejected because they still belonged to the mainline future chain"),
      ADD_STAT(vipRescueRejectNotShortUseCount,
               statistics::units::Count::get(),
               "How many rescue candidates were rejected because the next-use distance was too long"),
      ADD_STAT(vipRescueRejectNotCriticalCount,
               statistics::units::Count::get(),
               "How many rescue candidates were rejected because they were not in a critical row/window"),
      ADD_STAT(vipRescueRejectNotRescueCriticalCount,
               statistics::units::Count::get(),
               "How many rescue candidates were rejected because rescue recurrence criticality was too low"),
      ADD_STAT(vipRescueRejectSingleUseCount,
               statistics::units::Count::get(),
               "How many rescue candidates were rejected because no future tile reuse remained"),
      ADD_STAT(vipRescueRejectNonRepeatCount,
               statistics::units::Count::get(),
               "How many rescue candidates were rejected because they were neither repeated victims nor clearly multi-reuse"),
      ADD_STAT(repeatVictimCountTotal, statistics::units::Count::get(),
               "How many fallback victims had already repeated at least once"),
      ADD_STAT(repeatVictimPromotedCount, statistics::units::Count::get(),
               "How many repeated fallback victims were admitted into VIP"),
      ADD_STAT(criticalWindowVictimCount, statistics::units::Count::get(),
               "How many fallback victims landed in the critical B launch/window region"),
      ADD_STAT(criticalWindowVipInsertCount, statistics::units::Count::get(),
               "How many critical-window fallback victims were admitted into VIP"),
      ADD_STAT(weakRescuePromotedOnFirstFallbackCount,
               statistics::units::Count::get(),
               "How many first-seen fallback victims were weakly promoted into VIP"),
      ADD_STAT(strongRescuePromotedOnRecurrenceCount,
               statistics::units::Count::get(),
               "How many recurring fallback victims were strongly promoted into VIP"),
      ADD_STAT(rescueCriticalityHighCount, statistics::units::Count::get(),
               "How many fallback victims met the recurrence-aware rescue criticality test"),
      ADD_STAT(nextRescueOpportunityNearCount,
               statistics::units::Count::get(),
               "How many fallback victims had another likely rescue opportunity in a very short tile distance"),
      ADD_STAT(eligibleRescueVictimsFirstSeenCount,
               statistics::units::Count::get(),
               "How many first-seen fallback victims were eligible for an early rescue decision"),
      ADD_STAT(eligibleRescueVictimsPromotedEarlyCount,
               statistics::units::Count::get(),
               "How many eligible first-seen fallback victims were promoted into VIP immediately"),
      ADD_STAT(eligibleRescueVictimsHitAfterFirstPromotionCount,
               statistics::units::Count::get(),
               "How many rescue hits came from rows first admitted on their initial fallback"),
      ADD_STAT(vipRescueAInsertCount, statistics::units::Count::get(),
               "How many A rows were admitted into the shared VIP rescue buffer"),
      ADD_STAT(vipRescueAHitCount, statistics::units::Count::get(),
               "How many A-row VIP rescue lookups hit"),
      ADD_STAT(vipRescueAMissCount, statistics::units::Count::get(),
               "How many A-row VIP rescue lookups missed"),
      ADD_STAT(vipRescueAEvictionCount, statistics::units::Count::get(),
               "How many rescue-buffer admissions evicted an older A rescue row"),
      ADD_STAT(vipRescueAServedToComputeCount,
               statistics::units::Count::get(),
               "How many A rows were supplied to compute from the rescue buffer"),
      ADD_STAT(vipRescueAPreventedRemoteFetchCount,
               statistics::units::Count::get(),
               "How many remote A fetches were avoided by the rescue buffer"),
      ADD_STAT(vipRescueAInsertShortNextUseCount,
               statistics::units::Count::get(),
               "How many A rescue-buffer inserts had short estimated next-use distance"),
      ADD_STAT(vipRescueAInsertMultiFutureUseCount,
               statistics::units::Count::get(),
               "How many A rescue-buffer inserts had multiple future tile reuses remaining"),
      ADD_STAT(vipRescueAReusedCount, statistics::units::Count::get(),
               "How many A rescue-buffer rows were reused by a later hit"),
      ADD_STAT(mhotPoolCapacity, statistics::units::Count::get(),
               "Configured capacity of the Matrix-side B-mainline hot pool"),
      ADD_STAT(mhotOccupancyPeak, statistics::units::Count::get(),
               "Peak number of B rows resident in the M-hot pool"),
      ADD_STAT(mhotInsertCount, statistics::units::Count::get(),
               "How many B-side mainline rows were inserted into M-hot"),
      ADD_STAT(mhotInsertDefaultCount, statistics::units::Count::get(),
               "How many M-hot inserts came from the default mainline rule"),
      ADD_STAT(mhotInsertEnhancedCount, statistics::units::Count::get(),
               "How many M-hot inserts came from the enhanced mainline rule"),
      ADD_STAT(mhotInsertNextOutputImmediateCount,
               statistics::units::Count::get(),
               "How many M-hot inserts came from next-output/immediate rows"),
      ADD_STAT(mhotInsertNextOutputNearCount,
               statistics::units::Count::get(),
               "How many M-hot inserts came from next-output/near rows"),
      ADD_STAT(mhotInsertNextOutputFarCount,
               statistics::units::Count::get(),
               "How many M-hot inserts came from next-output/far rows"),
      ADD_STAT(mhotInsertClaimCount, statistics::units::Count::get(),
               "How many M-hot inserts came from claim rows"),
      ADD_STAT(mhotInsertCarryOverCount, statistics::units::Count::get(),
               "How many M-hot inserts came from carry-over rows"),
      ADD_STAT(mhotInsertNormalNearCount, statistics::units::Count::get(),
               "How many M-hot inserts came from normal/near rows"),
      ADD_STAT(mhotInsertCurrentWindowFarCount,
               statistics::units::Count::get(),
               "How many M-hot inserts came from current-window/far rows"),
      ADD_STAT(mhotInsertCoverageBlindspotCount,
               statistics::units::Count::get(),
               "How many M-hot inserts came from coverage-blindspot normal rows"),
      ADD_STAT(mhotHitCount, statistics::units::Count::get(),
               "How many B rows were served from M-hot instead of remote mainline supply"),
      ADD_STAT(mhotHitOnNextOutputImmediateCount,
               statistics::units::Count::get(),
               "How many M-hot hits came from next-output/immediate rows"),
      ADD_STAT(mhotHitOnNextOutputNearCount,
               statistics::units::Count::get(),
               "How many M-hot hits came from next-output/near rows"),
      ADD_STAT(mhotHitOnNextOutputFarCount,
               statistics::units::Count::get(),
               "How many M-hot hits came from next-output/far rows"),
      ADD_STAT(mhotHitOnClaimCount, statistics::units::Count::get(),
               "How many M-hot hits came from claim rows"),
      ADD_STAT(mhotHitOnCarryOverCount, statistics::units::Count::get(),
               "How many M-hot hits came from carry-over rows"),
      ADD_STAT(mhotHitOnNormalNearCount, statistics::units::Count::get(),
               "How many M-hot hits came from normal/near rows"),
      ADD_STAT(mhotHitOnCurrentWindowFarCount,
               statistics::units::Count::get(),
               "How many M-hot hits came from current-window/far rows"),
      ADD_STAT(mhotHitOnCoverageBlindspotCount,
               statistics::units::Count::get(),
               "How many M-hot hits came from coverage-blindspot normal rows"),
      ADD_STAT(mhotRowsServedToComputeCount,
               statistics::units::Count::get(),
               "How many B rows reached current-tile compute through M-hot"),
      ADD_STAT(mhotMaterializeToCurrentCount,
               statistics::units::Count::get(),
               "How many M-hot rows were materialized into current-tile state"),
      ADD_STAT(mhotEvictionCount, statistics::units::Count::get(),
               "How many M-hot entries were evicted to admit newer rows"),
      ADD_STAT(mhotEvictionCoverageBlindspotCount,
               statistics::units::Count::get(),
               "How many coverage-blindspot entries were evicted from M-hot"),
      ADD_STAT(mhotReuseHitCount, statistics::units::Count::get(),
               "How many M-hot hits reused a previously retained row"),
      ADD_STAT(mhotMainlinePreventedRemoteCount,
               statistics::units::Count::get(),
               "How many remote mainline B fetches were avoided by M-hot"),
      ADD_STAT(mhotRowsServedToComputeFromCoverageBlindspotCount,
               statistics::units::Count::get(),
               "How many B rows reached compute through coverage-blindspot M-hot hits"),
      ADD_STAT(mhotMainlinePreventedRemoteFromCoverageBlindspotCount,
               statistics::units::Count::get(),
               "How many remote fetches were avoided specifically by coverage-blindspot M-hot hits"),
      ADD_STAT(mhotCheckedOnFallbackCount, statistics::units::Count::get(),
               "How many current B fallback opportunities checked M-hot by hot-object label"),
      ADD_STAT(mhotMissThenRemoteCount, statistics::units::Count::get(),
               "How many fallback-time M-hot lookups missed and continued to remote fetch"),
      ADD_STAT(mhotHitBeforeRemoteCount, statistics::units::Count::get(),
               "How many fallback-time M-hot lookups hit before remote fetch"),
      ADD_STAT(coverageShadowPoolCapacity, statistics::units::Count::get(),
               "Configured capacity of the coverage-shadow SRAM slice"),
      ADD_STAT(coverageShadowOccupancyPeak, statistics::units::Count::get(),
               "Peak number of rows resident in the coverage-shadow slice"),
      ADD_STAT(coverageShadowLowPriorityOccupancyPeak,
               statistics::units::Count::get(),
               "Peak number of low-priority rows resident in the coverage-shadow slice"),
      ADD_STAT(coverageShadowInsertCount, statistics::units::Count::get(),
               "How many rows were inserted into the coverage-shadow slice"),
      ADD_STAT(coverageShadowInsertFromNextKCount,
               statistics::units::Count::get(),
               "How many coverage-shadow inserts came from next-k prefetch completions"),
      ADD_STAT(coverageShadowInsertFromNextOutputCount,
               statistics::units::Count::get(),
               "How many coverage-shadow inserts came from next-output prefetch completions"),
      ADD_STAT(coverageShadowInsertFromGatherCount,
               statistics::units::Count::get(),
               "How many coverage-shadow inserts came from dedicated 2D gather completions"),
      ADD_STAT(coverageShadowHitCount, statistics::units::Count::get(),
               "How many current-tile B lookups hit in the coverage-shadow slice"),
      ADD_STAT(coverageShadowRowsServedToComputeCount,
               statistics::units::Count::get(),
               "How many B rows reached compute through coverage-shadow hits"),
      ADD_STAT(coverageShadowPreventedRemoteCount,
               statistics::units::Count::get(),
               "How many remote fetches were avoided by the coverage-shadow slice"),
      ADD_STAT(coverageShadowEvictionCount, statistics::units::Count::get(),
               "How many coverage-shadow entries were evicted to admit newer rows"),
      ADD_STAT(coverageShadowCheckedOnFallbackCount,
               statistics::units::Count::get(),
               "How many fallback opportunities checked the coverage-shadow slice"),
      ADD_STAT(coverageShadowMissThenRemoteCount,
               statistics::units::Count::get(),
               "How many coverage-shadow checks missed and continued to remote"),
      ADD_STAT(coverageShadowHitBeforeRemoteCount,
               statistics::units::Count::get(),
               "How many coverage-shadow checks hit before remote fetch"),
      ADD_STAT(coverageShadowPrimeCheckCount,
               statistics::units::Count::get(),
               "How many tile-entry coverage-shadow prime lookups were attempted"),
      ADD_STAT(coverageShadowPrimeHitCount,
               statistics::units::Count::get(),
               "How many tile-entry coverage-shadow prime lookups hit"),
      ADD_STAT(coverageShadowPrimeMissCount,
               statistics::units::Count::get(),
               "How many tile-entry coverage-shadow prime lookups missed"),
      ADD_STAT(coverageShadowPrimeRowsMaterializedCount,
               statistics::units::Count::get(),
               "How many B rows were materialized from coverage-shadow during tile-entry priming"),
      ADD_STAT(coverageShadowFallbackLookupSkippedCount,
               statistics::units::Count::get(),
               "How many fallback-time coverage-shadow lookups were skipped due to an earlier known miss"),
      ADD_STAT(coverageGatherIssueCount, statistics::units::Count::get(),
               "How many dedicated B-side 2D gather reads were issued"),
      ADD_STAT(coverageGatherCompletionCount, statistics::units::Count::get(),
               "How many dedicated B-side 2D gather reads completed"),
      ADD_STAT(coverageGatherLateCompletionCount, statistics::units::Count::get(),
               "How many stale dedicated 2D gather completions were ignored"),
      ADD_STAT(coverageGatherTargetRows, statistics::units::Count::get(),
               "How many current-tile B rows matched the 2D gather blind-spot family"),
      ADD_STAT(coverageGatherCompletionWhileRowEmptyCount,
               statistics::units::Count::get(),
               "How many 2D gather completions arrived while the target current-tile row was still Empty"),
      ADD_STAT(coverageGatherCompletionWhileRowInflightCount,
               statistics::units::Count::get(),
               "How many 2D gather completions arrived while the target current-tile row was already Inflight"),
      ADD_STAT(coverageGatherCompletionWhileRowReadyCount,
               statistics::units::Count::get(),
               "How many 2D gather completions arrived while the target current-tile row was already Ready"),
      ADD_STAT(coverageGatherCompletionWhileRowConsumedCount,
               statistics::units::Count::get(),
               "How many 2D gather completions arrived after the target current-tile row was already Consumed"),
      ADD_STAT(coverageGatherCompletionDowngradedCount,
               statistics::units::Count::get(),
               "How many 2D gather completions were downgraded to low priority"),
      ADD_STAT(coverageGatherCompletionInsertedLowPriorityCount,
               statistics::units::Count::get(),
               "How many 2D gather completions were inserted as low-priority shadow rows"),
      ADD_STAT(coverageGatherCompletionDiscardedCount,
               statistics::units::Count::get(),
               "How many 2D gather completions were discarded after downgrade"),
      ADD_STAT(gatherMinBudgetReservedCount,
               statistics::units::Count::get(),
               "How many times a minimum gather budget was reserved per tile"),
      ADD_STAT(gatherBudgetGrantedCount,
               statistics::units::Count::get(),
               "How many gather issues were granted under the minimum budget"),
      ADD_STAT(gatherDeferredByMainlineCount,
               statistics::units::Count::get(),
               "How many gather opportunities were deferred to preserve mainline progress"),
      ADD_STAT(mainlineCreditReservedCount,
               statistics::units::Count::get(),
               "How many arbitration points explicitly reserved B-side credit for mainline traffic"),
      ADD_STAT(gatherBlockedByMainlineGraceWindowCount,
               statistics::units::Count::get(),
               "How many gather opportunities were blocked by the no-starvation mainline grace window"),
      ADD_STAT(gatherDeferredByVipCount,
               statistics::units::Count::get(),
               "How many gather opportunities yielded because VIP rescue took precedence"),
      ADD_STAT(vipDeferredByGatherCount,
               statistics::units::Count::get(),
               "How many VIP rescue opportunities were delayed by gather arbitration"),
      ADD_STAT(vipRescueModeActiveCount,
               statistics::units::Count::get(),
               "How many tiles ran with VIP rescue protection active in the current mode"),
      ADD_STAT(gatherBudgetExhaustedCount,
               statistics::units::Count::get(),
               "How many times gather hit its independent per-tile budget"),
      ADD_STAT(gatherMaxConsecutiveIssueHitsCount,
               statistics::units::Count::get(),
               "How many times gather hit the consecutive-issue cap inside its own queue"),
      ADD_STAT(gatherCandidateAgedUpCount,
               statistics::units::Count::get(),
               "How many target gather candidates were aged up while waiting in the gather queue"),
      ADD_STAT(gatherDroppedDueToDeadlineCount,
               statistics::units::Count::get(),
               "How many target gather candidates expired before they could be issued"),
      ADD_STAT(mhotCoverageBlindspotSeenCount,
               statistics::units::Count::get(),
               "How many normal coverage-blindspot fallback rows were observed"),
      ADD_STAT(mhotCoverageBlindspotRepeatCount,
               statistics::units::Count::get(),
               "How many coverage-blindspot rows were seen at least twice"),
      ADD_STAT(mhotCoverageBlindspotPromotedCount,
               statistics::units::Count::get(),
               "How many coverage-blindspot rows were promoted into M-hot"),
      ADD_STAT(mhotCoverageBlindspotRejectSingleCount,
               statistics::units::Count::get(),
               "How many coverage-blindspot rows were rejected on first sighting"),
      ADD_STAT(mhotCoverageBlindspotRejectLongDistanceCount,
               statistics::units::Count::get(),
               "How many coverage-blindspot rows were rejected because next use was too far"),
      ADD_STAT(mhotCoverageBlindspotRejectLowReuseCount,
               statistics::units::Count::get(),
               "How many coverage-blindspot rows were rejected because future reuse was weak"),
      ADD_STAT(repeatCoverageVictimCountTotal,
               statistics::units::Count::get(),
               "How many coverage-blindspot fallbacks came from repeated normal victims"),
      ADD_STAT(repeatCoverageVictimPromotedCount,
               statistics::units::Count::get(),
               "How many repeated coverage-blindspot victims were promoted into M-hot"),
      ADD_STAT(coverageBlindspotShortNextUseCount,
               statistics::units::Count::get(),
               "How many coverage-blindspot rows had short tile-steps to next use"),
      ADD_STAT(coverageBlindspotFutureReuseGt1Count,
               statistics::units::Count::get(),
               "How many coverage-blindspot rows had future_i_tile_reuse_remaining > 1"),
      ADD_STAT(coverageBlindspotFirstSeenMissCount,
               statistics::units::Count::get(),
               "How many Normal+Coverage misses were first-seen pattern misses"),
      ADD_STAT(coverageBlindspotRepeatMissCount,
               statistics::units::Count::get(),
               "How many Normal+Coverage misses repeated a previously seen pattern"),
      ADD_STAT(coverageBlindspotDistinctPatternCount,
               statistics::units::Count::get(),
               "How many distinct Normal+Coverage patterns were observed"),
      ADD_STAT(coverageBlindspotTargetMissCount,
               statistics::units::Count::get(),
               "How many Normal+Coverage misses matched the target blind-spot family"),
      ADD_STAT(coverageBlindspotTargetFirstSeenMissCount,
               statistics::units::Count::get(),
               "How many target-family Normal+Coverage misses were first-seen pattern misses"),
      ADD_STAT(coverageBlindspotTargetRepeatMissCount,
               statistics::units::Count::get(),
               "How many target-family Normal+Coverage misses repeated a previously seen pattern"),
      ADD_STAT(coverageBlindspotTargetDistinctPatternCount,
               statistics::units::Count::get(),
               "How many distinct target-family Normal+Coverage patterns were observed"),
      ADD_STAT(smartPrefetchAIssueCount,
               statistics::units::Count::get(),
               "How many A rows were issued via smart pattern-aware ordering"),
      ADD_STAT(smartPrefetchARepeatPriorityCount,
               statistics::units::Count::get(),
               "How many smart A issues prioritized repeat-victim rows"),
      ADD_STAT(smartPrefetchANearMultiPriorityCount,
               statistics::units::Count::get(),
               "How many smart A issues prioritized near/multi-reuse rows"),
      ADD_STAT(smartPrefetchBNextOutputPriorityIssueCount,
               statistics::units::Count::get(),
               "How many next-output B prefetch issues used smart pattern-aware ordering"),
      ADD_STAT(smartPrefetchBNextOutputCoverageTargetIssueCount,
               statistics::units::Count::get(),
               "How many smart next-output B prefetch issues targeted coverage-blindspot far rows"),
      ADD_STAT(smartPrefetchBNextKPriorityIssueCount,
               statistics::units::Count::get(),
               "How many next-k B prefetch issues used smart pattern-aware ordering"),
      ADD_STAT(fallbackAutopsyCoverageNormalByDistance,
               statistics::units::Count::get(),
               "How many Normal+Coverage fallbacks appeared in each distance bucket"),
      ADD_STAT(coverageBlindspotSeenByDistance,
               statistics::units::Count::get(),
               "How many coverage-blindspot candidates were observed in each distance bucket"),
      ADD_STAT(coverageBlindspotPromotedByDistance,
               statistics::units::Count::get(),
               "How many coverage-blindspot candidates were promoted in each distance bucket"),
      ADD_STAT(mhotHitOnCoverageBlindspotByDistance,
               statistics::units::Count::get(),
               "How many coverage-blindspot M-hot hits came from each distance bucket"),
      ADD_STAT(coverageBlindspotSeenByReuseBucket,
               statistics::units::Count::get(),
               "How many coverage-blindspot candidates were observed in each future-reuse bucket"),
      ADD_STAT(coverageBlindspotPromotedByReuseBucket,
               statistics::units::Count::get(),
               "How many coverage-blindspot candidates were promoted in each future-reuse bucket"),
      ADD_STAT(coverageBlindspotSeenByTileBand,
               statistics::units::Count::get(),
               "How many coverage-blindspot candidates were observed in each tile-step band"),
      ADD_STAT(coverageBlindspotPromotedByTileBand,
               statistics::units::Count::get(),
               "How many coverage-blindspot candidates were promoted in each tile-step band"),
      ADD_STAT(rxAIssueCount, statistics::units::Count::get(),
               "How many rows the logical RX-A path issued"),
      ADD_STAT(rxBIssueCount, statistics::units::Count::get(),
               "How many rows the logical RX-B path issued"),
      ADD_STAT(rxAReadyCount, statistics::units::Count::get(),
               "How many A rows became ready through RX-A"),
      ADD_STAT(rxBReadyCount, statistics::units::Count::get(),
               "How many B rows became ready through RX-B"),
      ADD_STAT(rxAQueueOccupancyPeak, statistics::units::Count::get(),
               "Peak logical RX-A queue occupancy"),
      ADD_STAT(rxBQueueOccupancyPeak, statistics::units::Count::get(),
               "Peak logical RX-B queue occupancy"),
      ADD_STAT(rxAStallCycles, statistics::units::Count::get(),
               "How many dual-RX arbitration rounds saw RX-A pending but not issue"),
      ADD_STAT(rxBStallCycles, statistics::units::Count::get(),
               "How many dual-RX arbitration rounds saw RX-B pending but not issue"),
      ADD_STAT(rxAPriorityWins, statistics::units::Count::get(),
               "How many dual-RX arbitration wins went to RX-A"),
      ADD_STAT(rxBPriorityWins, statistics::units::Count::get(),
               "How many dual-RX arbitration wins went to RX-B"),
      ADD_STAT(rxBDeficitWins, statistics::units::Count::get(),
               "How many dual-RX arbitration wins went to RX-B because of deficit"),
      ADD_STAT(rxADeficitWins, statistics::units::Count::get(),
               "How many dual-RX arbitration wins went to RX-A because of deficit"),
      ADD_STAT(rxBDeadlineWins, statistics::units::Count::get(),
               "How many dual-RX arbitration wins went to RX-B because of deadline/urgency"),
      ADD_STAT(rxADeadlineWins, statistics::units::Count::get(),
               "How many dual-RX arbitration wins went to RX-A because of deadline/urgency"),
      ADD_STAT(oracleLocalRows, statistics::units::Count::get(),
               "Oracle/profiler local-served B rows by source class and next-use distance"),
      ADD_STAT(oracleFallbackRows, statistics::units::Count::get(),
               "Oracle/profiler fallback-served B rows by source class and next-use distance"),
      ADD_STAT(oracleLocalReuseWeight, statistics::units::Count::get(),
               "Oracle/profiler local-served B reuse weight by source class and next-use distance"),
      ADD_STAT(oracleFallbackReuseWeight, statistics::units::Count::get(),
               "Oracle/profiler fallback-served B reuse weight by source class and next-use distance"),
      ADD_STAT(oracleALocalByDistance, statistics::units::Count::get(),
               "Oracle/profiler local-served A rows by next-use distance"),
      ADD_STAT(oracleARemoteByDistance, statistics::units::Count::get(),
               "Oracle/profiler remote-served A rows by next-use distance"),
      ADD_STAT(oracleALocalByReuse, statistics::units::Count::get(),
               "Oracle/profiler local-served A rows by future-reuse bucket"),
      ADD_STAT(oracleARemoteByReuse, statistics::units::Count::get(),
               "Oracle/profiler remote-served A rows by future-reuse bucket"),
      ADD_STAT(oracleALocalByRecurrence, statistics::units::Count::get(),
               "Oracle/profiler local-served A rows by remote-victim recurrence"),
      ADD_STAT(oracleARemoteByRecurrence, statistics::units::Count::get(),
               "Oracle/profiler remote-served A rows by remote-victim recurrence"),
      ADD_STAT(fallbackAutopsyTimelinessCount, statistics::units::Count::get(),
               "How many B fallbacks were caused by prefetch/claim arriving too late"),
      ADD_STAT(fallbackAutopsyChurnCount, statistics::units::Count::get(),
               "How many B fallbacks were caused by nearby-pool churn/eviction"),
      ADD_STAT(fallbackAutopsyCoverageCount, statistics::units::Count::get(),
               "How many B fallbacks had no hot/future-chain coverage before remote"),
      ADD_STAT(fallbackAutopsyRows, statistics::units::Count::get(),
               "B fallback autopsy rows by oracle source and death-cause bucket")
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
    hierarchicalDebugComputeDeferCount = 0;
    reqsIssuedA = reqsCompletedA = targetReqsA = 0;
    reqsIssuedB = reqsCompletedB = targetReqsB = 0;
    nextNormalBRowCursor = 0;
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
    lastFetchIssueWasA = false;
    abParallelOverlapActive = false;
    abParallelOverlapStartTick = 0;
    aWaitedForBThisTile = false;
    bWaitedForAThisTile = false;
    aInflightPeakObserved = 0;
    bInflightPeakObserved = 0;
    currentProtectedBWinStreak = 0;
    vipPoolOccupancy = 0;
    vipPoolOccupancyPeakObserved = 0;
    vipInsertEpoch = 0;
    mhotPoolOccupancy = 0;
    mhotPoolOccupancyPeakObserved = 0;
    mhotInsertEpoch = 0;
    aRemoteVictimCount.clear();
    bFallbackVictimCount.clear();
    bCoverageBlindspotPattern.clear();
    bCoverageBlindspotMissPatternCount.clear();
    std::fill(currentARowIssued.begin(), currentARowIssued.end(), false);
    std::fill(nextPrefetchARowIssued.begin(), nextPrefetchARowIssued.end(), false);
    std::fill(nextPrefetchBRowIssued.begin(), nextPrefetchBRowIssued.end(), false);
    std::fill(nextOutputPrefetchBRowIssued.begin(),
              nextOutputPrefetchBRowIssued.end(), false);
    clearCoverageGatherForCurrentTile();
    clearStagedCurrentVipBacked();
    clearStagedCurrentOracleVipSelected();
    clearCurrentVipBacked();
    clearCurrentOracleVipSelected();
    std::fill(currentBRowState.begin(), currentBRowState.end(),
              BRowState::Empty);
    std::fill(currentBProtectionClass.begin(), currentBProtectionClass.end(),
              BProtectionClass::None);
    clearNextOutputFutureClaims();
    clearStagedCurrentFutureClaims();
    clearCurrentFutureClaims();
    clearVipPool();
    clearMHotPool();
    clearCoverageShadowPool();
    clearNextTilePrefetch();
    clearNextOutputTilePrefetch();
    stats.vipPoolCapacity = vipBRowsCapacityConfig;
    stats.mhotPoolCapacity = mhotBRowsCapacityConfig;
    stats.coverageShadowPoolCapacity = coverageShadowRowsCapacityConfig;
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

bool
MatrixFlowEngine::baselineSchedulerMode() const
{
    return abSchedulerMode == ABSchedulerMode::Baseline;
}

bool
MatrixFlowEngine::parallelABSchedulerMode() const
{
    return abSchedulerMode != ABSchedulerMode::Baseline;
}

uint32_t
MatrixFlowEngine::computeLaunchWindowA() const
{
    if (baselineSchedulerMode()) {
        return targetReqsA;
    }
    if (abMinLaunchRowsAConfig > 0) {
        return std::min(targetReqsA, abMinLaunchRowsAConfig);
    }
    return std::min(targetReqsA, static_cast<uint32_t>(macArraySize));
}

uint32_t
MatrixFlowEngine::computeLaunchWindowB() const
{
    if (baselineSchedulerMode()) {
        return targetReqsB;
    }
    if (abMinLaunchRowsBConfig > 0) {
        return std::min(targetReqsB, abMinLaunchRowsBConfig);
    }
    return std::min(targetReqsB, static_cast<uint32_t>(macArraySize));
}

bool
MatrixFlowEngine::minimumInputsReadyForCompute() const
{
    return reqsCompletedA >= computeLaunchWindowA() &&
           reqsCompletedB >= computeLaunchWindowB();
}

bool
MatrixFlowEngine::fullInputsReadyForAccumulate() const
{
    return reqsCompletedA == targetReqsA &&
           reqsCompletedB == targetReqsB;
}

bool
MatrixFlowEngine::inputsReadyForCompute() const
{
    if (baselineSchedulerMode()) {
        return fullInputsReadyForAccumulate();
    }
    return minimumInputsReadyForCompute();
}

bool
MatrixFlowEngine::canIssueA() const
{
    return reqsIssuedA < targetReqsA &&
           satOutstanding(reqsIssuedA, reqsCompletedA) < kMaxInFlight;
}

uint32_t
MatrixFlowEngine::normalIssueLimit() const
{
    // Once compute is already in flight, the current tile must eventually
    // drain all remaining B rows. If the carry-over / hole-fill lead cap has
    // throttled normal B issue and all in-flight B has already drained, keep
    // issuing the tail rows instead of livelocking in computeDone deferrals.
    if (!baselineSchedulerMode() &&
        computeBusy &&
        reqsCompletedB < targetReqsB &&
        reqsCompletedB == reqsIssuedB) {
        return targetReqsB;
    }

    const bool carryOverInflightOutstanding =
        carryOverBActive && (carryOverBRowsCompleted < carryOverBRowsIssued);
    if (carryOverInflightOutstanding && holeFillLeadRowsConfig > 0) {
        return std::min(targetReqsB,
                        carryOverBRowsIssued + holeFillLeadRowsConfig);
    }
    return targetReqsB;
}

int
MatrixFlowEngine::nextIssuableBRow() const
{
    if (reqsIssuedB >= targetReqsB ||
        satOutstanding(reqsIssuedB, reqsCompletedB) >= kMaxInFlight) {
        return -1;
    }

    const uint32_t limit = normalIssueLimit();
    for (uint32_t row = nextNormalBRowCursor; row < limit; ++row) {
        if (currentBRowState[row] == BRowState::Empty &&
            !isClaimedByFuture(row)) {
            return static_cast<int>(row);
        }
    }

    // Some parallel A/B modes can end up with the cursor advanced past an
    // earlier empty row even though the current tile still has unissued B
    // work. Fall back to a full rescan before declaring there is no issuable
    // B row, otherwise computeDone can livelock waiting for rows that the
    // scheduler has effectively lost.
    for (uint32_t row = 0; row < std::min(limit, nextNormalBRowCursor); ++row) {
        if (currentBRowState[row] == BRowState::Empty &&
            !isClaimedByFuture(row)) {
            if (hierarchicalProtectedBSchedulerMode()) {
                warn("%s: hier-trace recovered skipped B row=%u cursor=%u "
                     "issuedB=%u completedB=%u targetB=%u\n",
                     name(), row, nextNormalBRowCursor, reqsIssuedB,
                     reqsCompletedB, targetReqsB);
            }
            return static_cast<int>(row);
        }
    }
    return -1;
}

bool
MatrixFlowEngine::canIssueB() const
{
    return nextIssuableBRow() >= 0;
}

bool
MatrixFlowEngine::isClaimedByFuture(uint32_t rowIdx) const
{
    return claimBasedHoleFillingMode() &&
           rowIdx < currentBFutureClaimed.size() &&
           currentBFutureClaimed[rowIdx];
}

int
MatrixFlowEngine::nextClaimBlockedBRow() const
{
    if (!claimBasedHoleFillingMode() || reqsIssuedB >= targetReqsB ||
        satOutstanding(reqsIssuedB, reqsCompletedB) >= kMaxInFlight) {
        return -1;
    }

    const uint32_t limit = normalIssueLimit();
    for (uint32_t row = nextNormalBRowCursor; row < limit; ++row) {
        if (currentBRowState[row] == BRowState::Empty &&
            isClaimedByFuture(row)) {
            return static_cast<int>(row);
        }
    }
    for (uint32_t row = 0; row < std::min(limit, nextNormalBRowCursor); ++row) {
        if (currentBRowState[row] == BRowState::Empty &&
            isClaimedByFuture(row)) {
            return static_cast<int>(row);
        }
    }
    return -1;
}

bool
MatrixFlowEngine::protectedBSchedulerMode() const
{
    return abSchedulerMode == ABSchedulerMode::ProtectedB;
}

bool
MatrixFlowEngine::hierarchicalProtectedBSchedulerMode() const
{
    return abSchedulerMode == ABSchedulerMode::HierarchicalProtectedB ||
           abSchedulerMode == ABSchedulerMode::HierarchicalClaimHoleFilling ||
           abSchedulerMode == ABSchedulerMode::HierarchicalClaimVIPPool ||
           abSchedulerMode == ABSchedulerMode::VIPOracleGuidedSingleRun ||
           abSchedulerMode == ABSchedulerMode::VIPOracleGuidedMainlineOnly ||
           abSchedulerMode == ABSchedulerMode::VIPRescueBufferSingleRun ||
           abSchedulerMode == ABSchedulerMode::BVIPRescueAntiDeadBlockAdmission ||
           abSchedulerMode == ABSchedulerMode::BVIPRescueRecurrenceAwareV2 ||
           abSchedulerMode == ABSchedulerMode::BVIPRescueRecurrenceAwareV3 ||
           abSchedulerMode == ABSchedulerMode::BMHotMainlineDefault ||
           abSchedulerMode == ABSchedulerMode::BMHotMainlineEnhanced ||
           abSchedulerMode == ABSchedulerMode::BMHotRuntimeFirstCut ||
           abSchedulerMode == ABSchedulerMode::BMHotGapAwareNextCut ||
           abSchedulerMode ==
               ABSchedulerMode::BMHotCoverageBlindspotCandidateFirstCut ||
           abSchedulerMode ==
               ABSchedulerMode::BMHotCoverageBlindspotCandidateV2 ||
           abSchedulerMode ==
               ABSchedulerMode::BCoverageShadowControllerFirstCut ||
           abSchedulerMode ==
               ABSchedulerMode::BCoverageShadowControllerV2 ||
           abSchedulerMode ==
               ABSchedulerMode::BCoverage2DGatherFirstCut ||
           abSchedulerMode ==
               ABSchedulerMode::BCoverage2DGatherV2 ||
           abSchedulerMode ==
               ABSchedulerMode::BCoverage2DGatherNoStarvationV3 ||
           abSchedulerMode ==
               ABSchedulerMode::BCoverage2DGatherMinGuaranteeFirstCut ||
           abSchedulerMode ==
               ABSchedulerMode::BCoverage2DGatherPingPongFirstCut ||
           abSchedulerMode == ABSchedulerMode::ABSmartPatternPrefetchFirstCut ||
           abSchedulerMode == ABSchedulerMode::VIPABRescueBufferSingleRun ||
           abSchedulerMode == ABSchedulerMode::DualRXABHierarchicalVIP;
}

bool
MatrixFlowEngine::claimBasedHoleFillingMode() const
{
    return abSchedulerMode == ABSchedulerMode::HierarchicalClaimHoleFilling ||
           abSchedulerMode == ABSchedulerMode::HierarchicalClaimVIPPool ||
           abSchedulerMode == ABSchedulerMode::VIPOracleGuidedSingleRun ||
           abSchedulerMode == ABSchedulerMode::VIPOracleGuidedMainlineOnly ||
           abSchedulerMode == ABSchedulerMode::VIPRescueBufferSingleRun ||
           abSchedulerMode == ABSchedulerMode::BVIPRescueAntiDeadBlockAdmission ||
           abSchedulerMode == ABSchedulerMode::BVIPRescueRecurrenceAwareV2 ||
           abSchedulerMode == ABSchedulerMode::BVIPRescueRecurrenceAwareV3 ||
           abSchedulerMode == ABSchedulerMode::BMHotMainlineDefault ||
           abSchedulerMode == ABSchedulerMode::BMHotMainlineEnhanced ||
           abSchedulerMode == ABSchedulerMode::BMHotRuntimeFirstCut ||
           abSchedulerMode == ABSchedulerMode::BMHotGapAwareNextCut ||
           abSchedulerMode ==
               ABSchedulerMode::BMHotCoverageBlindspotCandidateFirstCut ||
           abSchedulerMode ==
               ABSchedulerMode::BMHotCoverageBlindspotCandidateV2 ||
           abSchedulerMode ==
               ABSchedulerMode::BCoverageShadowControllerFirstCut ||
           abSchedulerMode ==
               ABSchedulerMode::BCoverageShadowControllerV2 ||
           abSchedulerMode ==
               ABSchedulerMode::BCoverage2DGatherFirstCut ||
           abSchedulerMode ==
               ABSchedulerMode::BCoverage2DGatherV2 ||
           abSchedulerMode ==
               ABSchedulerMode::BCoverage2DGatherNoStarvationV3 ||
           abSchedulerMode ==
               ABSchedulerMode::BCoverage2DGatherMinGuaranteeFirstCut ||
           abSchedulerMode ==
               ABSchedulerMode::BCoverage2DGatherPingPongFirstCut ||
           abSchedulerMode == ABSchedulerMode::ABSmartPatternPrefetchFirstCut ||
           abSchedulerMode == ABSchedulerMode::VIPABRescueBufferSingleRun ||
           abSchedulerMode == ABSchedulerMode::DualRXABHierarchicalVIP;
}

bool
MatrixFlowEngine::vipPoolMode() const
{
    return abSchedulerMode == ABSchedulerMode::HierarchicalClaimVIPPool ||
           abSchedulerMode == ABSchedulerMode::VIPOracleGuidedSingleRun ||
           abSchedulerMode == ABSchedulerMode::VIPOracleGuidedMainlineOnly ||
           abSchedulerMode == ABSchedulerMode::VIPRescueBufferSingleRun ||
           abSchedulerMode == ABSchedulerMode::BVIPRescueAntiDeadBlockAdmission ||
           abSchedulerMode == ABSchedulerMode::BVIPRescueRecurrenceAwareV2 ||
           abSchedulerMode == ABSchedulerMode::BVIPRescueRecurrenceAwareV3 ||
           abSchedulerMode == ABSchedulerMode::BMHotMainlineDefault ||
           abSchedulerMode == ABSchedulerMode::BMHotMainlineEnhanced ||
           abSchedulerMode == ABSchedulerMode::BMHotRuntimeFirstCut ||
           abSchedulerMode == ABSchedulerMode::BMHotGapAwareNextCut ||
           abSchedulerMode ==
               ABSchedulerMode::BMHotCoverageBlindspotCandidateFirstCut ||
           abSchedulerMode ==
               ABSchedulerMode::BMHotCoverageBlindspotCandidateV2 ||
           abSchedulerMode ==
               ABSchedulerMode::BCoverageShadowControllerFirstCut ||
           abSchedulerMode ==
               ABSchedulerMode::BCoverageShadowControllerV2 ||
           abSchedulerMode ==
               ABSchedulerMode::BCoverage2DGatherFirstCut ||
           abSchedulerMode ==
               ABSchedulerMode::BCoverage2DGatherV2 ||
           abSchedulerMode ==
               ABSchedulerMode::BCoverage2DGatherNoStarvationV3 ||
           abSchedulerMode ==
               ABSchedulerMode::BCoverage2DGatherMinGuaranteeFirstCut ||
           abSchedulerMode ==
               ABSchedulerMode::BCoverage2DGatherPingPongFirstCut ||
           abSchedulerMode == ABSchedulerMode::ABSmartPatternPrefetchFirstCut ||
           abSchedulerMode == ABSchedulerMode::VIPABRescueBufferSingleRun;
}

bool
MatrixFlowEngine::vipPoolEnabled() const
{
    return vipPoolMode() && vipBRowsCapacityConfig > 0;
}

bool
MatrixFlowEngine::mhotPoolMode() const
{
    return abSchedulerMode == ABSchedulerMode::BMHotMainlineDefault ||
           abSchedulerMode == ABSchedulerMode::BMHotMainlineEnhanced ||
           abSchedulerMode == ABSchedulerMode::BMHotRuntimeFirstCut ||
           abSchedulerMode == ABSchedulerMode::BMHotGapAwareNextCut ||
           abSchedulerMode ==
               ABSchedulerMode::BMHotCoverageBlindspotCandidateFirstCut ||
           abSchedulerMode == ABSchedulerMode::BMHotCoverageBlindspotCandidateV2 ||
           abSchedulerMode ==
               ABSchedulerMode::BCoverageShadowControllerFirstCut ||
           abSchedulerMode ==
               ABSchedulerMode::BCoverageShadowControllerV2 ||
           abSchedulerMode ==
               ABSchedulerMode::BCoverage2DGatherFirstCut ||
           abSchedulerMode ==
               ABSchedulerMode::BCoverage2DGatherV2 ||
           abSchedulerMode ==
               ABSchedulerMode::BCoverage2DGatherNoStarvationV3 ||
           abSchedulerMode ==
               ABSchedulerMode::BCoverage2DGatherMinGuaranteeFirstCut ||
           abSchedulerMode ==
               ABSchedulerMode::BCoverage2DGatherPingPongFirstCut ||
           abSchedulerMode == ABSchedulerMode::ABSmartPatternPrefetchFirstCut;
}

bool
MatrixFlowEngine::mhotPoolEnabled() const
{
    return mhotPoolMode() && mhotBRowsCapacityConfig > 0;
}

bool
MatrixFlowEngine::mhotDefaultMode() const
{
    return abSchedulerMode == ABSchedulerMode::BMHotMainlineDefault ||
           abSchedulerMode == ABSchedulerMode::BMHotRuntimeFirstCut;
}

bool
MatrixFlowEngine::mhotEnhancedMode() const
{
    return abSchedulerMode == ABSchedulerMode::BMHotMainlineEnhanced;
}

bool
MatrixFlowEngine::mhotRuntimeFirstCutMode() const
{
    return abSchedulerMode == ABSchedulerMode::BMHotRuntimeFirstCut;
}

bool
MatrixFlowEngine::mhotGapAwareNextCutMode() const
{
    return abSchedulerMode == ABSchedulerMode::BMHotGapAwareNextCut;
}

bool
MatrixFlowEngine::mhotCoverageBlindspotCandidateMode() const
{
    return abSchedulerMode ==
               ABSchedulerMode::BMHotCoverageBlindspotCandidateV2 ||
           abSchedulerMode ==
           ABSchedulerMode::BMHotCoverageBlindspotCandidateFirstCut;
}

bool
MatrixFlowEngine::mhotCoverageBlindspotCandidateV2Mode() const
{
    return abSchedulerMode ==
           ABSchedulerMode::BMHotCoverageBlindspotCandidateV2;
}

bool
MatrixFlowEngine::coverageShadowMode() const
{
    return abSchedulerMode ==
               ABSchedulerMode::BCoverageShadowControllerFirstCut ||
           abSchedulerMode ==
               ABSchedulerMode::BCoverageShadowControllerV2 ||
           abSchedulerMode ==
               ABSchedulerMode::BCoverage2DGatherFirstCut ||
           abSchedulerMode ==
               ABSchedulerMode::BCoverage2DGatherV2 ||
           abSchedulerMode ==
               ABSchedulerMode::BCoverage2DGatherNoStarvationV3 ||
           abSchedulerMode ==
               ABSchedulerMode::BCoverage2DGatherMinGuaranteeFirstCut ||
           abSchedulerMode ==
               ABSchedulerMode::BCoverage2DGatherPingPongFirstCut;
}

bool
MatrixFlowEngine::coverageShadowV2Mode() const
{
    return abSchedulerMode ==
           ABSchedulerMode::BCoverageShadowControllerV2;
}

bool
MatrixFlowEngine::coverage2DGatherMode() const
{
    return abSchedulerMode ==
               ABSchedulerMode::BCoverage2DGatherFirstCut ||
           abSchedulerMode ==
               ABSchedulerMode::BCoverage2DGatherV2 ||
           abSchedulerMode ==
               ABSchedulerMode::BCoverage2DGatherNoStarvationV3 ||
           abSchedulerMode ==
               ABSchedulerMode::BCoverage2DGatherMinGuaranteeFirstCut ||
           abSchedulerMode ==
               ABSchedulerMode::BCoverage2DGatherPingPongFirstCut;
}

bool
MatrixFlowEngine::coverage2DGatherV2Mode() const
{
    return abSchedulerMode ==
           ABSchedulerMode::BCoverage2DGatherV2;
}

bool
MatrixFlowEngine::coverage2DGatherNoStarvationV3Mode() const
{
    return abSchedulerMode ==
           ABSchedulerMode::BCoverage2DGatherNoStarvationV3;
}

bool
MatrixFlowEngine::coverage2DGatherMinGuaranteeFirstCutMode() const
{
    return abSchedulerMode ==
           ABSchedulerMode::BCoverage2DGatherMinGuaranteeFirstCut;
}

bool
MatrixFlowEngine::coverage2DGatherPingPongFirstCutMode() const
{
    return abSchedulerMode ==
           ABSchedulerMode::BCoverage2DGatherPingPongFirstCut;
}

bool
MatrixFlowEngine::coverageShadowPoolEnabled() const
{
    return coverageShadowMode() &&
           coverageShadowRowsCapacityConfig > 0;
}

bool
MatrixFlowEngine::coverageGatherMainlineGraceActive() const
{
    if (!(coverage2DGatherNoStarvationV3Mode() ||
          coverage2DGatherMinGuaranteeFirstCutMode() ||
          coverage2DGatherPingPongFirstCutMode()) ||
        !nextOutputPrefetchValid ||
        nextOutputFirstIssueTick == 0) {
        return false;
    }

    const Tick grace_ticks =
        static_cast<Tick>(kCoverageGatherMainlineGraceCycles) *
        clockPeriod();
    return curTick() < nextOutputFirstIssueTick + grace_ticks;
}

bool
MatrixFlowEngine::coverageShadowUsesPrefetchFeed() const
{
    return coverageShadowMode() && !coverage2DGatherMode();
}

bool
MatrixFlowEngine::smartPatternPrefetchMode() const
{
    return abSchedulerMode == ABSchedulerMode::ABSmartPatternPrefetchFirstCut;
}

void
MatrixFlowEngine::clearCoverageGatherForCurrentTile()
{
    const GemmContext &gctx =
        coverage2DGatherPingPongFirstCutMode() ? coverageGatherCtx : ctx;
    if ((coverage2DGatherNoStarvationV3Mode() ||
         coverage2DGatherMinGuaranteeFirstCutMode() ||
         coverage2DGatherPingPongFirstCutMode()) &&
        coverageGatherValid && gctx.curTileK > 0) {
        uint32_t dropped = 0;
        for (uint32_t row = 0; row < gctx.curTileK; ++row) {
            if (coverageGatherBRowIssued[row]) {
                continue;
            }
            const bool target =
                coverage2DGatherPingPongFirstCutMode()
                    ? isSmartCoverageTargetForContext(gctx, row)
                    : isCoverageBlindspotTargetPattern(
                          row, BProtectionClass::None,
                          gctx.baseB +
                              ((static_cast<Addr>(gctx.k + row) * gctx.size +
                                gctx.j) * gctx.elemBytes));
            if (target) {
                ++dropped;
            }
        }
        stats.gatherDroppedDueToDeadlineCount += dropped;
    }

    coverageGatherValid = false;
    coverageGatherReqsIssuedB = 0;
    coverageGatherRowsBCompleted = 0;
    coverageGatherTargetB = 0;
    coverageGatherBudgetUsedThisTile = 0;
    coverageGatherConsecutiveIssueStreak = 0;
    coverageGatherMinBudgetGrantedThisTile = 0;
    activeCoverageGatherGeneration = ++coverageGatherGeneration;
    coverageGatherCtx = GemmContext();
    std::fill(coverageGatherBRowIssued.begin(),
              coverageGatherBRowIssued.end(), false);
    std::fill(coverageGatherAgeScore.begin(),
              coverageGatherAgeScore.end(), 0);
}

void
MatrixFlowEngine::ageUpCoverageGatherCandidates()
{
    if (!(coverage2DGatherNoStarvationV3Mode() ||
          coverage2DGatherMinGuaranteeFirstCutMode() ||
          coverage2DGatherPingPongFirstCutMode()) ||
        !coverageGatherValid) {
        return;
    }

    const GemmContext &gctx =
        coverage2DGatherPingPongFirstCutMode() ? coverageGatherCtx : ctx;
    uint32_t aged = 0;
    for (uint32_t row = 0; row < gctx.curTileK; ++row) {
        if (coverageGatherBRowIssued[row]) {
            continue;
        }
        const bool target =
            coverage2DGatherPingPongFirstCutMode()
                ? isSmartCoverageTargetForContext(gctx, row)
                : isCoverageBlindspotTargetPattern(
                      row, BProtectionClass::None,
                      gctx.baseB +
                          ((static_cast<Addr>(gctx.k + row) * gctx.size +
                            gctx.j) * gctx.elemBytes));
        if (!target) {
            continue;
        }
        if (coverageGatherAgeScore[row] < std::numeric_limits<uint8_t>::max()) {
            ++coverageGatherAgeScore[row];
            ++aged;
        }
    }

    stats.gatherCandidateAgedUpCount += aged;
}

uint32_t
MatrixFlowEngine::futureITileReuseCount(const GemmContext &gctx) const
{
    if (gctx.i + gctx.curTileM >= gctx.size || gctx.tileM == 0) {
        return 0;
    }

    const uint32_t remaining_rows =
        gctx.size - std::min(gctx.size, gctx.i + gctx.curTileM);
    return (remaining_rows + gctx.tileM - 1) / gctx.tileM;
}

uint32_t
MatrixFlowEngine::futureJTileReuseCount(const GemmContext &gctx) const
{
    if (gctx.j + gctx.curTileN >= gctx.size || gctx.tileN == 0) {
        return 0;
    }

    const uint32_t remaining_cols =
        gctx.size - std::min(gctx.size, gctx.j + gctx.curTileN);
    return (remaining_cols + gctx.tileN - 1) / gctx.tileN;
}

uint32_t
MatrixFlowEngine::tileStepsToNextUse(const GemmContext &gctx) const
{
    const uint32_t future_reuse = futureITileReuseCount(gctx);
    if (future_reuse == 0 || gctx.tileN == 0 || gctx.tileK == 0) {
        return std::numeric_limits<uint32_t>::max();
    }

    const uint32_t total_j_tiles =
        (gctx.size + gctx.tileN - 1) / gctx.tileN;
    const uint32_t total_k_tiles =
        (gctx.size + gctx.tileK - 1) / gctx.tileK;
    if (total_j_tiles == 0 || total_k_tiles == 0) {
        return std::numeric_limits<uint32_t>::max();
    }
    return total_j_tiles * total_k_tiles - 1;
}

bool
MatrixFlowEngine::isSmartCoverageTargetForContext(const GemmContext &gctx,
                                                  uint32_t rowIdx) const
{
    if (rowIdx >= gctx.curTileK) {
        return false;
    }

    const auto dist = oracleDistanceBucket(rowIdx, gctx.curTileK);
    if (dist != OracleDistanceBucket::Far) {
        return false;
    }

    const uint32_t futureReuse = futureITileReuseCount(gctx);
    const uint32_t tileSteps = tileStepsToNextUse(gctx);
    return coverageBlindspotFutureReuseBucket(futureReuse) == 2 &&
           coverageBlindspotTileStepBand(tileSteps) == 1;
}

bool
MatrixFlowEngine::isSmartGapTargetForContext(const GemmContext &gctx,
                                             uint32_t rowIdx) const
{
    return rowIdx < gctx.curTileK &&
           oracleDistanceBucket(rowIdx, gctx.curTileK) ==
               OracleDistanceBucket::Near;
}

bool
MatrixFlowEngine::isSmartACandidateForContext(const GemmContext &gctx,
                                              uint32_t rowIdx) const
{
    if (rowIdx >= gctx.curTileM) {
        return false;
    }
    const Addr rowAddr = gctx.baseA +
        ((static_cast<Addr>(gctx.i + rowIdx) * gctx.size + gctx.k) *
         gctx.elemBytes);
    const bool repeat =
        oracleARecurrenceBucket(rowAddr) == OracleARecurrenceBucket::Repeat;
    const uint32_t futureReuse = futureJTileReuseCount(gctx);
    const bool multi = futureReuse > 1;
    const OracleDistanceBucket dist =
        futureReuse == 0 ? OracleDistanceBucket::Far
                         : (futureReuse == 1 ? OracleDistanceBucket::Immediate
                                             : OracleDistanceBucket::Near);
    return repeat || (multi && dist != OracleDistanceBucket::Far);
}

int
MatrixFlowEngine::selectSmartCurrentARow() const
{
    int bestRow = -1;
    int bestScore = -1;
    const OracleDistanceBucket dist = oracleADistanceBucket();
    const bool multi = oracleAReuseBucket() == OracleAReuseBucket::Multi;
    for (uint32_t row = 0; row < targetReqsA; ++row) {
        if (currentARowIssued[row]) {
            continue;
        }
        const Addr rowAddr = ctx.baseA +
            ((static_cast<Addr>(ctx.i + row) * ctx.size + ctx.k) *
             ctx.elemBytes);
        int score = 0;
        if (oracleARecurrenceBucket(rowAddr) == OracleARecurrenceBucket::Repeat) {
            score += 8;
        }
        if (multi) {
            score += 3;
        }
        if (dist == OracleDistanceBucket::Immediate) {
            score += 4;
        } else if (dist == OracleDistanceBucket::Near) {
            score += 2;
        }
        if (score > bestScore) {
            bestScore = score;
            bestRow = static_cast<int>(row);
        }
    }
    return bestRow;
}

int
MatrixFlowEngine::selectSmartNextPrefetchARow() const
{
    int bestRow = -1;
    int bestScore = -1;
    const uint32_t futureReuse = futureJTileReuseCount(nextCtx);
    const bool multi = futureReuse > 1;
    const OracleDistanceBucket dist =
        futureReuse == 0 ? OracleDistanceBucket::Far
                         : (futureReuse == 1 ? OracleDistanceBucket::Immediate
                                             : OracleDistanceBucket::Near);
    for (uint32_t row = 0; row < nextCtx.curTileM; ++row) {
        if (nextPrefetchARowIssued[row]) {
            continue;
        }
        const Addr rowAddr = nextCtx.baseA +
            ((static_cast<Addr>(nextCtx.i + row) * nextCtx.size + nextCtx.k) *
             nextCtx.elemBytes);
        int score = 0;
        if (oracleARecurrenceBucket(rowAddr) == OracleARecurrenceBucket::Repeat) {
            score += 8;
        }
        if (multi) {
            score += 3;
        }
        if (dist == OracleDistanceBucket::Immediate) {
            score += 4;
        } else if (dist == OracleDistanceBucket::Near) {
            score += 2;
        }
        if (score > bestScore) {
            bestScore = score;
            bestRow = static_cast<int>(row);
        }
    }
    return bestRow;
}

int
MatrixFlowEngine::selectSmartNextPrefetchBRow() const
{
    int bestRow = -1;
    int bestScore = -1;
    for (uint32_t row = 0; row < nextCtx.curTileK; ++row) {
        if (nextPrefetchBRowIssued[row]) {
            continue;
        }
        const auto dist = oracleDistanceBucket(row, nextCtx.curTileK);
        int score = 0;
        if (dist == OracleDistanceBucket::Immediate) {
            score = 300;
        } else if (dist == OracleDistanceBucket::Near) {
            score = 250;
        } else if (isSmartCoverageTargetForContext(nextCtx, row)) {
            score = 220;
        }
        if (score > bestScore) {
            bestScore = score;
            bestRow = static_cast<int>(row);
        }
    }
    return bestRow;
}

int
MatrixFlowEngine::selectSmartNextOutputPrefetchBRow() const
{
    int bestRow = -1;
    int bestScore = -1;
    for (uint32_t row = 0; row < nextOutputCtx.curTileK; ++row) {
        if (nextOutputPrefetchBRowIssued[row]) {
            continue;
        }
        const auto dist = oracleDistanceBucket(row, nextOutputCtx.curTileK);
        int score = 0;
        if (dist == OracleDistanceBucket::Immediate) {
            score = 300;
        } else if (dist == OracleDistanceBucket::Near) {
            score = 250;
        } else if (isSmartCoverageTargetForContext(nextOutputCtx, row)) {
            score = 220;
        }
        if (score > bestScore) {
            bestScore = score;
            bestRow = static_cast<int>(row);
        }
    }
    return bestRow;
}

bool
MatrixFlowEngine::mhotAllowsMainlineBypass() const
{
    return abSchedulerMode == ABSchedulerMode::BMHotMainlineDefault ||
           abSchedulerMode == ABSchedulerMode::BMHotMainlineEnhanced;
}

uint32_t
MatrixFlowEngine::mhotCoverageBlindspotQuotaRows() const
{
    if (!mhotCoverageBlindspotCandidateMode() || mhotBRowsCapacityConfig == 0) {
        return 0;
    }
    if (mhotCoverageBlindspotCandidateV2Mode()) {
        return std::min<uint32_t>(
            mhotBRowsCapacityConfig,
            std::max<uint32_t>(8, mhotBRowsCapacityConfig / 8));
    }
    return std::min<uint32_t>(
        mhotBRowsCapacityConfig,
        std::max<uint32_t>(16, mhotBRowsCapacityConfig / 4));
}

bool
MatrixFlowEngine::isCoverageBlindspotNormalCandidate(uint32_t rowIdx,
                                                     BProtectionClass cls,
                                                     Addr rowAddr) const
{
    OracleSourceClass src = OracleSourceClass::Normal;
    if (rowIdx < currentBOracleSource.size()) {
        src = currentBOracleSource[rowIdx];
    }
    if (src == OracleSourceClass::Normal) {
        src = oracleSourceFromProtectionClass(cls);
    }
    if (src != OracleSourceClass::Normal) {
        return false;
    }

    if (classifyBFallbackAutopsy(rowIdx, cls, rowAddr) !=
        FallbackAutopsyBucket::Coverage) {
        return false;
    }

    return oracleDistanceBucket(rowIdx, targetReqsB) ==
           OracleDistanceBucket::Far;
}

bool
MatrixFlowEngine::isCoverageBlindspotTargetPattern(uint32_t rowIdx,
                                                   BProtectionClass cls,
                                                   Addr rowAddr) const
{
    if (!isCoverageBlindspotNormalCandidate(rowIdx, cls, rowAddr)) {
        return false;
    }
    const uint32_t futureReuse = rescueFutureITileReuseCount();
    const uint32_t tileSteps = rescueTileStepsToNextUse();
    return coverageBlindspotFutureReuseBucket(futureReuse) == 2 &&
           coverageBlindspotTileStepBand(tileSteps) == 1;
}

uint8_t
MatrixFlowEngine::coverageBlindspotFutureReuseBucket(uint32_t futureReuse) const
{
    if (futureReuse == 0) {
        return 0;
    }
    if (futureReuse == 1) {
        return 1;
    }
    if (futureReuse <= 3) {
        return 2;
    }
    return 3;
}

uint8_t
MatrixFlowEngine::coverageBlindspotTileStepBand(uint32_t tileSteps) const
{
    if (tileSteps <= 4) {
        return 0;
    }
    if (tileSteps <= 12) {
        return 1;
    }
    if (tileSteps <= 24) {
        return 2;
    }
    return 3;
}

uint64_t
MatrixFlowEngine::coverageBlindspotPatternKey(uint32_t rowIdx,
                                              BProtectionClass cls,
                                              Addr rowAddr) const
{
    const auto src = OracleSourceClass::Normal;
    const auto autopsy = classifyBFallbackAutopsy(rowIdx, cls, rowAddr);
    const auto dist = oracleDistanceBucket(rowIdx, targetReqsB);
    const uint32_t futureReuse = rescueFutureITileReuseCount();
    const uint32_t tileSteps = rescueTileStepsToNextUse();
    const uint64_t srcBits = static_cast<uint64_t>(src) & 0xFFu;
    const uint64_t autopsyBits = static_cast<uint64_t>(autopsy) & 0xFFu;
    const uint64_t distBits = static_cast<uint64_t>(dist) & 0xFFu;
    const uint64_t reuseBits =
        static_cast<uint64_t>(coverageBlindspotFutureReuseBucket(futureReuse)) &
        0xFFu;
    const uint64_t tileBits =
        static_cast<uint64_t>(coverageBlindspotTileStepBand(tileSteps)) & 0xFFu;
    return (srcBits << 32) | (autopsyBits << 24) | (distBits << 16) |
           (reuseBits << 8) | tileBits;
}

bool
MatrixFlowEngine::isCoverageBlindspotPromoted(uint32_t rowIdx,
                                              BProtectionClass cls,
                                              Addr rowAddr) const
{
    if (!isCoverageBlindspotNormalCandidate(rowIdx, cls, rowAddr)) {
        return false;
    }

    const uint64_t key = coverageBlindspotPatternKey(rowIdx, cls, rowAddr);
    auto it = bCoverageBlindspotPattern.find(key);
    return it != bCoverageBlindspotPattern.end() && it->second.promoted;
}

bool
MatrixFlowEngine::oracleGuidedVipMode() const
{
    return abSchedulerMode == ABSchedulerMode::VIPOracleGuidedSingleRun ||
           abSchedulerMode == ABSchedulerMode::VIPOracleGuidedMainlineOnly;
}

bool
MatrixFlowEngine::vipForcedDiversionMode() const
{
    return abSchedulerMode == ABSchedulerMode::VIPOracleGuidedSingleRun;
}

bool
MatrixFlowEngine::pureMainlineVipMode() const
{
    return abSchedulerMode == ABSchedulerMode::VIPOracleGuidedMainlineOnly;
}

bool
MatrixFlowEngine::vipRescueMode() const
{
    return abSchedulerMode == ABSchedulerMode::VIPRescueBufferSingleRun ||
           abSchedulerMode == ABSchedulerMode::BVIPRescueAntiDeadBlockAdmission ||
           abSchedulerMode == ABSchedulerMode::BVIPRescueRecurrenceAwareV2 ||
           abSchedulerMode == ABSchedulerMode::BVIPRescueRecurrenceAwareV3 ||
           abSchedulerMode == ABSchedulerMode::BMHotMainlineDefault ||
           abSchedulerMode == ABSchedulerMode::BMHotMainlineEnhanced ||
           abSchedulerMode == ABSchedulerMode::BMHotRuntimeFirstCut ||
           abSchedulerMode == ABSchedulerMode::BMHotGapAwareNextCut ||
           abSchedulerMode ==
               ABSchedulerMode::BMHotCoverageBlindspotCandidateFirstCut ||
           abSchedulerMode ==
               ABSchedulerMode::BMHotCoverageBlindspotCandidateV2 ||
           abSchedulerMode ==
               ABSchedulerMode::BCoverageShadowControllerFirstCut ||
           abSchedulerMode ==
               ABSchedulerMode::BCoverageShadowControllerV2 ||
           abSchedulerMode ==
               ABSchedulerMode::BCoverage2DGatherFirstCut ||
           abSchedulerMode ==
               ABSchedulerMode::BCoverage2DGatherNoStarvationV3 ||
           abSchedulerMode ==
               ABSchedulerMode::BCoverage2DGatherMinGuaranteeFirstCut ||
           abSchedulerMode ==
               ABSchedulerMode::BCoverage2DGatherPingPongFirstCut ||
           abSchedulerMode == ABSchedulerMode::ABSmartPatternPrefetchFirstCut ||
           abSchedulerMode == ABSchedulerMode::VIPABRescueBufferSingleRun;
}

bool
MatrixFlowEngine::vipRescueAntiDeadBlockMode() const
{
    return abSchedulerMode ==
           ABSchedulerMode::BVIPRescueAntiDeadBlockAdmission;
}

bool
MatrixFlowEngine::vipRescueRecurrenceAwareV2Mode() const
{
    return abSchedulerMode ==
           ABSchedulerMode::BVIPRescueRecurrenceAwareV2;
}

bool
MatrixFlowEngine::vipRescueRecurrenceAwareV3Mode() const
{
    return abSchedulerMode ==
               ABSchedulerMode::BVIPRescueRecurrenceAwareV3 ||
           abSchedulerMode == ABSchedulerMode::BMHotMainlineDefault ||
           abSchedulerMode == ABSchedulerMode::BMHotMainlineEnhanced ||
           abSchedulerMode == ABSchedulerMode::BMHotRuntimeFirstCut ||
           abSchedulerMode == ABSchedulerMode::BMHotGapAwareNextCut ||
           abSchedulerMode ==
               ABSchedulerMode::BMHotCoverageBlindspotCandidateFirstCut ||
           abSchedulerMode ==
               ABSchedulerMode::BMHotCoverageBlindspotCandidateV2 ||
           abSchedulerMode ==
               ABSchedulerMode::BCoverageShadowControllerFirstCut ||
           abSchedulerMode ==
               ABSchedulerMode::BCoverageShadowControllerV2 ||
           abSchedulerMode ==
               ABSchedulerMode::BCoverage2DGatherFirstCut ||
           abSchedulerMode ==
               ABSchedulerMode::BCoverage2DGatherNoStarvationV3 ||
           abSchedulerMode ==
               ABSchedulerMode::BCoverage2DGatherMinGuaranteeFirstCut ||
           abSchedulerMode ==
               ABSchedulerMode::BCoverage2DGatherPingPongFirstCut ||
           abSchedulerMode == ABSchedulerMode::ABSmartPatternPrefetchFirstCut;
}

bool
MatrixFlowEngine::vipABRescueMode() const
{
    return abSchedulerMode == ABSchedulerMode::VIPABRescueBufferSingleRun;
}

bool
MatrixFlowEngine::dualRxSchedulerMode() const
{
    return abSchedulerMode == ABSchedulerMode::DualRXABHierarchicalVIP;
}

void
MatrixFlowEngine::clearVipPool()
{
    if (vipBSlots.empty()) {
        return;
    }

    for (auto &slot : vipBSlots) {
        slot = VipRowSlot();
    }
    vipPoolOccupancy = 0;
}

void
MatrixFlowEngine::clearMHotPool()
{
    if (mhotBSlots.empty()) {
        return;
    }

    for (auto &slot : mhotBSlots) {
        slot = VipRowSlot();
    }
    mhotPoolOccupancy = 0;
}

void
MatrixFlowEngine::clearCoverageShadowPool()
{
    clearCoverageShadowActiveBank();
    clearCoverageShadowFillBank();
    coverageShadowActiveCtx = GemmContext();
    coverageShadowFillCtx = GemmContext();
    coverageShadowActiveValid = false;
    coverageShadowFillValid = false;
}

void
MatrixFlowEngine::clearCoverageShadowActiveBank()
{
    if (coverageShadowBSlots.empty()) {
        return;
    }

    for (auto &slot : coverageShadowBSlots) {
        slot = VipRowSlot();
    }
    coverageShadowPoolOccupancy = 0;
    clearCoverageShadowLowPrioritySlots();
    coverageShadowActiveValid = false;
}

void
MatrixFlowEngine::clearCoverageShadowFillBank()
{
    if (coverageShadowFillBSlots.empty()) {
        return;
    }

    for (auto &slot : coverageShadowFillBSlots) {
        slot = VipRowSlot();
    }
    coverageShadowFillPoolOccupancy = 0;
    coverageShadowFillValid = false;
}

void
MatrixFlowEngine::clearCoverageShadowLowPrioritySlots()
{
    if (coverageShadowLowPrioritySlot.empty()) {
        return;
    }
    std::fill(coverageShadowLowPrioritySlot.begin(),
              coverageShadowLowPrioritySlot.end(), false);
    coverageShadowLowPriorityOccupancy = 0;
    coverageShadowLowPriorityOccupancyPeakObserved = 0;
    stats.coverageShadowLowPriorityOccupancyPeak = 0;
}

bool
MatrixFlowEngine::coverageShadowContextMatches(const GemmContext &a,
                                               const GemmContext &b) const
{
    return a.i == b.i && a.j == b.j && a.k == b.k &&
           a.curTileM == b.curTileM && a.curTileN == b.curTileN &&
           a.curTileK == b.curTileK;
}

bool
MatrixFlowEngine::buildCoverageGatherFillContext(GemmContext &gctx) const
{
    if (hasNextKTile()) {
        gctx = buildNextKContext();
        return true;
    }
    if (hasNextOutputTile()) {
        gctx = buildNextOutputContext();
        return true;
    }
    return false;
}

void
MatrixFlowEngine::maybeActivateCoverageShadowBankForCurrentTile()
{
    if (!coverage2DGatherPingPongFirstCutMode() ||
        !coverageShadowPoolEnabled()) {
        return;
    }

    if (coverageShadowFillValid &&
        coverageShadowContextMatches(coverageShadowFillCtx, ctx)) {
        std::swap(coverageShadowBBuffer, coverageShadowFillBBuffer);
        std::swap(coverageShadowBSlots, coverageShadowFillBSlots);
        std::swap(coverageShadowPoolOccupancy, coverageShadowFillPoolOccupancy);
        coverageShadowActiveCtx = coverageShadowFillCtx;
        coverageShadowActiveValid = true;
        clearCoverageShadowFillBank();
        coverageShadowFillCtx = GemmContext();
        return;
    }

    if (!coverageShadowActiveValid ||
        !coverageShadowContextMatches(coverageShadowActiveCtx, ctx)) {
        clearCoverageShadowActiveBank();
        coverageShadowActiveCtx = GemmContext();
    }
}

void
MatrixFlowEngine::clearStagedCurrentVipBacked()
{
    std::fill(stagedCurrentBVipBacked.begin(),
              stagedCurrentBVipBacked.end(), false);
    std::fill(stagedCurrentBVipSource.begin(),
              stagedCurrentBVipSource.end(), VipSourceClass::None);
    std::fill(stagedCurrentBVipProtectionClass.begin(),
              stagedCurrentBVipProtectionClass.end(),
              BProtectionClass::None);
    std::fill(stagedCurrentBVipAdmitClass.begin(),
              stagedCurrentBVipAdmitClass.end(),
              VipAdmitClass::Reject);
    std::fill(stagedCurrentBVipPrefetchedCounted.begin(),
              stagedCurrentBVipPrefetchedCounted.end(), false);
}

void
MatrixFlowEngine::clearStagedCurrentOracleVipSelected()
{
    std::fill(stagedCurrentBOracleVipSelected.begin(),
              stagedCurrentBOracleVipSelected.end(), false);
}

void
MatrixFlowEngine::clearCurrentVipBacked(bool invalidate)
{
    for (uint32_t row = 0; row < currentBVipBacked.size(); ++row) {
        if (invalidate && currentBVipBacked[row] &&
            currentBFutureClaimed[row]) {
            clearCurrentFutureClaimRow(row, false, false, true);
        }
        currentBVipBacked[row] = false;
        currentBServedFromVip[row] = false;
        currentBVipSource[row] = VipSourceClass::None;
        currentBVipAdmitClass[row] = VipAdmitClass::Reject;
        currentBVipPrefetchedCounted[row] = false;
    }
}

void
MatrixFlowEngine::clearCurrentOracleVipSelected()
{
    std::fill(currentBOracleVipSelected.begin(),
              currentBOracleVipSelected.end(), false);
}

void
MatrixFlowEngine::clearCurrentOracleTracking()
{
    std::fill(currentBOracleSource.begin(), currentBOracleSource.end(),
              OracleSourceClass::Normal);
    std::fill(currentBOracleOutcomeRecorded.begin(),
              currentBOracleOutcomeRecorded.end(), false);
}

void
MatrixFlowEngine::clearFallbackAutopsyTracking()
{
    bFallbackAutopsy.clear();
}

MatrixFlowEngine::OracleSourceClass
MatrixFlowEngine::oracleSourceFromVipSource(VipSourceClass src) const
{
    switch (src) {
      case VipSourceClass::NextOutput:
        return OracleSourceClass::NextOutput;
      case VipSourceClass::Claim:
        return OracleSourceClass::Claim;
      case VipSourceClass::CarryOver:
        return OracleSourceClass::CarryOver;
      case VipSourceClass::CurrentWindow:
        return OracleSourceClass::CurrentWindow;
      case VipSourceClass::Normal:
        return OracleSourceClass::Normal;
      case VipSourceClass::Rescue:
      case VipSourceClass::RescueA:
        return OracleSourceClass::Normal;
      case VipSourceClass::None:
      default:
        return OracleSourceClass::Normal;
    }
}

MatrixFlowEngine::OracleSourceClass
MatrixFlowEngine::oracleSourceFromProtectionClass(BProtectionClass cls) const
{
    switch (cls) {
      case BProtectionClass::NextOutput:
      case BProtectionClass::FutureHoleFilling:
        return OracleSourceClass::NextOutput;
      case BProtectionClass::CarryOver:
        return OracleSourceClass::CarryOver;
      case BProtectionClass::ComputeWindow:
      case BProtectionClass::HoleFilling:
        return OracleSourceClass::CurrentWindow;
      case BProtectionClass::None:
      default:
        return OracleSourceClass::Normal;
    }
}

MatrixFlowEngine::OracleDistanceBucket
MatrixFlowEngine::oracleDistanceBucket(uint32_t rowIdx, uint32_t totalRows) const
{
    const uint32_t immediate_rows =
        std::min(totalRows, computeLaunchWindowB());
    const uint32_t near_rows = std::min<uint32_t>(
        totalRows, immediate_rows + holeFillLeadRowsConfig);
    if (rowIdx < immediate_rows) {
        return OracleDistanceBucket::Immediate;
    }
    if (rowIdx < near_rows) {
        return OracleDistanceBucket::Near;
    }
    return OracleDistanceBucket::Far;
}

MatrixFlowEngine::OracleDistanceBucket
MatrixFlowEngine::oracleADistanceBucket() const
{
    const uint32_t future_reuse = rescueFutureJTileReuseCount();
    if (future_reuse == 0) {
        return OracleDistanceBucket::Far;
    }
    if (future_reuse == 1) {
        return OracleDistanceBucket::Immediate;
    }
    return OracleDistanceBucket::Near;
}

MatrixFlowEngine::OracleAReuseBucket
MatrixFlowEngine::oracleAReuseBucket() const
{
    return rescueFutureJTileReuseCount() > 1
        ? OracleAReuseBucket::Multi
        : OracleAReuseBucket::One;
}

MatrixFlowEngine::OracleARecurrenceBucket
MatrixFlowEngine::oracleARecurrenceBucket(Addr rowAddr) const
{
    auto it = aRemoteVictimCount.find(rowAddr);
    if (it != aRemoteVictimCount.end() && it->second > 0) {
        return OracleARecurrenceBucket::Repeat;
    }
    return OracleARecurrenceBucket::First;
}

MatrixFlowEngine::FallbackAutopsyBucket
MatrixFlowEngine::classifyBFallbackAutopsy(uint32_t rowIdx,
                                           BProtectionClass cls,
                                           Addr rowAddr) const
{
    OracleSourceClass src = OracleSourceClass::Normal;
    if (rowIdx < currentBOracleSource.size()) {
        src = currentBOracleSource[rowIdx];
    }
    if (src == OracleSourceClass::Normal) {
        src = oracleSourceFromProtectionClass(cls);
    }

    const auto it = bFallbackAutopsy.find(rowAddr);
    if (it != bFallbackAutopsy.end()) {
        const auto &meta = it->second;
        if (meta.nearbyInserted && meta.nearbyEvicted) {
            return FallbackAutopsyBucket::Churn;
        }
    }

    if (src != OracleSourceClass::Normal) {
        return FallbackAutopsyBucket::Timeliness;
    }

    return FallbackAutopsyBucket::Coverage;
}

void
MatrixFlowEngine::setCurrentOracleSource(uint32_t rowIdx, OracleSourceClass src)
{
    if (rowIdx >= currentBOracleSource.size()) {
        return;
    }
    currentBOracleSource[rowIdx] = src;
}

void
MatrixFlowEngine::recordCurrentBOracleOutcome(uint32_t rowIdx, bool local)
{
    if (rowIdx >= currentBOracleOutcomeRecorded.size() ||
        currentBOracleOutcomeRecorded[rowIdx]) {
        return;
    }

    const auto src = currentBOracleSource[rowIdx];
    const auto dist = oracleDistanceBucket(rowIdx, targetReqsB);
    const double reuseWeight = static_cast<double>(ctx.curTileM);
    if (local) {
        stats.oracleLocalRows[static_cast<int>(src)][static_cast<int>(dist)] += 1;
        stats.oracleLocalReuseWeight[static_cast<int>(src)][
            static_cast<int>(dist)] += reuseWeight;
    } else {
        stats.oracleFallbackRows[static_cast<int>(src)][
            static_cast<int>(dist)] += 1;
        stats.oracleFallbackReuseWeight[static_cast<int>(src)][
            static_cast<int>(dist)] += reuseWeight;
    }
    currentBOracleOutcomeRecorded[rowIdx] = true;
}

void
MatrixFlowEngine::recordCurrentAOracleOutcome(Addr rowAddr, bool local)
{
    const auto dist = oracleADistanceBucket();
    const auto reuse = oracleAReuseBucket();
    const auto recurrence = oracleARecurrenceBucket(rowAddr);

    if (local) {
        stats.oracleALocalByDistance[static_cast<int>(dist)] += 1;
        stats.oracleALocalByReuse[static_cast<int>(reuse)] += 1;
        stats.oracleALocalByRecurrence[static_cast<int>(recurrence)] += 1;
    } else {
        stats.oracleARemoteByDistance[static_cast<int>(dist)] += 1;
        stats.oracleARemoteByReuse[static_cast<int>(reuse)] += 1;
        stats.oracleARemoteByRecurrence[static_cast<int>(recurrence)] += 1;
        aRemoteVictimCount[rowAddr] += 1;
    }
}

void
MatrixFlowEngine::recordBFallbackAutopsy(uint32_t rowIdx,
                                         BProtectionClass cls,
                                         Addr rowAddr)
{
    OracleSourceClass src = OracleSourceClass::Normal;
    if (rowIdx < currentBOracleSource.size()) {
        src = currentBOracleSource[rowIdx];
    }
    if (src == OracleSourceClass::Normal) {
        src = oracleSourceFromProtectionClass(cls);
    }

    const auto bucket = classifyBFallbackAutopsy(rowIdx, cls, rowAddr);
    switch (bucket) {
      case FallbackAutopsyBucket::Timeliness:
        stats.fallbackAutopsyTimelinessCount++;
        break;
      case FallbackAutopsyBucket::Churn:
        stats.fallbackAutopsyChurnCount++;
        break;
      case FallbackAutopsyBucket::Coverage:
      case FallbackAutopsyBucket::NumBuckets:
      default:
        stats.fallbackAutopsyCoverageCount++;
        break;
    }

    stats.fallbackAutopsyRows[static_cast<int>(src)][
        static_cast<int>(bucket)] += 1;

    if (src == OracleSourceClass::Normal &&
        bucket == FallbackAutopsyBucket::Coverage) {
        const auto dist = oracleDistanceBucket(rowIdx, targetReqsB);
        stats.fallbackAutopsyCoverageNormalByDistance[
            static_cast<int>(dist)] += 1;

        const uint64_t patternKey =
            coverageBlindspotPatternKey(rowIdx, cls, rowAddr);
        auto [it, inserted] =
            bCoverageBlindspotMissPatternCount.emplace(patternKey, 0);
        const bool firstSeen = inserted;
        it->second += 1;
        if (firstSeen) {
            stats.coverageBlindspotFirstSeenMissCount++;
            stats.coverageBlindspotDistinctPatternCount++;
        } else {
            stats.coverageBlindspotRepeatMissCount++;
        }

        if (isCoverageBlindspotTargetPattern(rowIdx, cls, rowAddr)) {
            stats.coverageBlindspotTargetMissCount++;
            if (firstSeen) {
                stats.coverageBlindspotTargetFirstSeenMissCount++;
                stats.coverageBlindspotTargetDistinctPatternCount++;
            } else {
                stats.coverageBlindspotTargetRepeatMissCount++;
            }
        }
    }
}

void
MatrixFlowEngine::noteBFallbackAutopsyInsert(Addr rowAddr,
                                             VipSourceClass source)
{
    if (source == VipSourceClass::RescueA || source == VipSourceClass::None) {
        return;
    }
    auto &meta = bFallbackAutopsy[rowAddr];
    meta.nearbyInserted = true;
    meta.lastSource = source;
    meta.lastInsertTick = curTick();
}

void
MatrixFlowEngine::noteBFallbackAutopsyEvict(Addr rowAddr,
                                            VipSourceClass source)
{
    if (source == VipSourceClass::RescueA || source == VipSourceClass::None) {
        return;
    }
    auto &meta = bFallbackAutopsy[rowAddr];
    meta.nearbyEvicted = true;
    meta.lastSource = source;
    meta.lastEvictTick = curTick();
}

MatrixFlowEngine::VipSourceClass
MatrixFlowEngine::vipSourceFromProtectionClass(BProtectionClass cls) const
{
    switch (cls) {
      case BProtectionClass::CarryOver:
        return VipSourceClass::CarryOver;
      case BProtectionClass::NextOutput:
      case BProtectionClass::FutureHoleFilling:
        return VipSourceClass::NextOutput;
      case BProtectionClass::ComputeWindow:
        return VipSourceClass::CurrentWindow;
      case BProtectionClass::HoleFilling:
      case BProtectionClass::None:
      default:
        return VipSourceClass::Claim;
    }
}

uint32_t
MatrixFlowEngine::vipEmergencyReserveRows(uint32_t totalRows) const
{
    return std::min<uint32_t>(
        std::min<uint32_t>(totalRows, computeLaunchWindowB()),
        std::min<uint32_t>(vipBRowsCapacityConfig, 4));
}

uint32_t
MatrixFlowEngine::vipNearUseWindowRows(uint32_t totalRows) const
{
    const uint32_t immediate = computeLaunchWindowB();
    const uint32_t near_rows = immediate + holeFillLeadRowsConfig;
    return std::min(totalRows,
                    std::max(immediate,
                             std::min<uint32_t>(near_rows,
                                                vipBRowsCapacityConfig)));
}

MatrixFlowEngine::VipAdmitClass
MatrixFlowEngine::classifyVipAdmission(uint32_t rowIdx, uint32_t totalRows,
                                       VipSourceClass source,
                                       BProtectionClass protectionClass) const
{
    if (!vipPoolEnabled()) {
        return VipAdmitClass::Reject;
    }

    if (pureMainlineVipMode()) {
        const auto oracleSource = oracleSourceFromVipSource(source);
        const auto dist = oracleDistanceBucket(rowIdx, totalRows);
        if (oracleSource != OracleSourceClass::NextOutput) {
            return VipAdmitClass::Reject;
        }
        if (dist == OracleDistanceBucket::Immediate ||
            dist == OracleDistanceBucket::Near) {
            return VipAdmitClass::Strong;
        }
        return VipAdmitClass::Reject;
    }

    if (oracleGuidedVipMode()) {
        const auto oracleSource = oracleSourceFromVipSource(source);
        const auto dist = oracleDistanceBucket(rowIdx, totalRows);

        if (oracleSource == OracleSourceClass::NextOutput) {
            if (dist == OracleDistanceBucket::Immediate ||
                dist == OracleDistanceBucket::Near) {
                return VipAdmitClass::Strong;
            }
            if (dist == OracleDistanceBucket::Far) {
                return VipAdmitClass::Weak;
            }
            return VipAdmitClass::Reject;
        }

        if (oracleSource == OracleSourceClass::Claim ||
            oracleSource == OracleSourceClass::CarryOver) {
            return VipAdmitClass::Weak;
        }

        if (oracleSource == OracleSourceClass::CurrentWindow &&
            dist == OracleDistanceBucket::Immediate &&
            rowIdx < vipEmergencyReserveRows(totalRows)) {
            return VipAdmitClass::Weak;
        }
        return VipAdmitClass::Reject;
    }

    if (vipRescueMode()) {
        return VipAdmitClass::Reject;
    }

    const uint32_t immediate_rows =
        std::min(totalRows, computeLaunchWindowB());
    const uint32_t near_rows = vipNearUseWindowRows(totalRows);

    if (source == VipSourceClass::CurrentWindow) {
        return rowIdx < immediate_rows ? VipAdmitClass::Weak
                                       : VipAdmitClass::Reject;
    }

    if (source != VipSourceClass::NextOutput &&
        source != VipSourceClass::Claim &&
        source != VipSourceClass::CarryOver) {
        return VipAdmitClass::Reject;
    }

    if (rowIdx < immediate_rows) {
        return VipAdmitClass::Strong;
    }
    if (rowIdx < near_rows) {
        return VipAdmitClass::Weak;
    }
    if (protectionClass == BProtectionClass::FutureHoleFilling &&
        rowIdx < near_rows) {
        return VipAdmitClass::Weak;
    }
    return VipAdmitClass::Reject;
}

void
MatrixFlowEngine::recordVipOracleGuidedAdmit(uint32_t rowIdx,
                                             uint32_t totalRows,
                                             VipSourceClass source)
{
    if (!oracleGuidedVipMode()) {
        return;
    }

    const auto oracleSource = oracleSourceFromVipSource(source);
    const auto dist = oracleDistanceBucket(rowIdx, totalRows);
    if (oracleSource == OracleSourceClass::NextOutput) {
        if (dist == OracleDistanceBucket::Immediate) {
            stats.vipAdmitNextOutputImmediateCount++;
        } else if (dist == OracleDistanceBucket::Near) {
            stats.vipAdmitNextOutputNearCount++;
        } else {
            stats.vipAdmitNextOutputFarCount++;
        }
        return;
    }

    if (oracleSource == OracleSourceClass::Claim) {
        stats.vipAdmitClaimCount++;
        return;
    }

    if (oracleSource == OracleSourceClass::CarryOver) {
        stats.vipAdmitCarryOverCount++;
        return;
    }

    if (oracleSource == OracleSourceClass::CurrentWindow &&
        dist == OracleDistanceBucket::Immediate) {
        stats.vipAdmitCurrentWindowImmediateCount++;
        return;
    }
}

void
MatrixFlowEngine::recordVipOracleGuidedReject(OracleSourceClass source,
                                              OracleDistanceBucket dist)
{
    if (!oracleGuidedVipMode()) {
        return;
    }

    if (source == OracleSourceClass::Normal &&
        dist == OracleDistanceBucket::Near) {
        stats.vipRejectNormalNearCount++;
        return;
    }
    if (source == OracleSourceClass::Normal &&
        dist == OracleDistanceBucket::Far) {
        stats.vipRejectNormalFarCount++;
        return;
    }
    stats.vipRejectOtherCount++;
}

void
MatrixFlowEngine::recordOracleSelectedRowMiss(uint32_t rowIdx)
{
    if (!oracleGuidedVipMode() || rowIdx >= currentBOracleVipSelected.size() ||
        !currentBOracleVipSelected[rowIdx]) {
        return;
    }

    stats.oracleSelectedRowsMissedByVipCount++;
    currentBOracleVipSelected[rowIdx] = false;
}

uint32_t
MatrixFlowEngine::rescueFutureITileReuseCount() const
{
    if (ctx.i + ctx.curTileM >= ctx.size || ctx.tileM == 0) {
        return 0;
    }

    const uint32_t remaining_rows =
        ctx.size - std::min(ctx.size, ctx.i + ctx.curTileM);
    return (remaining_rows + ctx.tileM - 1) / ctx.tileM;
}

uint32_t
MatrixFlowEngine::rescueFutureJTileReuseCount() const
{
    if (ctx.j + ctx.curTileN >= ctx.size || ctx.tileN == 0) {
        return 0;
    }

    const uint32_t remaining_cols =
        ctx.size - std::min(ctx.size, ctx.j + ctx.curTileN);
    return (remaining_cols + ctx.tileN - 1) / ctx.tileN;
}

uint32_t
MatrixFlowEngine::rescueTileStepsToNextUse() const
{
    const uint32_t future_reuse = rescueFutureITileReuseCount();
    if (future_reuse == 0 || ctx.tileN == 0 || ctx.tileK == 0) {
        return std::numeric_limits<uint32_t>::max();
    }

    const uint32_t total_j_tiles =
        (ctx.size + ctx.tileN - 1) / ctx.tileN;
    const uint32_t total_k_tiles =
        (ctx.size + ctx.tileK - 1) / ctx.tileK;
    if (total_j_tiles == 0 || total_k_tiles == 0) {
        return std::numeric_limits<uint32_t>::max();
    }
    return total_j_tiles * total_k_tiles - 1;
}

bool
MatrixFlowEngine::rescueNextUseIsShort() const
{
    return rescueTileStepsToNextUse() <= 8;
}

bool
MatrixFlowEngine::rescueANextUseIsShort() const
{
    return rescueATileStepsToNextUse() <= 2;
}

bool
MatrixFlowEngine::nextRescueOpportunityNear(uint32_t rowIdx) const
{
    (void)rowIdx;
    const uint32_t tile_steps = rescueTileStepsToNextUse();
    if (vipRescueRecurrenceAwareV3Mode()) {
        return tile_steps <= 8;
    }
    return tile_steps <= 4;
}

uint32_t
MatrixFlowEngine::rescueATileStepsToNextUse() const
{
    const uint32_t future_reuse = rescueFutureJTileReuseCount();
    if (future_reuse == 0) {
        return std::numeric_limits<uint32_t>::max();
    }

    return 1;
}

uint32_t
MatrixFlowEngine::repeatVictimCount(Addr rowAddr) const
{
    auto it = bFallbackVictimCount.find(rowAddr);
    return it == bFallbackVictimCount.end() ? 0 : it->second;
}

bool
MatrixFlowEngine::criticalRowOrWindow(uint32_t rowIdx) const
{
    if (rowIdx >= targetReqsB) {
        return false;
    }

    const uint32_t launch_rows = std::min(targetReqsB, computeLaunchWindowB());
    const uint32_t lead_rows = std::max<uint32_t>(
        4, std::min<uint32_t>(holeFillLeadRowsConfig, launch_rows));
    const uint32_t critical_rows =
        std::min(targetReqsB, launch_rows + lead_rows);
    return rowIdx < critical_rows;
}

bool
MatrixFlowEngine::rescueCriticalityHigh(uint32_t rowIdx, Addr rowAddr) const
{
    const uint32_t future_reuse = rescueFutureITileReuseCount();
    const uint32_t repeat_count = repeatVictimCount(rowAddr);
    if (vipRescueRecurrenceAwareV3Mode()) {
        const uint32_t tile_steps = rescueTileStepsToNextUse();
        return repeat_count >= 2 ||
               nextRescueOpportunityNear(rowIdx) ||
               (future_reuse > 1 && tile_steps <= 12);
    }
    return nextRescueOpportunityNear(rowIdx) ||
           future_reuse > 1 ||
           repeat_count >= 2;
}

MatrixFlowEngine::VipAdmitClass
MatrixFlowEngine::classifyVipRescueAdmission(uint32_t rowIdx,
                                             Addr rowAddr) const
{
    if (!vipRescueMode() || !vipPoolEnabled() || rowIdx >= targetReqsB) {
        return VipAdmitClass::Reject;
    }

    const uint32_t future_reuse = rescueFutureITileReuseCount();
    if (future_reuse == 0) {
        return VipAdmitClass::Reject;
    }

    if (!rescueNextUseIsShort()) {
        return VipAdmitClass::Reject;
    }

    const auto src = currentBOracleSource[rowIdx];
    const auto dist = oracleDistanceBucket(rowIdx, targetReqsB);
    if (src == OracleSourceClass::NextOutput &&
        (dist == OracleDistanceBucket::Immediate ||
         dist == OracleDistanceBucket::Near)) {
        return VipAdmitClass::Reject;
    }

    if (vipRescueRecurrenceAwareV2Mode() ||
        vipRescueRecurrenceAwareV3Mode()) {
        const uint32_t repeat_count = repeatVictimCount(rowAddr);
        const bool rescue_high = rescueCriticalityHigh(rowIdx, rowAddr);
        const bool rescue_near = nextRescueOpportunityNear(rowIdx);
        const uint32_t tile_steps = rescueTileStepsToNextUse();
        const bool long_span = tile_steps > 4;
        const bool weak_admit =
            vipRescueRecurrenceAwareV3Mode()
                ? (long_span ? (future_reuse > 1 && rescue_near)
                             : (future_reuse > 0 || rescue_near))
                : (future_reuse > 0 || rescue_near);

        if (future_reuse == 0 && !rescue_near) {
            return VipAdmitClass::Reject;
        }

        if (!rescue_high) {
            return VipAdmitClass::Reject;
        }

        if (repeat_count >= 2) {
            return VipAdmitClass::Strong;
        }
        return weak_admit ? VipAdmitClass::Weak : VipAdmitClass::Reject;
    }

    const uint32_t repeat_count = repeatVictimCount(rowAddr);
    const bool critical = criticalRowOrWindow(rowIdx);
    if (future_reuse > 1 || repeat_count >= 2) {
        return critical ? VipAdmitClass::Strong : VipAdmitClass::Reject;
    }

    return critical ? VipAdmitClass::Weak : VipAdmitClass::Reject;
}

bool
MatrixFlowEngine::shouldAdmitVipRescueRow(uint32_t rowIdx) const
{
    if (!vipRescueMode() || !vipPoolEnabled() || rowIdx >= targetReqsB) {
        return false;
    }

    if (vipRescueAntiDeadBlockMode()) {
        const Addr rowAddr = ctx.baseB +
            ((static_cast<Addr>(ctx.k + rowIdx) * ctx.size + ctx.j) *
             ctx.elemBytes);
        return classifyVipRescueAdmission(rowIdx, rowAddr) !=
               VipAdmitClass::Reject;
    }

    const uint32_t future_reuse = rescueFutureITileReuseCount();
    if (future_reuse == 0) {
        return false;
    }

    const bool short_next_use = rescueNextUseIsShort();
    const bool multi_future_use = future_reuse > 1;
    if (!short_next_use && !multi_future_use) {
        return false;
    }

    const auto src = currentBOracleSource[rowIdx];
    const auto dist = oracleDistanceBucket(rowIdx, targetReqsB);
    if (src == OracleSourceClass::NextOutput &&
        (dist == OracleDistanceBucket::Immediate ||
         dist == OracleDistanceBucket::Near)) {
        return false;
    }

    return true;
}

bool
MatrixFlowEngine::shouldAdmitVipRescueARow(uint32_t rowIdx) const
{
    if (!vipABRescueMode() || !vipPoolEnabled() || rowIdx >= targetReqsA) {
        return false;
    }

    const uint32_t future_reuse = rescueFutureJTileReuseCount();
    if (future_reuse == 0) {
        return false;
    }

    const bool short_next_use = rescueANextUseIsShort();
    const bool multi_future_use = future_reuse > 1;
    return short_next_use || multi_future_use;
}

int
MatrixFlowEngine::vipSourcePriority(VipSourceClass src) const
{
    switch (src) {
      case VipSourceClass::NextOutput:
        return 5;
      case VipSourceClass::Claim:
        return 4;
      case VipSourceClass::CarryOver:
        return 3;
      case VipSourceClass::Rescue:
        return 2;
      case VipSourceClass::RescueA:
        return 1;
      case VipSourceClass::CurrentWindow:
        return 0;
      case VipSourceClass::None:
      default:
        return -1;
    }
}

int
MatrixFlowEngine::mhotSourcePriority(VipSourceClass src) const
{
    switch (src) {
      case VipSourceClass::NextOutput:
        return 3;
      case VipSourceClass::Claim:
        return 2;
      case VipSourceClass::Normal:
        return 2;
      case VipSourceClass::CarryOver:
        return 1;
      case VipSourceClass::CurrentWindow:
        return 1;
      case VipSourceClass::Rescue:
      case VipSourceClass::RescueA:
      case VipSourceClass::None:
      default:
        return -1;
    }
}

MatrixFlowEngine::VipAdmitClass
MatrixFlowEngine::classifyMHotAdmission(uint32_t rowIdx, uint32_t totalRows,
                                        VipSourceClass source,
                                        BProtectionClass protectionClass) const
{
    if (!mhotPoolEnabled()) {
        return VipAdmitClass::Reject;
    }

    const auto dist = oracleDistanceBucket(rowIdx, totalRows);
    if (mhotGapAwareNextCutMode() ||
        coverageShadowMode() ||
        smartPatternPrefetchMode() ||
        mhotCoverageBlindspotCandidateMode()) {
        switch (source) {
          case VipSourceClass::Normal:
            return dist == OracleDistanceBucket::Near
                ? VipAdmitClass::Strong
                : VipAdmitClass::Reject;
          case VipSourceClass::CurrentWindow:
            return dist == OracleDistanceBucket::Far
                ? VipAdmitClass::Weak
                : VipAdmitClass::Reject;
          default:
            return VipAdmitClass::Reject;
        }
    }

    switch (source) {
      case VipSourceClass::NextOutput:
        if (dist == OracleDistanceBucket::Immediate ||
            dist == OracleDistanceBucket::Near) {
            return VipAdmitClass::Strong;
        }
        if (mhotEnhancedMode() &&
            dist == OracleDistanceBucket::Far) {
            return VipAdmitClass::Weak;
        }
        return VipAdmitClass::Reject;
      case VipSourceClass::Claim:
      case VipSourceClass::CarryOver:
        return mhotEnhancedMode() ? VipAdmitClass::Weak
                                  : VipAdmitClass::Reject;
      case VipSourceClass::CurrentWindow:
      case VipSourceClass::Normal:
      case VipSourceClass::Rescue:
      case VipSourceClass::RescueA:
      case VipSourceClass::None:
      default:
        return VipAdmitClass::Reject;
    }
}

void
MatrixFlowEngine::recordMHotInsertStats(VipSourceClass source,
                                        OracleDistanceBucket dist,
                                        VipAdmitClass admitClass,
                                        bool coverageBlindspot)
{
    stats.mhotInsertCount++;
    if (coverageBlindspot) {
        stats.mhotInsertCoverageBlindspotCount++;
    }
    if (admitClass == VipAdmitClass::Strong) {
        stats.mhotInsertDefaultCount++;
    } else if (admitClass == VipAdmitClass::Weak) {
        stats.mhotInsertEnhancedCount++;
    }

    switch (source) {
      case VipSourceClass::NextOutput:
        if (dist == OracleDistanceBucket::Immediate) {
            stats.mhotInsertNextOutputImmediateCount++;
        } else if (dist == OracleDistanceBucket::Near) {
            stats.mhotInsertNextOutputNearCount++;
        } else if (dist == OracleDistanceBucket::Far) {
            stats.mhotInsertNextOutputFarCount++;
        }
        break;
      case VipSourceClass::Claim:
        stats.mhotInsertClaimCount++;
        break;
      case VipSourceClass::CarryOver:
        stats.mhotInsertCarryOverCount++;
        break;
      case VipSourceClass::Normal:
        if (dist == OracleDistanceBucket::Near) {
            stats.mhotInsertNormalNearCount++;
        }
        break;
      case VipSourceClass::CurrentWindow:
        if (dist == OracleDistanceBucket::Far) {
            stats.mhotInsertCurrentWindowFarCount++;
        }
        break;
      default:
        break;
    }
}

void
MatrixFlowEngine::recordMHotHitStats(VipSourceClass source,
                                     OracleDistanceBucket dist,
                                     bool coverageBlindspot)
{
    stats.mhotHitCount++;
    stats.mhotReuseHitCount++;
    stats.mhotMainlinePreventedRemoteCount++;
    if (coverageBlindspot) {
        stats.mhotHitOnCoverageBlindspotCount++;
        stats.mhotMainlinePreventedRemoteFromCoverageBlindspotCount++;
        stats.mhotHitOnCoverageBlindspotByDistance[
            static_cast<int>(dist)] += 1;
    }

    switch (source) {
      case VipSourceClass::NextOutput:
        if (dist == OracleDistanceBucket::Immediate) {
            stats.mhotHitOnNextOutputImmediateCount++;
        } else if (dist == OracleDistanceBucket::Near) {
            stats.mhotHitOnNextOutputNearCount++;
        } else if (dist == OracleDistanceBucket::Far) {
            stats.mhotHitOnNextOutputFarCount++;
        }
        break;
      case VipSourceClass::Claim:
        stats.mhotHitOnClaimCount++;
        break;
      case VipSourceClass::CarryOver:
        stats.mhotHitOnCarryOverCount++;
        break;
      case VipSourceClass::Normal:
        if (dist == OracleDistanceBucket::Near) {
            stats.mhotHitOnNormalNearCount++;
        }
        break;
      case VipSourceClass::CurrentWindow:
        if (dist == OracleDistanceBucket::Far) {
            stats.mhotHitOnCurrentWindowFarCount++;
        }
        break;
      default:
        break;
    }
}

void
MatrixFlowEngine::recordMHotMaterializeCurrent(VipSourceClass source,
                                               uint32_t rowIdx,
                                               uint32_t totalRows,
                                               bool coverageBlindspot)
{
    stats.mhotRowsServedToComputeCount++;
    stats.mhotMaterializeToCurrentCount++;
    if (coverageBlindspot) {
        stats.mhotRowsServedToComputeFromCoverageBlindspotCount++;
    }
    (void)source;
    (void)rowIdx;
    (void)totalRows;
}

int
MatrixFlowEngine::findVipSlot(Addr rowAddr, Addr rowBytes) const
{
    if (!vipPoolEnabled()) {
        return -1;
    }

    for (size_t i = 0; i < vipBSlots.size(); ++i) {
        const auto &slot = vipBSlots[i];
        if (slot.valid && slot.rowAddr == rowAddr &&
            slot.rowBytes == rowBytes) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

int
MatrixFlowEngine::findMHotSlot(Addr rowAddr, Addr rowBytes) const
{
    if (!mhotPoolEnabled()) {
        return -1;
    }

    for (size_t i = 0; i < mhotBSlots.size(); ++i) {
        const auto &slot = mhotBSlots[i];
        if (slot.valid && slot.rowAddr == rowAddr &&
            slot.rowBytes == rowBytes) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

int
MatrixFlowEngine::findCoverageShadowSlot(Addr rowAddr, Addr rowBytes) const
{
    if (!coverageShadowPoolEnabled()) {
        return -1;
    }

    if (coverage2DGatherPingPongFirstCutMode() && !coverageShadowActiveValid) {
        return -1;
    }

    return findCoverageShadowSlotInBank(coverageShadowBSlots, rowAddr, rowBytes);
}

int
MatrixFlowEngine::findCoverageShadowSlotInBank(
    const std::vector<VipRowSlot> &slots, Addr rowAddr, Addr rowBytes) const
{
    for (size_t i = 0; i < slots.size(); ++i) {
        const auto &slot = slots[i];
        if (slot.valid && slot.rowAddr == rowAddr &&
            slot.rowBytes == rowBytes) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

int
MatrixFlowEngine::selectVipVictimSlot(VipSourceClass incoming,
                                      VipAdmitClass admitClass,
                                      uint32_t rescueRepeatVictimCount,
                                      bool rescueCritical,
                                      bool rescueShortNextUse,
                                      bool rescueNearOpportunity,
                                      uint32_t rescueFutureReuseCount) const
{
    if (!vipPoolEnabled()) {
        return -1;
    }

    for (size_t i = 0; i < vipBSlots.size(); ++i) {
        if (!vipBSlots[i].valid) {
            return static_cast<int>(i);
        }
    }

    const int incomingPri = vipSourcePriority(incoming);
    const bool incomingStrong = admitClass == VipAdmitClass::Strong;
    const bool incomingRescue = incoming == VipSourceClass::Rescue ||
                                incoming == VipSourceClass::RescueA;
    const int incomingRescueScore =
        incomingRescue ? rescueRetentionScore(rescueRepeatVictimCount,
                                              rescueCritical,
                                              rescueShortNextUse,
                                              rescueNearOpportunity,
                                              rescueFutureReuseCount,
                                              admitClass)
                       : 0;
    int victim = -1;
    int victimPri = std::numeric_limits<int>::max();
    bool victimStrong = true;
    int victimRescueScore = std::numeric_limits<int>::max();
    uint64_t victimEpoch = 0;
    for (size_t i = 0; i < vipBSlots.size(); ++i) {
        const auto &slot = vipBSlots[i];
        const int slotPri = vipSourcePriority(slot.source);
        const bool slotStrong = slot.admitClass == VipAdmitClass::Strong;
        const bool slotRescue = slot.source == VipSourceClass::Rescue ||
                                slot.source == VipSourceClass::RescueA;
        const int slotRescueScore =
            slotRescue ? rescueRetentionScore(slot.rescueRepeatVictimCount,
                                              slot.rescueCritical,
                                              slot.rescueShortNextUse,
                                              slot.rescueNearOpportunity,
                                              slot.rescueFutureReuseCount,
                                              slot.admitClass)
                       : 0;
        if (slotPri > incomingPri) {
            continue;
        }
        if (slotPri == incomingPri) {
            if (slotRescue && incomingRescue) {
                if (slotRescueScore > incomingRescueScore) {
                    continue;
                }
            } else if (slotStrong && !incomingStrong) {
                continue;
            }
        }
        if (victim < 0 || slotPri < victimPri ||
            (slotPri == victimPri && slotRescue && incomingRescue &&
             slotRescueScore < victimRescueScore) ||
            (slotPri == victimPri &&
             (!slotRescue || !incomingRescue) &&
             slotStrong != victimStrong && !slotStrong) ||
            (slotPri == victimPri &&
             ((slotRescue && incomingRescue &&
               slotRescueScore == victimRescueScore) ||
              (((!slotRescue || !incomingRescue) &&
                slotStrong == victimStrong))) &&
             slot.epoch < victimEpoch)) {
            victim = static_cast<int>(i);
            victimPri = slotPri;
            victimStrong = slotStrong;
            victimRescueScore = slotRescueScore;
            victimEpoch = slot.epoch;
        }
    }
    return victim;
}

int
MatrixFlowEngine::selectMHotVictimSlot(VipSourceClass incoming,
                                       VipAdmitClass admitClass,
                                       bool coverageBlindspotCandidate) const
{
    if (!mhotPoolEnabled()) {
        return -1;
    }

    size_t begin = 0;
    size_t end = mhotBSlots.size();
    if (mhotCoverageBlindspotCandidateMode()) {
        const size_t blindspot_quota = mhotCoverageBlindspotQuotaRows();
        if (coverageBlindspotCandidate) {
            end = std::min(end, blindspot_quota);
        } else {
            begin = std::min(end, blindspot_quota);
        }
    }

    for (size_t i = begin; i < end; ++i) {
        if (!mhotBSlots[i].valid) {
            return static_cast<int>(i);
        }
    }

    const int incomingScore =
        mhotSourcePriority(incoming) * 8 +
        (admitClass == VipAdmitClass::Strong ? 4 : 0);
    int victim = -1;
    int victimScore = std::numeric_limits<int>::max();
    uint64_t victimEpoch = 0;
    for (size_t i = begin; i < end; ++i) {
        const auto &slot = mhotBSlots[i];
        const int slotScore =
            mhotSourcePriority(slot.source) * 8 +
            (slot.admitClass == VipAdmitClass::Strong ? 4 : 0);
        if (slotScore > incomingScore) {
            continue;
        }
        if (victim < 0 || slotScore < victimScore ||
            (slotScore == victimScore && slot.epoch < victimEpoch)) {
            victim = static_cast<int>(i);
            victimScore = slotScore;
            victimEpoch = slot.epoch;
        }
    }
    return victim;
}

int
MatrixFlowEngine::selectCoverageShadowVictimSlot() const
{
    if (!coverageShadowPoolEnabled()) {
        return -1;
    }

    return selectCoverageShadowVictimSlotInBank(coverageShadowBSlots);
}

int
MatrixFlowEngine::selectCoverageShadowVictimSlotInBank(
    const std::vector<VipRowSlot> &slots) const
{
    for (size_t i = 0; i < slots.size(); ++i) {
        if (!slots[i].valid) {
            return static_cast<int>(i);
        }
    }

    int victim = -1;
    uint64_t victimEpoch = 0;
    for (size_t i = 0; i < slots.size(); ++i) {
        const auto &slot = slots[i];
        if (victim < 0 || slot.epoch < victimEpoch) {
            victim = static_cast<int>(i);
            victimEpoch = slot.epoch;
        }
    }
    return victim;
}

int
MatrixFlowEngine::rescueRetentionScore(uint32_t repeatVictimCount,
                                       bool criticalRow,
                                       bool shortNextUse,
                                       bool nearOpportunity,
                                       uint32_t futureReuseCount,
                                       VipAdmitClass admitClass) const
{
    int score = 0;
    if (admitClass == VipAdmitClass::Strong) {
        score += 8;
    } else if (admitClass == VipAdmitClass::Weak) {
        score += 4;
    }
    if (criticalRow) {
        score += 6;
    }
    if (shortNextUse) {
        score += 4;
    }
    if (nearOpportunity) {
        score += 5;
    }
    score += 3 * static_cast<int>(std::min<uint32_t>(futureReuseCount, 3));
    score += 4 * static_cast<int>(std::min<uint32_t>(repeatVictimCount, 3));
    return score;
}

bool
MatrixFlowEngine::tryInsertVipRow(Addr rowAddr, Addr rowBytes,
                                  const uint8_t *src,
                                  VipSourceClass source,
                                  BProtectionClass protectionClass,
                                  VipAdmitClass admitClass,
                                  uint32_t rescueRepeatVictimCount,
                                  bool rescueCritical,
                                  bool rescueShortNextUse,
                                  bool rescueNearOpportunity,
                                  uint32_t rescueFutureReuseCount)
{
    if (!vipPoolEnabled() || rowBytes == 0 || rowBytes > kMaxTileRowBytes ||
        src == nullptr || admitClass == VipAdmitClass::Reject) {
        return false;
    }

    int slotIdx = findVipSlot(rowAddr, rowBytes);
    if (slotIdx < 0) {
        slotIdx = selectVipVictimSlot(source, admitClass,
                                      rescueRepeatVictimCount,
                                      rescueCritical,
                                      rescueShortNextUse,
                                      rescueNearOpportunity,
                                      rescueFutureReuseCount);
    }
    if (slotIdx < 0) {
        return false;
    }

    auto &slot = vipBSlots[slotIdx];
    if (slot.valid &&
        (slot.rowAddr != rowAddr || slot.rowBytes != rowBytes)) {
        noteBFallbackAutopsyEvict(slot.rowAddr, slot.source);
        stats.vipEvictionCount++;
        if (slot.source == VipSourceClass::Rescue ||
            slot.source == VipSourceClass::RescueA) {
            stats.vipRescueEvictionCount++;
            if (slot.source == VipSourceClass::RescueA) {
                stats.vipRescueAEvictionCount++;
            }
        }
        if (slot.admitClass == VipAdmitClass::Weak) {
            stats.vipEvictWeakAdmitCount++;
        }
        if (slot.source == VipSourceClass::CurrentWindow) {
            stats.vipEvictLowPriorityCount++;
        }
        if (slot.source == VipSourceClass::None) {
            stats.vipEvictNormalCount++;
        }
        if (vipPoolOccupancy > 0) {
            --vipPoolOccupancy;
        }
    }

    uint8_t *dst = vipBBuffer.data() +
        static_cast<size_t>(slotIdx) * kMaxTileRowBytes;
    std::memcpy(dst, src, rowBytes);

    const bool wasValid = slot.valid;
    slot.valid = true;
    slot.rowAddr = rowAddr;
    slot.rowBytes = rowBytes;
    slot.source = source;
    slot.protectionClass = protectionClass;
    slot.admitClass = admitClass;
    slot.rescueRepeatVictimCount = rescueRepeatVictimCount;
    slot.rescueFutureReuseCount = rescueFutureReuseCount;
    slot.rescueCritical = rescueCritical;
    slot.rescueShortNextUse = rescueShortNextUse;
    slot.rescueNearOpportunity = rescueNearOpportunity;
    slot.epoch = ++vipInsertEpoch;
    noteBFallbackAutopsyInsert(rowAddr, source);

    stats.vipInsertCount++;
    if (admitClass == VipAdmitClass::Strong) {
        stats.vipStrongAdmitCount++;
    } else if (admitClass == VipAdmitClass::Weak) {
        stats.vipWeakAdmitCount++;
    }
    switch (source) {
      case VipSourceClass::NextOutput:
        stats.vipInsertFromNextOutputCount++;
        break;
      case VipSourceClass::Claim:
        stats.vipInsertFromClaimCount++;
        break;
      case VipSourceClass::CarryOver:
        stats.vipInsertFromCarryOverCount++;
        break;
      case VipSourceClass::Rescue:
        stats.vipRescueInsertCount++;
        if (admitClass == VipAdmitClass::Strong) {
            stats.vipRescueInsertStrongCount++;
        } else if (admitClass == VipAdmitClass::Weak) {
            stats.vipRescueInsertWeakCount++;
        }
        if (rescueRepeatVictimCount >= 2) {
            stats.repeatVictimPromotedCount++;
        }
        if (rescueCritical) {
            stats.criticalWindowVipInsertCount++;
        }
        break;
      case VipSourceClass::RescueA:
        stats.vipRescueInsertCount++;
        stats.vipRescueAInsertCount++;
        break;
      default:
        break;
    }

    if (!wasValid) {
        ++vipPoolOccupancy;
        if (vipPoolOccupancy > vipPoolOccupancyPeakObserved) {
            vipPoolOccupancyPeakObserved = vipPoolOccupancy;
            stats.vipPoolOccupancyPeak = vipPoolOccupancyPeakObserved;
        }
    }

    return true;
}

bool
MatrixFlowEngine::tryInsertMHotRow(Addr rowAddr, Addr rowBytes,
                                   const uint8_t *src,
                                   VipSourceClass source,
                                   BProtectionClass protectionClass,
                                   VipAdmitClass admitClass,
                                   OracleDistanceBucket dist,
                                   bool coverageBlindspotCandidate)
{
    if (!mhotPoolEnabled() || rowBytes == 0 || rowBytes > kMaxTileRowBytes ||
        src == nullptr || admitClass == VipAdmitClass::Reject) {
        return false;
    }

    int slotIdx = findMHotSlot(rowAddr, rowBytes);
    if (slotIdx < 0) {
        slotIdx = selectMHotVictimSlot(source, admitClass,
                                       coverageBlindspotCandidate);
    }
    if (slotIdx < 0) {
        return false;
    }

    auto &slot = mhotBSlots[slotIdx];
    if (slot.valid &&
        (slot.rowAddr != rowAddr || slot.rowBytes != rowBytes)) {
        noteBFallbackAutopsyEvict(slot.rowAddr, slot.source);
        stats.mhotEvictionCount++;
        if (slot.coverageBlindspot) {
            stats.mhotEvictionCoverageBlindspotCount++;
        }
        if (mhotPoolOccupancy > 0) {
            --mhotPoolOccupancy;
        }
    }

    uint8_t *dst = mhotBBuffer.data() +
        static_cast<size_t>(slotIdx) * kMaxTileRowBytes;
    std::memcpy(dst, src, rowBytes);

    const bool wasValid = slot.valid;
    slot.valid = true;
    slot.rowAddr = rowAddr;
    slot.rowBytes = rowBytes;
    slot.source = source;
    slot.protectionClass = protectionClass;
    slot.admitClass = admitClass;
    slot.coverageBlindspot = coverageBlindspotCandidate;
    slot.epoch = ++mhotInsertEpoch;
    noteBFallbackAutopsyInsert(rowAddr, source);
    recordMHotInsertStats(source, dist, admitClass,
                          coverageBlindspotCandidate);

    if (!wasValid) {
        ++mhotPoolOccupancy;
        if (mhotPoolOccupancy > mhotPoolOccupancyPeakObserved) {
            mhotPoolOccupancyPeakObserved = mhotPoolOccupancy;
            stats.mhotOccupancyPeak = mhotPoolOccupancyPeakObserved;
        }
    }

    return true;
}

bool
MatrixFlowEngine::tryInsertCoverageShadowRow(Addr rowAddr, Addr rowBytes,
                                             const uint8_t *src,
                                             CoverageShadowInsertSource source)
{
    return tryInsertCoverageShadowRowInBank(
        coverageShadowBBuffer, coverageShadowBSlots, coverageShadowPoolOccupancy,
        rowAddr, rowBytes, src, source);
}

bool
MatrixFlowEngine::tryInsertCoverageShadowRowLowPriority(
    Addr rowAddr, Addr rowBytes, const uint8_t *src,
    CoverageShadowInsertSource source)
{
    if (!coverageShadowPoolEnabled() || rowBytes == 0 ||
        rowBytes > kMaxTileRowBytes || src == nullptr) {
        return false;
    }

    int slotIdx = -1;
    for (size_t i = 0; i < coverageShadowBSlots.size(); ++i) {
        if (!coverageShadowBSlots[i].valid) {
            slotIdx = static_cast<int>(i);
            break;
        }
    }
    if (slotIdx < 0) {
        return false;
    }

    auto &slot = coverageShadowBSlots[slotIdx];
    uint8_t *dst = coverageShadowBBuffer.data() +
        static_cast<size_t>(slotIdx) * kMaxTileRowBytes;
    std::memcpy(dst, src, rowBytes);

    slot.valid = true;
    slot.rowAddr = rowAddr;
    slot.rowBytes = rowBytes;
    slot.source = VipSourceClass::Normal;
    slot.protectionClass = BProtectionClass::None;
    slot.admitClass = VipAdmitClass::Strong;
    slot.coverageBlindspot = true;
    slot.epoch = ++coverageShadowInsertEpoch;
    noteBFallbackAutopsyInsert(rowAddr, slot.source);

    stats.coverageShadowInsertCount++;
    if (source == CoverageShadowInsertSource::NextOutput) {
        stats.coverageShadowInsertFromNextOutputCount++;
    } else if (source == CoverageShadowInsertSource::Gather) {
        stats.coverageShadowInsertFromGatherCount++;
    } else {
        stats.coverageShadowInsertFromNextKCount++;
    }

    ++coverageShadowPoolOccupancy;
    if (coverageShadowPoolOccupancy >
        coverageShadowPoolOccupancyPeakObserved) {
        coverageShadowPoolOccupancyPeakObserved =
            coverageShadowPoolOccupancy;
        stats.coverageShadowOccupancyPeak =
            coverageShadowPoolOccupancyPeakObserved;
    }

    if (slotIdx < static_cast<int>(coverageShadowLowPrioritySlot.size()) &&
        !coverageShadowLowPrioritySlot[slotIdx]) {
        coverageShadowLowPrioritySlot[slotIdx] = true;
        ++coverageShadowLowPriorityOccupancy;
        if (coverageShadowLowPriorityOccupancy >
            coverageShadowLowPriorityOccupancyPeakObserved) {
            coverageShadowLowPriorityOccupancyPeakObserved =
                coverageShadowLowPriorityOccupancy;
            stats.coverageShadowLowPriorityOccupancyPeak =
                coverageShadowLowPriorityOccupancyPeakObserved;
        }
    }

    return true;
}

bool
MatrixFlowEngine::tryInsertCoverageShadowRowInBank(
    std::vector<uint8_t> &buffer, std::vector<VipRowSlot> &slots,
    uint32_t &occupancy, Addr rowAddr, Addr rowBytes, const uint8_t *src,
    CoverageShadowInsertSource source)
{
    if (!coverageShadowPoolEnabled() || rowBytes == 0 ||
        rowBytes > kMaxTileRowBytes || src == nullptr) {
        return false;
    }

    int slotIdx = findCoverageShadowSlotInBank(slots, rowAddr, rowBytes);
    if (slotIdx < 0) {
        slotIdx = selectCoverageShadowVictimSlotInBank(slots);
    }
    if (slotIdx < 0) {
        return false;
    }

    auto &slot = slots[slotIdx];
    if (&slots == &coverageShadowBSlots &&
        slotIdx < static_cast<int>(coverageShadowLowPrioritySlot.size()) &&
        coverageShadowLowPrioritySlot[slotIdx]) {
        coverageShadowLowPrioritySlot[slotIdx] = false;
        if (coverageShadowLowPriorityOccupancy > 0) {
            --coverageShadowLowPriorityOccupancy;
        }
    }
    if (slot.valid &&
        (slot.rowAddr != rowAddr || slot.rowBytes != rowBytes)) {
        noteBFallbackAutopsyEvict(slot.rowAddr, slot.source);
        stats.coverageShadowEvictionCount++;
        if (occupancy > 0) {
            --occupancy;
        }
    }

    uint8_t *dst = buffer.data() +
        static_cast<size_t>(slotIdx) * kMaxTileRowBytes;
    std::memcpy(dst, src, rowBytes);

    const bool wasValid = slot.valid;
    slot.valid = true;
    slot.rowAddr = rowAddr;
    slot.rowBytes = rowBytes;
    slot.source = VipSourceClass::Normal;
    slot.protectionClass = BProtectionClass::None;
    slot.admitClass = VipAdmitClass::Strong;
    slot.coverageBlindspot = true;
    slot.epoch = ++coverageShadowInsertEpoch;
    noteBFallbackAutopsyInsert(rowAddr, slot.source);

    stats.coverageShadowInsertCount++;
    if (source == CoverageShadowInsertSource::NextOutput) {
        stats.coverageShadowInsertFromNextOutputCount++;
    } else if (source == CoverageShadowInsertSource::Gather) {
        stats.coverageShadowInsertFromGatherCount++;
    } else {
        stats.coverageShadowInsertFromNextKCount++;
    }

    if (!wasValid) {
        ++occupancy;
        if (&slots == &coverageShadowBSlots &&
            occupancy >
            coverageShadowPoolOccupancyPeakObserved) {
            coverageShadowPoolOccupancyPeakObserved =
                occupancy;
            stats.coverageShadowOccupancyPeak =
                coverageShadowPoolOccupancyPeakObserved;
        }
    }

    return true;
}

bool
MatrixFlowEngine::tryServeCurrentARowFromVip(uint32_t rowIdx, bool countMiss)
{
    if (!vipABRescueMode() || !vipPoolEnabled() || rowIdx >= targetReqsA) {
        return false;
    }

    const Addr rowBytes =
        static_cast<Addr>(ctx.curTileK) * ctx.elemBytes;
    const Addr rowAddr = ctx.baseA +
        ((static_cast<Addr>(ctx.i + rowIdx) * ctx.size + ctx.k) *
         ctx.elemBytes);
    const int slotIdx = findVipSlot(rowAddr, rowBytes);
    if (slotIdx < 0) {
        if (countMiss) {
            stats.vipMissCount++;
            stats.vipRescueMissCount++;
            stats.vipRescueAMissCount++;
        }
        return false;
    }

    const auto slot = vipBSlots[slotIdx];
    uint8_t *dst = tileABuffer.data() + rowIdx * rowBytes;
    const uint8_t *src = vipBBuffer.data() +
        static_cast<size_t>(slotIdx) * kMaxTileRowBytes;
    std::memcpy(dst, src, rowBytes);

    ++reqsIssuedA;
    ++reqsCompletedA;
    stats.vipHitCount++;
    stats.vipMaterializeToCurrentCount++;
    stats.vipRescueHitCount++;
    stats.vipRescueAHitCount++;
    stats.vipRescueServedToComputeCount++;
    stats.vipRescueAServedToComputeCount++;
    stats.vipRescueAPreventedRemoteFetchCount++;
    stats.vipRescueReusedCount++;
    stats.vipRescueAReusedCount++;
    stats.rxAReadyCount++;
    recordCurrentAOracleOutcome(rowAddr, true);
    if (reqsCompletedB < targetReqsB) {
        stats.aFetchProgressDuringBFetch++;
    }
    updateABParallelOverlapTracking();
    if (reqsCompletedA == targetReqsA &&
        nextPrefetchTrigger == "a_ready") {
        maybePrefetchNextTile();
        maybePrefetchNextOutputTile();
    }

    if (slot.source == VipSourceClass::RescueA) {
        vipBSlots[slotIdx].epoch = ++vipInsertEpoch;
    } else {
        vipBSlots[slotIdx] = VipRowSlot();
        if (vipPoolOccupancy > 0) {
            --vipPoolOccupancy;
        }
    }
    return true;
}

bool
MatrixFlowEngine::tryServeCurrentBRowFromMHot(uint32_t rowIdx,
                                              BProtectionClass cls)
{
    if (!mhotPoolEnabled() || rowIdx >= targetReqsB ||
        currentBRowState[rowIdx] != BRowState::Empty) {
        return false;
    }

    const Addr rowBytes =
        static_cast<Addr>(ctx.curTileN) * ctx.elemBytes;
    const Addr rowAddr = ctx.baseB +
        ((static_cast<Addr>(ctx.k + rowIdx) * ctx.size + ctx.j) *
         ctx.elemBytes);
    const int slotIdx = findMHotSlot(rowAddr, rowBytes);
    if (slotIdx < 0) {
        return false;
    }

    const auto slot = mhotBSlots[slotIdx];
    uint8_t *dst = tileBBuffer.data() + rowIdx * rowBytes;
    const uint8_t *src = mhotBBuffer.data() +
        static_cast<size_t>(slotIdx) * kMaxTileRowBytes;
    std::memcpy(dst, src, rowBytes);

    const bool prefetchedLike =
        slot.source == VipSourceClass::NextOutput ||
        slot.source == VipSourceClass::Claim ||
        slot.source == VipSourceClass::CarryOver;
    currentBRowState[rowIdx] = prefetchedLike
        ? BRowState::ReadyFromNextOutput
        : BRowState::ReadyFromNormal;
    currentBProtectionClass[rowIdx] = cls;
    setCurrentOracleSource(rowIdx, oracleSourceFromVipSource(slot.source));
    if (cls != BProtectionClass::None) {
        recordProtectedBIssue(cls);
    }
    recordProtectedBReady(cls);
    ++reqsIssuedB;
    ++reqsCompletedB;
    ++stats.rxBReadyCount;
    recordMHotHitStats(slot.source, oracleDistanceBucket(rowIdx, targetReqsB),
                       slot.coverageBlindspot);
    recordMHotMaterializeCurrent(slot.source, rowIdx, targetReqsB,
                                 slot.coverageBlindspot);
    if (slot.source == VipSourceClass::Claim &&
        currentBFutureClaimed[rowIdx]) {
        clearCurrentFutureClaimRow(rowIdx, false, true, false);
    }
    if (slot.source == VipSourceClass::Claim ||
        slot.source == VipSourceClass::NextOutput) {
        stats.prefetchedBRowsConsumed++;
    }
    if (slot.source == VipSourceClass::CarryOver) {
        stats.carryOverRowsConsumedPostBoundary++;
    }
    if (reqsCompletedA < targetReqsA) {
        stats.bFetchProgressDuringAFetch++;
    }
    recordCurrentBOracleOutcome(rowIdx, true);
    updateABParallelOverlapTracking();
    mhotBSlots[slotIdx].epoch = ++mhotInsertEpoch;
    return true;
}

bool
MatrixFlowEngine::tryPrimeCurrentBRowFromCoverageShadow(uint32_t rowIdx,
                                                        BProtectionClass cls)
{
    if (!coverageShadowPoolEnabled() || rowIdx >= targetReqsB ||
        currentBRowState[rowIdx] != BRowState::Empty) {
        return false;
    }

    if (coverage2DGatherPingPongFirstCutMode() &&
        (!coverageShadowActiveValid ||
         !coverageShadowContextMatches(coverageShadowActiveCtx, ctx))) {
        return false;
    }

    const Addr rowBytes =
        static_cast<Addr>(ctx.curTileN) * ctx.elemBytes;
    const Addr rowAddr = ctx.baseB +
        ((static_cast<Addr>(ctx.k + rowIdx) * ctx.size + ctx.j) *
         ctx.elemBytes);
    const int slotIdx = findCoverageShadowSlot(rowAddr, rowBytes);
    if (slotIdx < 0) {
        return false;
    }

    uint8_t *dst = tileBBuffer.data() + rowIdx * rowBytes;
    const uint8_t *src = coverageShadowBBuffer.data() +
        static_cast<size_t>(slotIdx) * kMaxTileRowBytes;
    std::memcpy(dst, src, rowBytes);

    currentBRowState[rowIdx] = BRowState::ReadyFromNormal;
    currentBProtectionClass[rowIdx] = cls;
    setCurrentOracleSource(rowIdx, OracleSourceClass::Normal);
    ++reqsIssuedB;
    ++reqsCompletedB;
    ++stats.rxBReadyCount;
    ++stats.coverageShadowHitCount;
    ++stats.coverageShadowPrimeHitCount;
    ++stats.coverageShadowRowsServedToComputeCount;
    ++stats.coverageShadowPreventedRemoteCount;
    ++stats.coverageShadowPrimeRowsMaterializedCount;
    if (reqsCompletedA < targetReqsA) {
        stats.bFetchProgressDuringAFetch++;
    }
    recordCurrentBOracleOutcome(rowIdx, true);
    updateABParallelOverlapTracking();
    coverageShadowBSlots[slotIdx].epoch = ++coverageShadowInsertEpoch;
    currentBCoverageShadowScanned[rowIdx] = true;
    currentBCoverageShadowKnownMiss[rowIdx] = false;
    return true;
}

void
MatrixFlowEngine::primeCurrentBRowsFromCoverageShadow()
{
    if (!(coverageShadowV2Mode() || coverage2DGatherV2Mode() ||
          coverage2DGatherNoStarvationV3Mode() ||
          coverage2DGatherMinGuaranteeFirstCutMode() ||
          coverage2DGatherPingPongFirstCutMode()) ||
        !coverageShadowPoolEnabled()) {
        return;
    }

    if (coverage2DGatherPingPongFirstCutMode() &&
        (!coverageShadowActiveValid ||
         !coverageShadowContextMatches(coverageShadowActiveCtx, ctx))) {
        return;
    }

    for (uint32_t row = 0; row < targetReqsB; ++row) {
        if (currentBRowState[row] != BRowState::Empty) {
            continue;
        }
        const Addr rowAddr = ctx.baseB +
            ((static_cast<Addr>(ctx.k + row) * ctx.size + ctx.j) *
             ctx.elemBytes);
        if (!isCoverageBlindspotTargetPattern(row, BProtectionClass::None,
                                              rowAddr)) {
            continue;
        }
        currentBCoverageShadowScanned[row] = true;
        ++stats.coverageShadowPrimeCheckCount;
        if (!tryPrimeCurrentBRowFromCoverageShadow(row,
                                                   BProtectionClass::None)) {
            currentBCoverageShadowKnownMiss[row] = true;
            ++stats.coverageShadowPrimeMissCount;
        }
    }
}

bool
MatrixFlowEngine::tryServeCurrentBRowFromCoverageShadow(uint32_t rowIdx,
                                                        BProtectionClass cls)
{
    if (!coverageShadowPoolEnabled() || rowIdx >= targetReqsB ||
        currentBRowState[rowIdx] != BRowState::Empty) {
        return false;
    }

    if (coverage2DGatherPingPongFirstCutMode() &&
        (!coverageShadowActiveValid ||
         !coverageShadowContextMatches(coverageShadowActiveCtx, ctx))) {
        return false;
    }

    const Addr rowBytes =
        static_cast<Addr>(ctx.curTileN) * ctx.elemBytes;
    const Addr rowAddr = ctx.baseB +
        ((static_cast<Addr>(ctx.k + rowIdx) * ctx.size + ctx.j) *
         ctx.elemBytes);
    const int slotIdx = findCoverageShadowSlot(rowAddr, rowBytes);
    if (slotIdx < 0) {
        return false;
    }

    uint8_t *dst = tileBBuffer.data() + rowIdx * rowBytes;
    const uint8_t *src = coverageShadowBBuffer.data() +
        static_cast<size_t>(slotIdx) * kMaxTileRowBytes;
    std::memcpy(dst, src, rowBytes);

    currentBRowState[rowIdx] = BRowState::ReadyFromNormal;
    currentBProtectionClass[rowIdx] = cls;
    setCurrentOracleSource(rowIdx, OracleSourceClass::Normal);
    ++reqsIssuedB;
    ++reqsCompletedB;
    ++stats.rxBReadyCount;
    ++stats.coverageShadowHitCount;
    ++stats.coverageShadowRowsServedToComputeCount;
    ++stats.coverageShadowPreventedRemoteCount;
    if (reqsCompletedA < targetReqsA) {
        stats.bFetchProgressDuringAFetch++;
    }
    recordCurrentBOracleOutcome(rowIdx, true);
    updateABParallelOverlapTracking();
    coverageShadowBSlots[slotIdx].epoch = ++coverageShadowInsertEpoch;
    currentBCoverageShadowScanned[rowIdx] = true;
    currentBCoverageShadowKnownMiss[rowIdx] = false;
    return true;
}

bool
MatrixFlowEngine::tryServeNextOutputRowFromMHot(uint32_t rowIdx,
                                                BProtectionClass cls)
{
    if (!mhotPoolEnabled() || rowIdx >= nextOutputCtx.curTileK) {
        return false;
    }

    const Addr rowBytes =
        static_cast<Addr>(nextOutputCtx.curTileN) * nextOutputCtx.elemBytes;
    const Addr rowAddr = nextOutputCtx.baseB +
        ((static_cast<Addr>(nextOutputCtx.k + rowIdx) * nextOutputCtx.size +
          nextOutputCtx.j) * nextOutputCtx.elemBytes);
    const int slotIdx = findMHotSlot(rowAddr, rowBytes);
    if (slotIdx < 0) {
        return false;
    }

    const auto slot = mhotBSlots[slotIdx];
    uint8_t *dst = nextOutputTileBBuffer.data() + rowIdx * rowBytes;
    const uint8_t *src = mhotBBuffer.data() +
        static_cast<size_t>(slotIdx) * kMaxTileRowBytes;
    std::memcpy(dst, src, rowBytes);

    nextOutputFetchBBounceActive[rowIdx] = false;
    nextOutputFetchBRowGeneration[rowIdx] = activeNextOutputPrefetchGeneration;
    nextOutputPrefetchFromMHot[rowIdx] = true;
    nextOutputPrefetchMHotSource[rowIdx] = slot.source;
    ++nextOutputPrefetchReqsIssuedB;
    stats.nextOutputPrefetchRowsIssued++;
    ++nextOutputPrefetchRowsBCompleted;
    ++stats.rxBReadyCount;
    setNextOutputFutureClaim(rowIdx, cls);
    recordProtectedBIssue(cls);
    recordProtectedBReady(cls);
    recordMHotHitStats(slot.source,
                       oracleDistanceBucket(rowIdx, nextOutputCtx.curTileK),
                       slot.coverageBlindspot);
    if (phase == Phase::WriteC && writeCOverlapWindowActive) {
        ++writeCBRowsIssued;
        stats.bRowsIssuedDuringWriteC++;
        stats.nextOutputProgressDuringWriteC++;
    }
    if (nextOutputPrefetchRowsBCompleted == nextOutputPrefetchTargetB) {
        nextOutputPrefetchReadyB = true;
    }
    mhotBSlots[slotIdx].epoch = ++mhotInsertEpoch;
    return true;
}

bool
MatrixFlowEngine::tryServeCurrentBRowFromVip(uint32_t rowIdx, bool countMiss)
{
    if (!vipPoolEnabled() || rowIdx >= targetReqsB ||
        currentBRowState[rowIdx] != BRowState::Empty) {
        return false;
    }

    const Addr rowBytes =
        static_cast<Addr>(ctx.curTileN) * ctx.elemBytes;
    const Addr rowAddr = ctx.baseB +
        ((static_cast<Addr>(ctx.k + rowIdx) * ctx.size + ctx.j) *
         ctx.elemBytes);
    const int slotIdx = findVipSlot(rowAddr, rowBytes);
    if (slotIdx < 0) {
        if (countMiss || currentBVipBacked[rowIdx]) {
            stats.vipMissCount++;
            if (vipRescueMode()) {
                stats.vipRescueMissCount++;
            }
        }
        recordOracleSelectedRowMiss(rowIdx);
        currentBVipBacked[rowIdx] = false;
        currentBVipSource[rowIdx] = VipSourceClass::None;
        currentBVipAdmitClass[rowIdx] = VipAdmitClass::Reject;
        currentBVipPrefetchedCounted[rowIdx] = false;
        return false;
    }

    const auto slot = vipBSlots[slotIdx];
    uint8_t *dst = tileBBuffer.data() + rowIdx * rowBytes;
    const uint8_t *src = vipBBuffer.data() +
        static_cast<size_t>(slotIdx) * kMaxTileRowBytes;
    std::memcpy(dst, src, rowBytes);

    currentBRowState[rowIdx] = slot.source == VipSourceClass::Rescue
        ? BRowState::ReadyFromNormal
        : BRowState::ReadyFromNextOutput;
    currentBProtectionClass[rowIdx] = slot.protectionClass;
    currentBVipBacked[rowIdx] = false;
    currentBServedFromVip[rowIdx] = true;
    currentBVipSource[rowIdx] = slot.source;
    currentBVipAdmitClass[rowIdx] = slot.admitClass;
    setCurrentOracleSource(rowIdx, oracleSourceFromVipSource(slot.source));
    recordProtectedBReady(slot.protectionClass);

    ++reqsIssuedB;
    ++reqsCompletedB;
    stats.vipHitCount++;
    stats.vipMaterializeToCurrentCount++;
    stats.vipBRowsServedToCompute++;
    stats.vipBRowsPreventedFallbackCount++;
    if (slot.source == VipSourceClass::Rescue) {
        stats.vipRescueHitCount++;
        stats.vipRescueServedToComputeCount++;
        stats.vipRescuePreventedFallbackCount++;
        stats.vipRescueReusedCount++;
        if (slot.rescueRepeatVictimCount <= 1) {
            stats.eligibleRescueVictimsHitAfterFirstPromotionCount++;
        }
    }
    stats.rxBReadyCount++;
    if (!currentBVipPrefetchedCounted[rowIdx] &&
        (slot.source == VipSourceClass::NextOutput ||
         slot.source == VipSourceClass::Claim)) {
        stats.prefetchedBRowsConsumed++;
    }
    if (slot.source == VipSourceClass::CarryOver) {
        stats.carryOverRowsConsumedPostBoundary++;
    }
    switch (slot.source) {
      case VipSourceClass::NextOutput:
        stats.vipHitOnNextOutputCount++;
        if (oracleDistanceBucket(rowIdx, targetReqsB) ==
            OracleDistanceBucket::Immediate) {
            stats.vipHitOnNextOutputImmediateCount++;
        } else if (oracleDistanceBucket(rowIdx, targetReqsB) ==
                   OracleDistanceBucket::Near) {
            stats.vipHitOnNextOutputNearCount++;
        }
        break;
      case VipSourceClass::Claim:
        stats.vipHitOnClaimedFutureBCount++;
        break;
      case VipSourceClass::CarryOver:
        stats.vipHitOnCarryOverBCount++;
        break;
      case VipSourceClass::Rescue:
      case VipSourceClass::RescueA:
        break;
      default:
        break;
    }
    if (currentBFutureClaimed[rowIdx]) {
        clearCurrentFutureClaimRow(rowIdx, false, true, false);
    }
    currentBVipPrefetchedCounted[rowIdx] = false;
    if (oracleGuidedVipMode() && rowIdx < currentBOracleVipSelected.size() &&
        !currentBOracleVipSelected[rowIdx]) {
        currentBOracleVipSelected[rowIdx] = true;
        stats.oracleSelectedBRowsCount++;
    }
    if (oracleGuidedVipMode() && rowIdx < currentBOracleVipSelected.size() &&
        currentBOracleVipSelected[rowIdx]) {
        stats.oracleSelectedRowsServedByVipCount++;
        currentBOracleVipSelected[rowIdx] = false;
    }
    if (reqsCompletedA < targetReqsA) {
        stats.bFetchProgressDuringAFetch++;
    }
    recordCurrentBOracleOutcome(rowIdx, true);
    updateABParallelOverlapTracking();

    if (slot.source == VipSourceClass::Rescue) {
        vipBSlots[slotIdx].epoch = ++vipInsertEpoch;
    } else {
        vipBSlots[slotIdx] = VipRowSlot();
        if (vipPoolOccupancy > 0) {
            --vipPoolOccupancy;
        }
    }
    return true;
}

void
MatrixFlowEngine::primeCurrentBRowsFromVip()
{
    if (!vipPoolEnabled()) {
        return;
    }

    for (uint32_t row = 0; row < targetReqsB; ++row) {
        if (currentBRowState[row] != BRowState::Empty ||
            !currentBVipBacked[row]) {
            continue;
        }
        tryServeCurrentBRowFromVip(row, true);
    }
}

void
MatrixFlowEngine::clearNextOutputFutureClaims()
{
    std::fill(nextOutputFutureClaimed.begin(), nextOutputFutureClaimed.end(),
              false);
    std::fill(nextOutputFutureClaimClass.begin(),
              nextOutputFutureClaimClass.end(), BProtectionClass::None);
}

void
MatrixFlowEngine::clearStagedCurrentFutureClaims()
{
    std::fill(stagedCurrentBFutureClaimed.begin(),
              stagedCurrentBFutureClaimed.end(), false);
    std::fill(stagedCurrentBFutureClaimClass.begin(),
              stagedCurrentBFutureClaimClass.end(), BProtectionClass::None);
    stagedCurrentBFutureClaimGeneration = 0;
}

void
MatrixFlowEngine::clearCurrentFutureClaims(bool invalidated)
{
    if (invalidated && currentBFutureClaimActiveCount > 0) {
        stats.futureClaimInvalidatedCount += currentBFutureClaimActiveCount;
        stats.futureClaimClearedCount += currentBFutureClaimActiveCount;
    }

    std::fill(currentBFutureClaimed.begin(), currentBFutureClaimed.end(),
              false);
    std::fill(currentBFutureClaimClass.begin(),
              currentBFutureClaimClass.end(), BProtectionClass::None);
    currentBFutureClaimGeneration = 0;
    currentBFutureClaimActiveCount = 0;
}

void
MatrixFlowEngine::setNextOutputFutureClaim(uint32_t rowIdx,
                                           BProtectionClass cls)
{
    if (!claimBasedHoleFillingMode() || rowIdx >= nextOutputFutureClaimed.size()) {
        return;
    }

    if (!nextOutputFutureClaimed[rowIdx]) {
        nextOutputFutureClaimed[rowIdx] = true;
        nextOutputFutureClaimClass[rowIdx] = cls;
        stats.futureClaimSetCount++;
    }
}

void
MatrixFlowEngine::stageCurrentFutureClaimsFromNextOutput(
    uint32_t firstClaimedRow, uint32_t issuedRows, uint64_t generation)
{
    if (!claimBasedHoleFillingMode() || issuedRows == 0) {
        return;
    }

    clearStagedCurrentFutureClaims();
    stagedCurrentBFutureClaimGeneration = generation;

    const uint32_t limit = std::min<uint32_t>(
        std::min<uint32_t>(issuedRows, ctx.curTileK), kMaxTileDim);
    for (uint32_t row = firstClaimedRow; row < limit; ++row) {
        if (!nextOutputFutureClaimed[row]) {
            continue;
        }
        stagedCurrentBFutureClaimed[row] = true;
        stagedCurrentBFutureClaimClass[row] = nextOutputFutureClaimClass[row];
    }
}

void
MatrixFlowEngine::applyStagedCurrentFutureClaims()
{
    if (!claimBasedHoleFillingMode() || stagedCurrentBFutureClaimGeneration == 0) {
        return;
    }

    currentBFutureClaimGeneration = stagedCurrentBFutureClaimGeneration;
    currentBFutureClaimActiveCount = 0;
    for (uint32_t row = 0; row < targetReqsB; ++row) {
        if (!stagedCurrentBFutureClaimed[row] ||
            currentBRowState[row] != BRowState::Empty) {
            continue;
        }
        currentBFutureClaimed[row] = true;
        currentBFutureClaimClass[row] = stagedCurrentBFutureClaimClass[row];
        currentBProtectionClass[row] = stagedCurrentBFutureClaimClass[row];
        setCurrentOracleSource(row, OracleSourceClass::Claim);
        ++currentBFutureClaimActiveCount;
    }

    peakFutureClaimRowsObserved =
        std::max(peakFutureClaimRowsObserved, currentBFutureClaimActiveCount);
    stats.bRowsClaimedByFuture = peakFutureClaimRowsObserved;
    clearStagedCurrentFutureClaims();
}

void
MatrixFlowEngine::stageCurrentVipBackedRow(uint32_t rowIdx,
                                           VipSourceClass source,
                                           BProtectionClass protectionClass,
                                           VipAdmitClass admitClass,
                                           bool prefetchedCounted)
{
    if (!vipPoolEnabled() || rowIdx >= stagedCurrentBVipBacked.size() ||
        admitClass == VipAdmitClass::Reject) {
        return;
    }

    stagedCurrentBVipBacked[rowIdx] = true;
    stagedCurrentBVipSource[rowIdx] = source;
    stagedCurrentBVipProtectionClass[rowIdx] = protectionClass;
    stagedCurrentBVipAdmitClass[rowIdx] = admitClass;
    stagedCurrentBVipPrefetchedCounted[rowIdx] = prefetchedCounted;
}

void
MatrixFlowEngine::stageCurrentOracleVipSelectedRow(uint32_t rowIdx)
{
    if (!oracleGuidedVipMode() ||
        rowIdx >= stagedCurrentBOracleVipSelected.size()) {
        return;
    }

    stagedCurrentBOracleVipSelected[rowIdx] = true;
}

void
MatrixFlowEngine::applyStagedCurrentVipBackedRows()
{
    if (!vipPoolEnabled()) {
        clearStagedCurrentVipBacked();
        return;
    }

    for (uint32_t row = 0; row < targetReqsB; ++row) {
        if (!stagedCurrentBVipBacked[row]) {
            continue;
        }

        if (currentBRowState[row] == BRowState::ReadyFromNextOutput) {
            currentBRowState[row] = BRowState::Empty;
            if (reqsIssuedB > 0) {
                --reqsIssuedB;
            }
            if (reqsCompletedB > 0) {
                --reqsCompletedB;
            }
        }

        currentBVipBacked[row] = true;
        currentBVipSource[row] = stagedCurrentBVipSource[row];
        currentBVipAdmitClass[row] = stagedCurrentBVipAdmitClass[row];
        currentBVipPrefetchedCounted[row] =
            stagedCurrentBVipPrefetchedCounted[row];
        currentBProtectionClass[row] =
            stagedCurrentBVipProtectionClass[row];
        setCurrentOracleSource(
            row, oracleSourceFromVipSource(stagedCurrentBVipSource[row]));
        if (currentBFutureClaimed[row]) {
            clearCurrentFutureClaimRow(row, false, false, false);
        }
    }

    clearStagedCurrentVipBacked();
}

void
MatrixFlowEngine::applyStagedCurrentOracleVipSelectedRows()
{
    if (!oracleGuidedVipMode()) {
        clearStagedCurrentOracleVipSelected();
        return;
    }

    for (uint32_t row = 0; row < targetReqsB; ++row) {
        if (!stagedCurrentBOracleVipSelected[row] ||
            currentBOracleVipSelected[row]) {
            continue;
        }
        currentBOracleVipSelected[row] = true;
        stats.oracleSelectedBRowsCount++;
    }

    clearStagedCurrentOracleVipSelected();
}

void
MatrixFlowEngine::clearCurrentFutureClaimRow(uint32_t rowIdx, bool expired,
                                             bool consumedSuccess,
                                             bool invalidated)
{
    if (!claimBasedHoleFillingMode() || rowIdx >= currentBFutureClaimed.size() ||
        !currentBFutureClaimed[rowIdx]) {
        return;
    }

    currentBFutureClaimed[rowIdx] = false;
    currentBFutureClaimClass[rowIdx] = BProtectionClass::None;
    if (currentBFutureClaimActiveCount > 0) {
        --currentBFutureClaimActiveCount;
    }
    stats.futureClaimClearedCount++;
    if (expired) {
        stats.futureClaimExpiredCount++;
    }
    if (consumedSuccess) {
        stats.futureClaimConsumedSuccessCount++;
    }
    if (invalidated) {
        stats.futureClaimInvalidatedCount++;
    }
    if (currentBFutureClaimActiveCount == 0) {
        currentBFutureClaimGeneration = 0;
    }
}

bool
MatrixFlowEngine::expireCurrentFutureClaimsIfBlocked()
{
    if (!claimBasedHoleFillingMode() || reqsCompletedB >= targetReqsB) {
        return false;
    }

    if (nextIssuableBRow() >= 0 || nextClaimBlockedBRow() < 0) {
        return false;
    }

    uint32_t expired = 0;
    for (uint32_t row = 0; row < targetReqsB; ++row) {
        if (currentBRowState[row] != BRowState::Empty ||
            !currentBFutureClaimed[row]) {
            continue;
        }
        clearCurrentFutureClaimRow(row, true, false, false);
        currentBProtectionClass[row] = BProtectionClass::None;
        ++expired;
    }

    if (expired > 0 && hierarchicalProtectedBSchedulerMode()) {
        warn("%s: hier-trace expired %u blocked future-B claims at "
             "i=%u j=%u k=%u readyB=%u/%u issuedB=%u\n",
             name(), expired, ctx.i, ctx.j, ctx.k,
             reqsCompletedB, targetReqsB, reqsIssuedB);
    }

    return expired > 0;
}

uint32_t
MatrixFlowEngine::protectedBFrontier() const
{
    return std::min(targetReqsB, reqsCompletedB + computeLaunchWindowB());
}

bool
MatrixFlowEngine::isProtectedComputeWindowRow(uint32_t rowIdx) const
{
    return !computeDoneEvent.scheduled() &&
           reqsCompletedB < computeLaunchWindowB() &&
           rowIdx < computeLaunchWindowB();
}

bool
MatrixFlowEngine::isProtectedHoleFillingRow(uint32_t rowIdx) const
{
    return rowIdx < protectedBFrontier();
}

int
MatrixFlowEngine::nextProtectedBRow() const
{
    const int rowIdx = nextIssuableBRow();
    if (rowIdx < 0) {
        return -1;
    }

    const uint32_t row = static_cast<uint32_t>(rowIdx);
    if (isProtectedComputeWindowRow(row) ||
        isProtectedHoleFillingRow(row)) {
        return rowIdx;
    }
    return -1;
}

bool
MatrixFlowEngine::protectedNextOutputPending() const
{
    return nextOutputPrefetchValid &&
           !nextOutputPrefetchReadyB &&
           nextOutputPrefetchReqsIssuedB < nextOutputPrefetchTargetB &&
           (sharedPrefetchBCredits() > 0 ||
            nextOutputPendingRowCanUseMHot());
}

bool
MatrixFlowEngine::nextOutputPendingRowCanUseMHot() const
{
    if (!mhotPoolEnabled() || !mhotAllowsMainlineBypass() ||
        !nextOutputPrefetchValid ||
        nextOutputPrefetchReadyB ||
        nextOutputPrefetchReqsIssuedB >= nextOutputPrefetchTargetB) {
        return false;
    }

    uint32_t row = nextOutputPrefetchReqsIssuedB;
    if (smartPatternPrefetchMode()) {
        const int smartRow = selectSmartNextOutputPrefetchBRow();
        if (smartRow < 0) {
            return false;
        }
        row = static_cast<uint32_t>(smartRow);
    }
    if (row >= nextOutputCtx.curTileK) {
        return false;
    }

    const Addr rowBytes =
        static_cast<Addr>(nextOutputCtx.curTileN) * nextOutputCtx.elemBytes;
    const Addr rowAddr = nextOutputCtx.baseB +
        ((static_cast<Addr>(nextOutputCtx.k + row) * nextOutputCtx.size +
          nextOutputCtx.j) * nextOutputCtx.elemBytes);
    return findMHotSlot(rowAddr, rowBytes) >= 0;
}

bool
MatrixFlowEngine::hasProtectedBPressure() const
{
    return protectedNextOutputPending() || nextProtectedBRow() >= 0;
}

bool
MatrixFlowEngine::hasNormalBPressure() const
{
    const int rowIdx = nextIssuableBRow();
    if (rowIdx < 0) {
        return false;
    }
    return !isProtectedComputeWindowRow(rowIdx) &&
           !isProtectedHoleFillingRow(rowIdx);
}

bool
MatrixFlowEngine::hasFutureProtectedBPressure() const
{
    if (reqsCompletedB < targetReqsB) {
        return false;
    }

    return nextOutputPrefetchValid &&
           !nextOutputPrefetchReadyB &&
           nextOutputPrefetchReqsIssuedB < nextOutputPrefetchTargetB;
}

bool
MatrixFlowEngine::hasCurrentProtectedBPressure() const
{
    return nextProtectedBRow() >= 0;
}

bool
MatrixFlowEngine::currentProtectedBQuotaAvailable() const
{
    return currentProtectedBWinStreak < kCurrentProtectedBQuotaRows;
}

int
MatrixFlowEngine::nextCompetitiveBRow() const
{
    const uint32_t limit = normalIssueLimit();
    if (reqsIssuedB >= targetReqsB ||
        satOutstanding(reqsIssuedB, reqsCompletedB) >= kMaxInFlight) {
        return -1;
    }

    for (uint32_t row = nextNormalBRowCursor; row < limit; ++row) {
        if (currentBRowState[row] != BRowState::Empty) {
            continue;
        }
        if (isClaimedByFuture(row)) {
            continue;
        }
        if (isProtectedComputeWindowRow(row) ||
            isProtectedHoleFillingRow(row)) {
            continue;
        }
        return static_cast<int>(row);
    }

    for (uint32_t row = 0; row < std::min(limit, nextNormalBRowCursor); ++row) {
        if (currentBRowState[row] != BRowState::Empty) {
            continue;
        }
        if (isClaimedByFuture(row)) {
            continue;
        }
        if (isProtectedComputeWindowRow(row) ||
            isProtectedHoleFillingRow(row)) {
            continue;
        }
        if (hierarchicalProtectedBSchedulerMode()) {
            warn("%s: hier-trace recovered skipped competitive B row=%u "
                 "cursor=%u issuedB=%u completedB=%u targetB=%u\n",
                 name(), row, nextNormalBRowCursor, reqsIssuedB,
                 reqsCompletedB, targetReqsB);
        }
        return static_cast<int>(row);
    }
    return -1;
}

bool
MatrixFlowEngine::canIssueCompetitiveB() const
{
    return nextCompetitiveBRow() >= 0;
}

void
MatrixFlowEngine::maybeSeedFutureProtectedB()
{
    if (!hierarchicalProtectedBSchedulerMode() || !prefetchEnabled() ||
        nextOutputPrefetchValid || nextOutputPrefetchBDrainActive ||
        !hasNextOutputTile() || hasNextKTile()) {
        return;
    }

    // Future-B must not start stealing B-side credits before the current
    // tile's B stream is fully established. Otherwise compute can launch on
    // the minimum window and then processComputeDone() defers forever waiting
    // for remaining current-tile B rows.
    if (reqsCompletedB < targetReqsB) {
        return;
    }

    startOutputPrefetchTile(buildNextOutputContext());
}

uint8_t
MatrixFlowEngine::urgencyBucketA() const
{
    if (reqsCompletedA < computeLaunchWindowA()) {
        return 2;
    }
    if (reqsCompletedA < targetReqsA) {
        return computeDoneEvent.scheduled() ? 2 : 1;
    }
    return 0;
}

uint8_t
MatrixFlowEngine::urgencyBucketB() const
{
    if (reqsCompletedB < computeLaunchWindowB()) {
        return 2;
    }
    if (reqsCompletedB < targetReqsB) {
        return computeDoneEvent.scheduled() ? 2 : 1;
    }
    return 0;
}

uint8_t
MatrixFlowEngine::deficitBucketA() const
{
    const uint32_t remaining = targetReqsA > reqsIssuedA ?
        (targetReqsA - reqsIssuedA) : 0;
    if (remaining == 0) {
        return 0;
    }
    if (remaining > std::max<uint32_t>(computeLaunchWindowA(), 8)) {
        return 2;
    }
    return 1;
}

uint8_t
MatrixFlowEngine::deficitBucketB() const
{
    const uint32_t remaining = targetReqsB > reqsIssuedB ?
        (targetReqsB - reqsIssuedB) : 0;
    if (remaining == 0) {
        return 0;
    }
    if (remaining > std::max<uint32_t>(computeLaunchWindowB(), 8)) {
        return 2;
    }
    return 1;
}

uint8_t
MatrixFlowEngine::reuseBucketA() const
{
    return 0;
}

uint8_t
MatrixFlowEngine::reuseBucketB() const
{
    return 1;
}

uint8_t
MatrixFlowEngine::fallbackRiskBucketA() const
{
    return 0;
}

uint8_t
MatrixFlowEngine::fallbackRiskBucketB() const
{
    return canIssueB() ? 1 : 0;
}

void
MatrixFlowEngine::updateScoreStats(int aScore, int bScore)
{
    scoreSamplesAObserved++;
    scoreSamplesBObserved++;
    scoreSumAObserved += aScore;
    scoreSumBObserved += bScore;
    maxScoreAObserved = std::max<int64_t>(maxScoreAObserved, aScore);
    maxScoreBObserved = std::max<int64_t>(maxScoreBObserved, bScore);

    stats.avgScoreA = static_cast<double>(scoreSumAObserved) /
        std::max<uint64_t>(1, scoreSamplesAObserved);
    stats.avgScoreB = static_cast<double>(scoreSumBObserved) /
        std::max<uint64_t>(1, scoreSamplesBObserved);
    stats.maxScoreA = maxScoreAObserved;
    stats.maxScoreB = maxScoreBObserved;
}

void
MatrixFlowEngine::recordProtectedBReady(BProtectionClass cls)
{
    if (cls == BProtectionClass::None ||
        (!protectedBSchedulerMode() &&
         !hierarchicalProtectedBSchedulerMode())) {
        return;
    }

    stats.protectedBReadyCount++;
    if (cls == BProtectionClass::CarryOver) {
        stats.protectedBFromCarryOverCount++;
    }
}

void
MatrixFlowEngine::recordProtectedBIssue(BProtectionClass cls)
{
    if (cls == BProtectionClass::None ||
        (!protectedBSchedulerMode() &&
         !hierarchicalProtectedBSchedulerMode())) {
        return;
    }

    stats.protectedBIssueCount++;
    switch (cls) {
      case BProtectionClass::NextOutput:
        stats.protectedBFromNextOutputCount++;
        break;
      case BProtectionClass::HoleFilling:
      case BProtectionClass::FutureHoleFilling:
        stats.protectedBFromHoleFillingCount++;
        break;
      case BProtectionClass::ComputeWindow:
        stats.protectedBFromComputeWindowCount++;
        break;
      case BProtectionClass::CarryOver:
      case BProtectionClass::None:
        break;
    }

    if (!hierarchicalProtectedBSchedulerMode()) {
        return;
    }

    switch (cls) {
      case BProtectionClass::NextOutput:
        stats.futureProtectedBIssueCount++;
        stats.futureProtectedBFromNextOutputCount++;
        break;
      case BProtectionClass::FutureHoleFilling:
        stats.futureProtectedBIssueCount++;
        stats.futureProtectedBFromFutureHoleFillingCount++;
        break;
      case BProtectionClass::HoleFilling:
        stats.currentProtectedBIssueCount++;
        stats.currentProtectedBFromCurrentHoleFillingCount++;
        break;
      case BProtectionClass::ComputeWindow:
        stats.currentProtectedBIssueCount++;
        stats.currentProtectedBFromComputeWindowCount++;
        break;
      case BProtectionClass::CarryOver:
      case BProtectionClass::None:
        break;
    }
}

bool
MatrixFlowEngine::issueOneDualRxB()
{
    if (hasFutureProtectedBPressure()) {
        if (issueOneFutureProtectedB()) {
            currentProtectedBWinStreak = 0;
            stats.rxBPriorityWins++;
            return true;
        }
        return false;
    }

    if (hasCurrentProtectedBPressure()) {
        const bool mustServeCurrentProtected =
            !canIssueA() && !canIssueCompetitiveB();
        if (currentProtectedBQuotaAvailable() || mustServeCurrentProtected) {
            if (issueOneCurrentProtectedB()) {
                stats.rxBPriorityWins++;
                return true;
            }
        } else {
            stats.currentProtectedBQuotaExhaustCount++;
        }
    }

    if (issueOneCompetitiveB()) {
        currentProtectedBWinStreak = 0;
        return true;
    }

    return false;
}

uint32_t
MatrixFlowEngine::dualRxADeadlineScore() const
{
    if (reqsCompletedA < computeLaunchWindowA()) {
        return 3;
    }
    if (computeDoneEvent.scheduled() && reqsCompletedA < targetReqsA) {
        return 2;
    }
    if (reqsCompletedA < targetReqsA) {
        return 1;
    }
    return 0;
}

uint32_t
MatrixFlowEngine::dualRxBDeadlineScore() const
{
    if (hasFutureProtectedBPressure()) {
        return 4;
    }
    if (hasCurrentProtectedBPressure() && currentProtectedBQuotaAvailable()) {
        return 3;
    }
    if (reqsCompletedB < computeLaunchWindowB()) {
        return 2;
    }
    if (computeDoneEvent.scheduled() && reqsCompletedB < targetReqsB) {
        return 2;
    }
    if (canIssueCompetitiveB()) {
        return 1;
    }
    return 0;
}

uint32_t
MatrixFlowEngine::dualRxADeficitScore() const
{
    const uint32_t target = computeDoneEvent.scheduled()
        ? targetReqsA
        : computeLaunchWindowA();
    return target > reqsCompletedA ? (target - reqsCompletedA) : 0;
}

uint32_t
MatrixFlowEngine::dualRxBDeficitScore() const
{
    if (hasFutureProtectedBPressure()) {
        return nextOutputPrefetchTargetB > nextOutputPrefetchReqsIssuedB
            ? (nextOutputPrefetchTargetB - nextOutputPrefetchReqsIssuedB)
            : 0;
    }
    const uint32_t target = computeDoneEvent.scheduled()
        ? targetReqsB
        : computeLaunchWindowB();
    return target > reqsCompletedB ? (target - reqsCompletedB) : 0;
}

bool
MatrixFlowEngine::chooseDualRxIssueA()
{
    const bool aPending = canIssueA();
    const bool bPending = canIssueCompetitiveB();

    if (!bPending) {
        return true;
    }
    if (!aPending) {
        return false;
    }

    const uint32_t aCreditFloor =
        std::min(targetReqsA, abAMinCreditRowsConfig);
    if (aCreditFloor > 0 && reqsIssuedA < aCreditFloor) {
        stats.aCreditFloorHits++;
        stats.rxAPriorityWins++;
        return true;
    }

    const uint32_t aDeadline = dualRxADeadlineScore();
    const uint32_t bDeadline = dualRxBDeadlineScore();
    if (aDeadline != bDeadline) {
        if (aDeadline > bDeadline) {
            stats.rxADeadlineWins++;
            stats.rxAPriorityWins++;
            return true;
        }
        stats.rxBDeadlineWins++;
        stats.rxBPriorityWins++;
        return false;
    }

    const uint32_t aDeficit = dualRxADeficitScore();
    const uint32_t bDeficit = dualRxBDeficitScore();
    if (aDeficit != bDeficit) {
        if (aDeficit > bDeficit) {
            stats.rxADeficitWins++;
            stats.rxAPriorityWins++;
            return true;
        }
        stats.rxBDeficitWins++;
        stats.rxBPriorityWins++;
        return false;
    }

    if (abBiasBConfig > 0) {
        stats.bBiasWins++;
        stats.rxBPriorityWins++;
        return false;
    }

    stats.scoreTieBreakCount++;
    if (!lastFetchIssueWasA) {
        stats.rxAPriorityWins++;
        return true;
    }
    stats.rxBPriorityWins++;
    return false;
}

void
MatrixFlowEngine::updateDualRxQueueStats()
{
    if (!dualRxSchedulerMode()) {
        return;
    }

    const uint32_t rxAOcc = satOutstanding(reqsIssuedA, reqsCompletedA);
    const uint32_t rxBOcc = satOutstanding(reqsIssuedB, reqsCompletedB) +
        sharedPrefetchBOutstanding();
    if (rxAOcc > rxAQueuePeakObserved) {
        rxAQueuePeakObserved = rxAOcc;
        stats.rxAQueueOccupancyPeak = rxAQueuePeakObserved;
    }
    if (rxBOcc > rxBQueuePeakObserved) {
        rxBQueuePeakObserved = rxBOcc;
        stats.rxBQueueOccupancyPeak = rxBQueuePeakObserved;
    }
}

bool
MatrixFlowEngine::issueCurrentBRow(uint32_t rowIdx, BProtectionClass cls)
{
    const uint32_t issueLimit = normalIssueLimit();
    if (rowIdx >= issueLimit || rowIdx >= targetReqsB ||
        currentBRowState[rowIdx] != BRowState::Empty) {
        return false;
    }

    if (oracleGuidedVipMode() && !currentBOracleVipSelected[rowIdx]) {
        VipSourceClass vipSrc = VipSourceClass::None;
        switch (currentBOracleSource[rowIdx]) {
          case OracleSourceClass::NextOutput:
            vipSrc = VipSourceClass::NextOutput;
            break;
          case OracleSourceClass::Claim:
            vipSrc = VipSourceClass::Claim;
            break;
          case OracleSourceClass::CarryOver:
            vipSrc = VipSourceClass::CarryOver;
            break;
          case OracleSourceClass::CurrentWindow:
            vipSrc = VipSourceClass::CurrentWindow;
            break;
          case OracleSourceClass::Normal:
          case OracleSourceClass::NumClasses:
            vipSrc = VipSourceClass::None;
            break;
        }
        if (vipSrc != VipSourceClass::None &&
            classifyVipAdmission(rowIdx, targetReqsB, vipSrc, cls) !=
                VipAdmitClass::Reject) {
            currentBOracleVipSelected[rowIdx] = true;
            stats.oracleSelectedBRowsCount++;
        }
    }

    bool target_blindspot = false;
    if (mhotRuntimeFirstCutMode() || mhotGapAwareNextCutMode() ||
        coverageShadowMode() ||
        mhotCoverageBlindspotCandidateMode()) {
        const Addr rowAddr = ctx.baseB +
            ((static_cast<Addr>(ctx.k + rowIdx) * ctx.size + ctx.j) *
             ctx.elemBytes);
        target_blindspot =
            coverageShadowPoolEnabled() &&
            isCoverageBlindspotTargetPattern(rowIdx, cls, rowAddr);

        if ((coverage2DGatherNoStarvationV3Mode() ||
             coverage2DGatherMinGuaranteeFirstCutMode()) &&
            tryServeCurrentBRowFromVip(rowIdx, vipRescueMode())) {
            if (target_blindspot) {
                ++stats.gatherDeferredByVipCount;
            }
            return true;
        }

        if (target_blindspot) {
            if ((coverageShadowV2Mode() ||
                 coverage2DGatherNoStarvationV3Mode() ||
                 coverage2DGatherMinGuaranteeFirstCutMode()) &&
                currentBCoverageShadowScanned[rowIdx] &&
                currentBCoverageShadowKnownMiss[rowIdx]) {
                ++stats.coverageShadowFallbackLookupSkippedCount;
            } else {
                stats.coverageShadowCheckedOnFallbackCount++;
                currentBCoverageShadowScanned[rowIdx] = true;
                if (tryServeCurrentBRowFromCoverageShadow(rowIdx, cls)) {
                    stats.coverageShadowHitBeforeRemoteCount++;
                    return true;
                }
                currentBCoverageShadowKnownMiss[rowIdx] = true;
                stats.coverageShadowMissThenRemoteCount++;
            }
        }
        OracleSourceClass fallbackOracleSrc = currentBOracleSource[rowIdx];
        if (fallbackOracleSrc == OracleSourceClass::Normal) {
            // Fallback-time M-hot should still recognize rows that originated
            // from the mainline future-B chain even if the current row label
            // was never materialized before the normal path is about to fire.
            fallbackOracleSrc = oracleSourceFromProtectionClass(cls);
        }
        VipSourceClass hotSrc = VipSourceClass::None;
        switch (fallbackOracleSrc) {
          case OracleSourceClass::NextOutput:
            hotSrc = VipSourceClass::NextOutput;
            break;
          case OracleSourceClass::Claim:
            hotSrc = VipSourceClass::Claim;
            break;
          case OracleSourceClass::CarryOver:
            hotSrc = VipSourceClass::CarryOver;
            break;
          case OracleSourceClass::CurrentWindow:
            hotSrc = VipSourceClass::CurrentWindow;
            break;
          case OracleSourceClass::Normal:
            hotSrc = VipSourceClass::Normal;
            break;
          case OracleSourceClass::NumClasses:
          default:
            hotSrc = VipSourceClass::None;
            break;
        }

        const bool mhotCandidate =
            hotSrc != VipSourceClass::None &&
            classifyMHotAdmission(rowIdx, targetReqsB, hotSrc, cls) !=
                VipAdmitClass::Reject;
        const bool coverageCandidate =
            mhotCoverageBlindspotCandidateMode() &&
            hotSrc == VipSourceClass::Normal &&
            isCoverageBlindspotPromoted(rowIdx, cls, rowAddr);

        if (mhotCandidate || coverageCandidate) {
            stats.mhotCheckedOnFallbackCount++;
            if (tryServeCurrentBRowFromMHot(rowIdx, cls)) {
                stats.mhotHitBeforeRemoteCount++;
                return true;
            }
            stats.mhotMissThenRemoteCount++;
        } else if (!(coverage2DGatherNoStarvationV3Mode() ||
                     coverage2DGatherMinGuaranteeFirstCutMode()) &&
                   tryServeCurrentBRowFromVip(rowIdx, vipRescueMode())) {
            return true;
        }
    } else {
        if (tryServeCurrentBRowFromMHot(rowIdx, cls)) {
            return true;
        }

        if (tryServeCurrentBRowFromVip(rowIdx, vipRescueMode())) {
            return true;
        }
    }

    if (coverage2DGatherMinGuaranteeFirstCutMode() &&
        target_blindspot && coverageGatherValid &&
        !coverageGatherBRowIssued[rowIdx] &&
        sharedPrefetchBCredits() > 0) {
        issueCoverageGatherRow(rowIdx, ctx);
    }

    if (satOutstanding(reqsIssuedB, reqsCompletedB) >= kMaxInFlight) {
        return false;
    }

    if (oracleGuidedVipMode()) {
        recordOracleSelectedRowMiss(rowIdx);
        recordVipOracleGuidedReject(
            currentBOracleSource[rowIdx],
            oracleDistanceBucket(rowIdx, targetReqsB));
    }

    const Addr rowBytes =
        static_cast<Addr>(ctx.curTileN) * ctx.elemBytes;
    const Addr rowAddr = ctx.baseB +
        ((static_cast<Addr>(ctx.k + rowIdx) * ctx.size + ctx.j) *
         ctx.elemBytes);
    auto *dst = tileBBuffer.data() + rowIdx * rowBytes;
    auto [reqAddr, reqBytes, reqOffset] = planReadRequest(rowAddr, rowBytes);
    uint8_t *dmaDst = dst;
    if (reqBytes != rowBytes || reqOffset != 0) {
        fetchBBounceActive[rowIdx] = true;
        fetchBBounceReqAddr[rowIdx] = reqAddr;
        fetchBBounceReqBytes[rowIdx] = reqBytes;
        fetchBBounceRowBytes[rowIdx] = rowBytes;
        fetchBBounceOffset[rowIdx] = reqOffset;
        dmaDst = fetchBBounceBuffer.data() + rowIdx * readBouncePitch;
    } else {
        fetchBBounceActive[rowIdx] = false;
    }

    DPRINTF(MatrixFlow,
            "DMA FetchB row=%u cls=%u addr=%#llx bytes=%llu reqAddr=%#llx "
            "reqBytes=%llu reqOffset=%llu (tile i=%u j=%u k=%u)\n",
            rowIdx, static_cast<unsigned>(cls),
            static_cast<unsigned long long>(rowAddr),
            static_cast<unsigned long long>(rowBytes),
            static_cast<unsigned long long>(reqAddr),
            static_cast<unsigned long long>(reqBytes),
            static_cast<unsigned long long>(reqOffset), ctx.i, ctx.j,
            ctx.k);

    recordBFallbackAutopsy(rowIdx, cls, rowAddr);
    stats.totalDmaBytesRead += reqBytes;
    stats.fallbackBRowsFetched++;
    stats.normalFetchHoleRows++;
    stats.rxBIssueCount++;
    if (currentBOracleSource[rowIdx] == OracleSourceClass::Normal) {
        setCurrentOracleSource(rowIdx, oracleSourceFromProtectionClass(cls));
    }
    dmaPort.dmaAction(MemCmd::ReadReq, reqAddr, reqBytes,
                      &fetchBRowEvents[rowIdx], dmaDst, 0);
    currentBRowState[rowIdx] = BRowState::InflightFromNormal;
    currentBProtectionClass[rowIdx] = cls;
    ++reqsIssuedB;
    nextNormalBRowCursor = rowIdx + 1;
    if (cls != BProtectionClass::None) {
        recordProtectedBIssue(cls);
    }
    updateABParallelOverlapTracking();
    return true;
}

bool
MatrixFlowEngine::issueOneNextOutputProtectedB()
{
    if (!protectedNextOutputPending()) {
        return false;
    }

    if (nextOutputFirstIssueTick == 0) {
        nextOutputFirstIssueTick = curTick();
    }

    const Addr rowBytes =
        static_cast<Addr>(nextOutputCtx.curTileN) * nextOutputCtx.elemBytes;
    uint32_t row = nextOutputPrefetchReqsIssuedB;
    if (smartPatternPrefetchMode()) {
        const int smartRow = selectSmartNextOutputPrefetchBRow();
        if (smartRow < 0) {
            return false;
        }
        row = static_cast<uint32_t>(smartRow);
    }
    if (nextOutputPrefetchBRowIssued[row]) {
        return false;
    }
    const Addr rowAddr = nextOutputCtx.baseB +
        ((static_cast<Addr>(nextOutputCtx.k + row) * nextOutputCtx.size +
          nextOutputCtx.j) * nextOutputCtx.elemBytes);
    auto *dst = nextOutputTileBBuffer.data() + row * rowBytes;
    auto [reqAddr, reqBytes, reqOffset] = planReadRequest(rowAddr, rowBytes);
    uint8_t *dmaDst = dst;
    if (reqBytes != rowBytes || reqOffset != 0) {
        nextOutputFetchBBounceActive[row] = true;
        nextOutputFetchBBounceReqAddr[row] = reqAddr;
        nextOutputFetchBBounceReqBytes[row] = reqBytes;
        nextOutputFetchBBounceRowBytes[row] = rowBytes;
        nextOutputFetchBBounceOffset[row] = reqOffset;
        dmaDst = nextOutputFetchBBounceBuffer.data() + row * readBouncePitch;
    } else {
        nextOutputFetchBBounceActive[row] = false;
    }

    BProtectionClass cls = BProtectionClass::NextOutput;
    if (hierarchicalProtectedBSchedulerMode() &&
        row >= computeLaunchWindowB()) {
        cls = BProtectionClass::FutureHoleFilling;
    }

    if (mhotAllowsMainlineBypass() &&
        tryServeNextOutputRowFromMHot(row, cls)) {
        nextOutputPrefetchBRowIssued[row] = true;
        if (smartPatternPrefetchMode()) {
            stats.smartPrefetchBNextOutputPriorityIssueCount++;
            if (isSmartCoverageTargetForContext(nextOutputCtx, row)) {
                stats.smartPrefetchBNextOutputCoverageTargetIssueCount++;
            }
        }
        return true;
    }

    if (sharedPrefetchBCredits() == 0) {
        return false;
    }

    stats.totalDmaBytesRead += reqBytes;
    nextOutputFetchBRowGeneration[row] = activeNextOutputPrefetchGeneration;
    stats.rxBIssueCount++;
    dmaPort.dmaAction(MemCmd::ReadReq, reqAddr, reqBytes,
                      &nextOutputFetchBRowEvents[row], dmaDst, 0);
    ++nextOutputPrefetchReqsIssuedB;
    nextOutputPrefetchBRowIssued[row] = true;
    stats.nextOutputPrefetchRowsIssued++;
    if (smartPatternPrefetchMode()) {
        stats.smartPrefetchBNextOutputPriorityIssueCount++;
        if (isSmartCoverageTargetForContext(nextOutputCtx, row)) {
            stats.smartPrefetchBNextOutputCoverageTargetIssueCount++;
        }
    }
    setNextOutputFutureClaim(row, cls);
    recordProtectedBIssue(cls);
    if (phase == Phase::WriteC && writeCOverlapWindowActive) {
        ++writeCBRowsIssued;
        stats.bRowsIssuedDuringWriteC++;
    }
    return true;
}

bool
MatrixFlowEngine::issueOneCompetitiveB()
{
    const int rowIdx = nextCompetitiveBRow();
    if (rowIdx < 0) {
        return false;
    }
    return issueCurrentBRow(static_cast<uint32_t>(rowIdx),
                            BProtectionClass::None);
}

bool
MatrixFlowEngine::issueOneProtectedB()
{
    if (issueOneNextOutputProtectedB()) {
        stats.protectedBPriorityWins++;
        if (canIssueA()) {
            stats.protectedBBlocksACount++;
        }
        if (hasNormalBPressure()) {
            stats.protectedBBlocksNormalBCount++;
        }
        return true;
    }

    const int rowIdx = nextProtectedBRow();
    if (rowIdx < 0) {
        return false;
    }

    BProtectionClass cls = BProtectionClass::HoleFilling;
    if (isProtectedComputeWindowRow(rowIdx)) {
        cls = BProtectionClass::ComputeWindow;
    }
    if (issueCurrentBRow(static_cast<uint32_t>(rowIdx), cls)) {
        stats.protectedBPriorityWins++;
        if (canIssueA()) {
            stats.protectedBBlocksACount++;
        }
        if (hasNormalBPressure()) {
            stats.protectedBBlocksNormalBCount++;
        }
        return true;
    }

    return false;
}

bool
MatrixFlowEngine::issueOneFutureProtectedB()
{
    if (!hasFutureProtectedBPressure()) {
        return false;
    }

    if (issueOneNextOutputProtectedB()) {
        stats.futureProtectedBPriorityWins++;
        if (canIssueA()) {
            stats.futureProtectedBBlocksACount++;
        }
        if (hasCurrentProtectedBPressure()) {
            stats.futureProtectedBBlocksCurrentBCount++;
        }
        return true;
    }

    return false;
}

bool
MatrixFlowEngine::issueOneCurrentProtectedB()
{
    const int rowIdx = nextProtectedBRow();
    if (rowIdx < 0) {
        return false;
    }

    BProtectionClass cls = BProtectionClass::HoleFilling;
    if (isProtectedComputeWindowRow(rowIdx)) {
        cls = BProtectionClass::ComputeWindow;
    }

    if (!issueCurrentBRow(static_cast<uint32_t>(rowIdx), cls)) {
        return false;
    }

    stats.currentProtectedBPriorityWins++;
    if (canIssueA()) {
        stats.currentProtectedBBlocksACount++;
    }
    currentProtectedBWinStreak++;
    return true;
}

void
MatrixFlowEngine::recordFactorWin(bool choseA, int urgencyDiff,
                                  int deficitDiff, int reuseDiff,
                                  int fallbackRiskDiff, int biasDiff)
{
    const int winnerSign = choseA ? 1 : -1;
    const int urgencyMag =
        winnerSign * urgencyDiff > 0 ? std::abs(urgencyDiff) : 0;
    const int deficitMag =
        winnerSign * deficitDiff > 0 ? std::abs(deficitDiff) : 0;
    const int reuseMag =
        winnerSign * reuseDiff > 0 ? std::abs(reuseDiff) : 0;
    const int fallbackRiskMag =
        winnerSign * fallbackRiskDiff > 0 ? std::abs(fallbackRiskDiff) : 0;
    const int biasMag =
        winnerSign * biasDiff > 0 ? std::abs(biasDiff) : 0;

    if (biasMag > 0 && biasMag >= urgencyMag && biasMag >= deficitMag &&
        biasMag >= reuseMag && biasMag >= fallbackRiskMag) {
        stats.bBiasWins++;
        return;
    }
    if (urgencyMag >= deficitMag && urgencyMag >= reuseMag &&
        urgencyMag >= fallbackRiskMag && urgencyMag > 0) {
        stats.urgencyPriorityWins++;
        return;
    }
    if (deficitMag >= reuseMag && deficitMag >= fallbackRiskMag &&
        deficitMag > 0) {
        stats.deficitPriorityWins++;
        return;
    }
    if (fallbackRiskMag >= reuseMag && fallbackRiskMag > 0) {
        stats.fallbackRiskPriorityWins++;
        return;
    }
    if (reuseMag > 0) {
        stats.reusePriorityWins++;
        return;
    }
    stats.scoreTieBreakCount++;
}

bool
MatrixFlowEngine::chooseHierarchicalIssueA()
{
    const bool aPending = canIssueA();
    const bool bPending = canIssueCompetitiveB();

    if (!bPending) {
        return true;
    }
    if (!aPending) {
        return false;
    }

    const uint32_t aCreditFloor =
        std::min(targetReqsA, abAMinCreditRowsConfig);
    if (aCreditFloor > 0 && reqsIssuedA < aCreditFloor) {
        stats.aCreditFloorHits++;
        return true;
    }

    const int aUrgency = urgencyBucketA();
    const int bUrgency = 1;
    if (aUrgency != bUrgency) {
        stats.urgencyPriorityWins++;
        return aUrgency > bUrgency;
    }

    uint32_t normalBRemaining = 0;
    const uint32_t limit = normalIssueLimit();
    for (uint32_t row = nextNormalBRowCursor; row < limit; ++row) {
        if (currentBRowState[row] != BRowState::Empty) {
            continue;
        }
        if (isClaimedByFuture(row)) {
            continue;
        }
        if (isProtectedComputeWindowRow(row) ||
            isProtectedHoleFillingRow(row)) {
            continue;
        }
        ++normalBRemaining;
    }

    const int aDeficit = deficitBucketA();
    const int bDeficit = normalBRemaining >
        std::max<uint32_t>(computeLaunchWindowB(), 8) ? 2 : 1;
    if (aDeficit != bDeficit) {
        stats.deficitPriorityWins++;
        return aDeficit > bDeficit;
    }

    const int aFallbackRisk = fallbackRiskBucketA();
    const int bFallbackRisk = 1;
    if (aFallbackRisk != bFallbackRisk) {
        stats.fallbackRiskPriorityWins++;
        return aFallbackRisk > bFallbackRisk;
    }

    const int aReuse = reuseBucketA();
    const int bReuse = 1;
    if (aReuse != bReuse) {
        stats.reusePriorityWins++;
        return aReuse > bReuse;
    }

    if (abBiasBConfig > 0) {
        stats.bBiasWins++;
        return false;
    }

    stats.scoreTieBreakCount++;
    return !lastFetchIssueWasA;
}

bool
MatrixFlowEngine::chooseLightweightIssueA()
{
    const bool aPending = canIssueA();
    const bool bPending = canIssueB();

    if (!bPending) {
        return true;
    }
    if (!aPending) {
        return false;
    }

    const uint32_t aCreditFloor =
        std::min(targetReqsA, abAMinCreditRowsConfig);
    if (aCreditFloor > 0 && reqsIssuedA < aCreditFloor) {
        stats.aCreditFloorHits++;
        return true;
    }

    const int aUrgency = urgencyBucketA();
    const int bUrgency = urgencyBucketB();
    if (aUrgency != bUrgency) {
        stats.urgencyPriorityWins++;
        return aUrgency > bUrgency;
    }

    const int aDeficit = deficitBucketA();
    const int bDeficit = deficitBucketB();
    if (aDeficit != bDeficit) {
        stats.deficitPriorityWins++;
        return aDeficit > bDeficit;
    }

    const int aFallbackRisk = fallbackRiskBucketA();
    const int bFallbackRisk = fallbackRiskBucketB();
    if (aFallbackRisk != bFallbackRisk) {
        stats.fallbackRiskPriorityWins++;
        return aFallbackRisk > bFallbackRisk;
    }

    const int aReuse = reuseBucketA();
    const int bReuse = reuseBucketB();
    if (aReuse != bReuse) {
        stats.reusePriorityWins++;
        return aReuse > bReuse;
    }

    if (abBiasBConfig > 0) {
        stats.bBiasWins++;
        return false;
    }

    stats.scoreTieBreakCount++;
    return !lastFetchIssueWasA;
}

bool
MatrixFlowEngine::chooseFullScoreIssueA()
{
    const bool aPending = canIssueA();
    const bool bPending = canIssueB();

    if (!bPending) {
        return true;
    }
    if (!aPending) {
        return false;
    }

    const uint32_t aCreditFloor =
        std::min(targetReqsA, abAMinCreditRowsConfig);
    if (aCreditFloor > 0 && reqsIssuedA < aCreditFloor) {
        stats.aCreditFloorHits++;
        return true;
    }

    const int aUrgency = urgencyBucketA();
    const int bUrgency = urgencyBucketB();
    const int aDeficit = deficitBucketA();
    const int bDeficit = deficitBucketB();
    const int aReuse = reuseBucketA();
    const int bReuse = reuseBucketB();
    const int aFallbackRisk = fallbackRiskBucketA();
    const int bFallbackRisk = fallbackRiskBucketB();

    const int aScore =
        abWeightUrgencyConfig * aUrgency +
        abWeightDeficitConfig * aDeficit +
        abWeightReuseConfig * aReuse +
        abWeightFallbackRiskConfig * aFallbackRisk;
    const int bScore =
        abWeightUrgencyConfig * bUrgency +
        abWeightDeficitConfig * bDeficit +
        abWeightReuseConfig * bReuse +
        abWeightFallbackRiskConfig * bFallbackRisk +
        abBiasBConfig;

    updateScoreStats(aScore, bScore);

    if (aScore != bScore) {
        recordFactorWin(
            aScore > bScore,
            abWeightUrgencyConfig * (aUrgency - bUrgency),
            abWeightDeficitConfig * (aDeficit - bDeficit),
            abWeightReuseConfig * (aReuse - bReuse),
            abWeightFallbackRiskConfig * (aFallbackRisk - bFallbackRisk),
            -abBiasBConfig);
        return aScore > bScore;
    }

    stats.scoreTieBreakCount++;
    return !lastFetchIssueWasA;
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
    std::fill(nextPrefetchARowIssued.begin(), nextPrefetchARowIssued.end(),
              false);
    std::fill(nextPrefetchBRowIssued.begin(), nextPrefetchBRowIssued.end(),
              false);
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
    std::fill(nextOutputPrefetchFromMHot.begin(),
              nextOutputPrefetchFromMHot.end(), false);
    std::fill(nextOutputPrefetchMHotSource.begin(),
              nextOutputPrefetchMHotSource.end(), VipSourceClass::None);
    std::fill(nextOutputPrefetchBRowIssued.begin(),
              nextOutputPrefetchBRowIssued.end(), false);
    clearNextOutputFutureClaims();
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
    if (mhotPoolEnabled() && prefetched_rows > 0) {
        for (uint32_t row = 0; row < prefetched_rows; ++row) {
            if (!nextOutputPrefetchFromMHot[row]) {
                continue;
            }
            recordMHotMaterializeCurrent(nextOutputPrefetchMHotSource[row],
                                         row, ctx.curTileK, false);
        }
    }
    if (vipPoolEnabled() && prefetched_rows > 0) {
        const Addr rowBytes =
            static_cast<Addr>(ctx.curTileN) * ctx.elemBytes;
        for (uint32_t row = 0; row < prefetched_rows; ++row) {
            const BProtectionClass cls =
                row < computeLaunchWindowB()
                    ? BProtectionClass::NextOutput
                    : BProtectionClass::FutureHoleFilling;
            const VipAdmitClass admit =
                classifyVipAdmission(row, ctx.curTileK,
                                     VipSourceClass::NextOutput, cls);
            if (admit == VipAdmitClass::Reject) {
                continue;
            }
            stageCurrentOracleVipSelectedRow(row);
            const Addr rowAddr = ctx.baseB +
                ((static_cast<Addr>(ctx.k + row) * ctx.size + ctx.j) *
                 ctx.elemBytes);
            if (!tryInsertVipRow(rowAddr, rowBytes,
                                 tileBBuffer.data() + row * rowBytes,
                                 VipSourceClass::NextOutput, cls, admit)) {
                continue;
            }
            recordVipOracleGuidedAdmit(
                row, ctx.curTileK, VipSourceClass::NextOutput);
            if (vipForcedDiversionMode()) {
                stageCurrentVipBackedRow(row, VipSourceClass::NextOutput,
                                         cls, admit, true);
            }
        }
    }
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
    if (hierarchicalProtectedBSchedulerMode()) {
        warn("%s: hier-trace issueFetchDescriptor desc=%#llx\n",
             name(), static_cast<unsigned long long>(pendingDescAddr));
    }
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
    if (hierarchicalProtectedBSchedulerMode()) {
        warn("%s: hier-trace descriptorLoaded size=%u A=%#llx B=%#llx "
             "C=%#llx flag=%#llx\n",
             name(), ctx.size,
             static_cast<unsigned long long>(ctx.baseA),
             static_cast<unsigned long long>(ctx.baseB),
             static_cast<unsigned long long>(ctx.baseC),
             static_cast<unsigned long long>(ctx.flagAddr));
    }

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
    std::fill(currentARowIssued.begin(), currentARowIssued.end(), false);
    aWaitedForBThisTile = false;
    bWaitedForAThisTile = false;
    lastFetchIssueWasA = false;
    currentProtectedBWinStreak = 0;
    clearCoverageGatherForCurrentTile();
    if (coverage2DGatherPingPongFirstCutMode()) {
        maybeActivateCoverageShadowBankForCurrentTile();
    }
    if (coverage2DGatherV2Mode() ||
        coverage2DGatherNoStarvationV3Mode() ||
        coverage2DGatherMinGuaranteeFirstCutMode() ||
        coverage2DGatherPingPongFirstCutMode()) {
        startCoverageGatherForCurrentTile();
    }
    if (baselineSchedulerMode()) {
        if (targetReqsA == 0) {
            issueFetchBTile();
            return;
        }
        trySendMoreA();
        return;
    }

    // Parallel modes set up B immediately so A/B DMA can overlap.
    issueFetchBTile();
}

bool
MatrixFlowEngine::issueOneFetchA()
{
    if (reqsIssuedA >= targetReqsA) {
        return false;
    }

    uint32_t r = reqsIssuedA;
    if (smartPatternPrefetchMode()) {
        const int smartRow = selectSmartCurrentARow();
        if (smartRow >= 0) {
            r = static_cast<uint32_t>(smartRow);
        }
    }
    if (r >= targetReqsA || currentARowIssued[r]) {
        return false;
    }
    currentARowIssued[r] = true;
    if (tryServeCurrentARowFromVip(r, vipABRescueMode())) {
        if (smartPatternPrefetchMode()) {
            const Addr rowAddr = ctx.baseA +
                ((static_cast<Addr>(ctx.i + r) * ctx.size + ctx.k) *
                 ctx.elemBytes);
            stats.smartPrefetchAIssueCount++;
            if (oracleARecurrenceBucket(rowAddr) ==
                OracleARecurrenceBucket::Repeat) {
                stats.smartPrefetchARepeatPriorityCount++;
            }
            if (oracleAReuseBucket() == OracleAReuseBucket::Multi &&
                oracleADistanceBucket() != OracleDistanceBucket::Far) {
                stats.smartPrefetchANearMultiPriorityCount++;
            }
        }
        return true;
    }

    if (satOutstanding(reqsIssuedA, reqsCompletedA) >= kMaxInFlight) {
        currentARowIssued[r] = false;
        return false;
    }

    const Addr rowBytes =
        static_cast<Addr>(ctx.curTileK) * ctx.elemBytes;
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
    stats.aRowsIssued++;
    stats.rxAIssueCount++;
    if (smartPatternPrefetchMode()) {
        const Addr oracleRowAddr = ctx.baseA +
            ((static_cast<Addr>(ctx.i + r) * ctx.size + ctx.k) *
             ctx.elemBytes);
        stats.smartPrefetchAIssueCount++;
        if (oracleARecurrenceBucket(oracleRowAddr) ==
            OracleARecurrenceBucket::Repeat) {
            stats.smartPrefetchARepeatPriorityCount++;
        }
        if (oracleAReuseBucket() == OracleAReuseBucket::Multi &&
            oracleADistanceBucket() != OracleDistanceBucket::Far) {
            stats.smartPrefetchANearMultiPriorityCount++;
        }
    }
    dmaPort.dmaAction(MemCmd::ReadReq, reqAddr, reqBytes,
                      &fetchARowEvents[r], dmaDst, 0);
    ++reqsIssuedA;
    updateABParallelOverlapTracking();
    return true;
}

void
MatrixFlowEngine::trySendMoreA()
{
    while (issueOneFetchA()) {
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
    std::fill(currentBProtectionClass.begin(), currentBProtectionClass.end(),
              BProtectionClass::None);
    clearCurrentVipBacked();
    clearCurrentOracleVipSelected();
    clearCurrentOracleTracking();
    clearCurrentFutureClaims(true);
    std::fill(currentBCoverageShadowScanned.begin(),
              currentBCoverageShadowScanned.end(), false);
    std::fill(currentBCoverageShadowKnownMiss.begin(),
              currentBCoverageShadowKnownMiss.end(), false);
    if ((coverage2DGatherNoStarvationV3Mode() ||
         coverage2DGatherMinGuaranteeFirstCutMode() ||
         coverage2DGatherPingPongFirstCutMode()) &&
        vipRescueRecurrenceAwareV3Mode()) {
        ++stats.vipRescueModeActiveCount;
    }

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
            currentBProtectionClass[r] = BProtectionClass::CarryOver;
            setCurrentOracleSource(r, OracleSourceClass::CarryOver);
            recordProtectedBReady(BProtectionClass::CarryOver);
        }
        for (uint32_t r = prefetched_rows; r < covered_rows; ++r) {
            currentBRowState[r] = BRowState::InflightFromNextOutput;
            currentBProtectionClass[r] = BProtectionClass::CarryOver;
            setCurrentOracleSource(r, OracleSourceClass::CarryOver);
        }
    } else if (prefetched_rows > 0) {
        for (uint32_t r = 0; r < prefetched_rows; ++r) {
            currentBRowState[r] = BRowState::ReadyFromNextOutput;
            currentBProtectionClass[r] = BProtectionClass::NextOutput;
            setCurrentOracleSource(r, OracleSourceClass::NextOutput);
            recordProtectedBReady(BProtectionClass::NextOutput);
        }
    }

    if (hierarchicalProtectedBSchedulerMode() && prefetched_rows > 0) {
        stats.futureProtectedBReadyAtBoundaryCount += prefetched_rows;
    }

    reqsIssuedB = prefetched_rows + inflight_rows;
    reqsCompletedB = prefetched_rows;
    targetReqsB = ctx.curTileK;
    nextNormalBRowCursor = 0;
    applyStagedCurrentFutureClaims();
    applyStagedCurrentVipBackedRows();
    applyStagedCurrentOracleVipSelectedRows();
    primeCurrentBRowsFromVip();
    primeCurrentBRowsFromCoverageShadow();
    if (!(coverage2DGatherV2Mode() ||
          coverage2DGatherNoStarvationV3Mode() ||
          coverage2DGatherMinGuaranteeFirstCutMode())) {
        startCoverageGatherForCurrentTile();
    }
    if (oracleGuidedVipMode()) {
        for (uint32_t row = 0; row < targetReqsB; ++row) {
            if (currentBOracleVipSelected[row] &&
                currentBRowState[row] == BRowState::ReadyFromNextOutput &&
                !currentBServedFromVip[row]) {
                recordOracleSelectedRowMiss(row);
            }
        }
    }
    for (uint32_t row = 0; row < targetReqsB; ++row) {
        if (currentBRowState[row] == BRowState::ReadyFromNextOutput) {
            recordCurrentBOracleOutcome(row, true);
        }
    }
    currentPrefetchedBValid = false;
    currentPrefetchedBRows = 0;
    lastFetchIssueWasA = false;
    if (baselineSchedulerMode()) {
        trySendMoreB();
        tryLaunchComputeTile();
        return;
    }
    serviceParallelFetchAB();
}

bool
MatrixFlowEngine::issueOneFetchB()
{
    const int rowIdx = nextIssuableBRow();
    const uint32_t issueLimit = normalIssueLimit();
    if (rowIdx < 0) {
        if (issueLimit < targetReqsB &&
            nextNormalBRowCursor >= issueLimit) {
            stats.normalFetchDeferredByCarry++;
        }
        return false;
    }
    return issueCurrentBRow(static_cast<uint32_t>(rowIdx),
                            BProtectionClass::None);
}

void
MatrixFlowEngine::trySendMoreB()
{
    while (issueOneFetchB()) {
    }
}

void
MatrixFlowEngine::serviceParallelFetchAB(bool allowLaunchCompute)
{
    if (!parallelABSchedulerMode()) {
        return;
    }

    bool madeProgress = true;

    while (madeProgress) {
        madeProgress = false;
        if (dualRxSchedulerMode()) {
            bool aIssuedThisRound = false;
            bool bIssuedThisRound = false;
            const bool aPendingAtStart = canIssueA();
            const bool bPendingAtStart =
                hasFutureProtectedBPressure() ||
                hasCurrentProtectedBPressure() ||
                canIssueCompetitiveB();

            maybeSeedFutureProtectedB();

            if (hasFutureProtectedBPressure() ||
                (hasCurrentProtectedBPressure() &&
                 (currentProtectedBQuotaAvailable() ||
                  (!canIssueA() && !canIssueCompetitiveB())))) {
                if (issueOneDualRxB()) {
                    lastFetchIssueWasA = false;
                    bIssuedThisRound = true;
                    madeProgress = true;
                }
            }

            const bool compareAPending = canIssueA();
            const bool compareBPending = canIssueCompetitiveB();
            if (compareAPending || compareBPending) {
                const bool preferA = chooseDualRxIssueA();
                if (preferA) {
                    if (issueOneFetchA()) {
                        lastFetchIssueWasA = true;
                        aIssuedThisRound = true;
                        madeProgress = true;
                    } else if (issueOneCompetitiveB()) {
                        lastFetchIssueWasA = false;
                        bIssuedThisRound = true;
                        madeProgress = true;
                    }
                } else {
                    if (issueOneCompetitiveB()) {
                        lastFetchIssueWasA = false;
                        bIssuedThisRound = true;
                        madeProgress = true;
                    } else if (issueOneFetchA()) {
                        lastFetchIssueWasA = true;
                        aIssuedThisRound = true;
                        madeProgress = true;
                    }
                }
            }

            if (bIssuedThisRound && canIssueA() && !aIssuedThisRound) {
                if (issueOneFetchA()) {
                    lastFetchIssueWasA = true;
                    aIssuedThisRound = true;
                    madeProgress = true;
                }
            } else if (aIssuedThisRound &&
                       (hasFutureProtectedBPressure() ||
                        hasCurrentProtectedBPressure() ||
                        canIssueCompetitiveB()) &&
                       !bIssuedThisRound) {
                if (issueOneDualRxB()) {
                    lastFetchIssueWasA = false;
                    bIssuedThisRound = true;
                    madeProgress = true;
                }
            }

            if (aPendingAtStart && !aIssuedThisRound) {
                stats.rxAStallCycles++;
            }
            if (bPendingAtStart && !bIssuedThisRound) {
                stats.rxBStallCycles++;
            }
            currentProtectedBWinStreak = bIssuedThisRound ? currentProtectedBWinStreak : 0;
            updateABParallelOverlapTracking();
            continue;
        }

        if (hierarchicalProtectedBSchedulerMode()) {
            maybeSeedFutureProtectedB();

            if (hasFutureProtectedBPressure()) {
                if (issueOneFutureProtectedB()) {
                    lastFetchIssueWasA = false;
                    currentProtectedBWinStreak = 0;
                    madeProgress = true;
                    updateABParallelOverlapTracking();
                    continue;
                }
            } else if (hasCurrentProtectedBPressure()) {
                const bool mustServeCurrentProtected =
                    !canIssueA() && !canIssueCompetitiveB();
                if (currentProtectedBQuotaAvailable() ||
                    mustServeCurrentProtected) {
                    if (issueOneCurrentProtectedB()) {
                        lastFetchIssueWasA = false;
                        madeProgress = true;
                        updateABParallelOverlapTracking();
                        continue;
                    }
                } else {
                    stats.currentProtectedBQuotaExhaustCount++;
                }
            }

            if (claimBasedHoleFillingMode() &&
                nextClaimBlockedBRow() >= 0 &&
                !hasFutureProtectedBPressure() &&
                !hasCurrentProtectedBPressure()) {
                stats.futureClaimBlockedNormalFetchCount++;
            }

            const bool preferA = chooseHierarchicalIssueA();
            if (preferA) {
                if (issueOneFetchA()) {
                    lastFetchIssueWasA = true;
                    currentProtectedBWinStreak = 0;
                    madeProgress = true;
                } else if (issueOneCompetitiveB()) {
                    lastFetchIssueWasA = false;
                    currentProtectedBWinStreak = 0;
                    madeProgress = true;
                }
            } else {
                if (issueOneCompetitiveB()) {
                    lastFetchIssueWasA = false;
                    currentProtectedBWinStreak = 0;
                    madeProgress = true;
                } else if (issueOneFetchA()) {
                    lastFetchIssueWasA = true;
                    currentProtectedBWinStreak = 0;
                    madeProgress = true;
                }
            }
            updateABParallelOverlapTracking();
            continue;
        }

        if (protectedBSchedulerMode() && hasProtectedBPressure()) {
            if (issueOneProtectedB()) {
                lastFetchIssueWasA = false;
                currentProtectedBWinStreak = 0;
                madeProgress = true;
                updateABParallelOverlapTracking();
                continue;
            }
        }

        bool preferA = false;
        if (abSchedulerMode == ABSchedulerMode::FullScore) {
            preferA = chooseFullScoreIssueA();
        } else {
            preferA = chooseLightweightIssueA();
        }

        if (preferA) {
            if (issueOneFetchA()) {
                lastFetchIssueWasA = true;
                currentProtectedBWinStreak = 0;
                madeProgress = true;
            } else if (issueOneFetchB()) {
                lastFetchIssueWasA = false;
                currentProtectedBWinStreak = 0;
                madeProgress = true;
            }
        } else {
            if (issueOneFetchB()) {
                lastFetchIssueWasA = false;
                currentProtectedBWinStreak = 0;
                madeProgress = true;
            } else if (issueOneFetchA()) {
                lastFetchIssueWasA = true;
                currentProtectedBWinStreak = 0;
                madeProgress = true;
            }
        }
        updateABParallelOverlapTracking();
    }

    if (allowLaunchCompute) {
        tryLaunchComputeTile();
    }
}

void
MatrixFlowEngine::trySendMoreNextA()
{
    const Addr rowBytes =
        static_cast<Addr>(nextCtx.curTileK) * nextCtx.elemBytes;

    while (nextPrefetchReqsIssuedA < nextPrefetchTargetA &&
           satOutstanding(nextPrefetchReqsIssuedA,
                          nextPrefetchRowsACompleted) < kMaxInFlight) {
        uint32_t r = nextPrefetchReqsIssuedA;
        if (smartPatternPrefetchMode()) {
            const int smartRow = selectSmartNextPrefetchARow();
            if (smartRow < 0) {
                break;
            }
            r = static_cast<uint32_t>(smartRow);
        }
        if (r >= nextCtx.curTileM || nextPrefetchARowIssued[r]) {
            break;
        }
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
        nextPrefetchARowIssued[r] = true;
        if (smartPatternPrefetchMode() &&
            isSmartACandidateForContext(nextCtx, r)) {
            stats.smartPrefetchAIssueCount++;
            if (oracleARecurrenceBucket(rowAddr) ==
                OracleARecurrenceBucket::Repeat) {
                stats.smartPrefetchARepeatPriorityCount++;
            }
            const uint32_t futureReuse = futureJTileReuseCount(nextCtx);
            if (futureReuse > 1) {
                stats.smartPrefetchANearMultiPriorityCount++;
            }
        }
    }
}

void
MatrixFlowEngine::trySendMoreNextB()
{
    trySendMoreNextKBBudgeted();
}

void
MatrixFlowEngine::startCoverageGatherForCurrentTile()
{
    if (!coverage2DGatherMode() || !coverageShadowPoolEnabled()) {
        return;
    }
    if (coverageGatherValid && coverageGatherTargetB > 0) {
        return;
    }

    clearCoverageGatherForCurrentTile();

    GemmContext gather_ctx = ctx;
    if (coverage2DGatherPingPongFirstCutMode()) {
        if (!buildCoverageGatherFillContext(gather_ctx)) {
            return;
        }
        clearCoverageShadowFillBank();
        coverageShadowFillCtx = gather_ctx;
        coverageShadowFillValid = true;
    }

    coverageGatherValid = true;
    coverageGatherCtx = gather_ctx;
    if (coverage2DGatherMinGuaranteeFirstCutMode() &&
        coverageGatherMinIssueBudgetConfig > 0) {
        stats.gatherMinBudgetReservedCount++;
    }
    uint32_t target_rows = 0;
    const uint32_t current_rows_b = gather_ctx.curTileK;
    for (uint32_t row = 0; row < current_rows_b; ++row) {
        const bool target =
            coverage2DGatherPingPongFirstCutMode()
                ? isSmartCoverageTargetForContext(gather_ctx, row)
                : isCoverageBlindspotTargetPattern(
                      row, BProtectionClass::None,
                      gather_ctx.baseB +
                          ((static_cast<Addr>(gather_ctx.k + row) *
                            gather_ctx.size + gather_ctx.j) *
                           gather_ctx.elemBytes));
        if (target) {
            ++target_rows;
        }
    }
    coverageGatherTargetB = target_rows;
    stats.coverageGatherTargetRows += target_rows;
    if (target_rows == 0) {
        coverageGatherValid = false;
        return;
    }
    trySendMoreCoverageGatherB();
}

uint32_t
MatrixFlowEngine::sharedPrefetchBOutstanding() const
{
    const uint32_t next_k_outstanding =
        satOutstanding(nextPrefetchReqsIssuedB, nextPrefetchRowsBCompleted);
    const uint32_t next_output_outstanding =
        satOutstanding(nextOutputPrefetchReqsIssuedB,
                       nextOutputPrefetchRowsBCompleted);
    const uint32_t coverage_gather_outstanding =
        coverage2DGatherMode()
            ? satOutstanding(coverageGatherReqsIssuedB,
                             coverageGatherRowsBCompleted)
            : 0;
    return next_k_outstanding + next_output_outstanding +
           coverage_gather_outstanding;
}

uint32_t
MatrixFlowEngine::sharedPrefetchBCredits() const
{
    const uint32_t outstanding = sharedPrefetchBOutstanding();
    return outstanding >= kMaxInFlight ? 0 : (kMaxInFlight - outstanding);
}

bool
MatrixFlowEngine::issueCoverageGatherRow(uint32_t rowIdx,
                                         const GemmContext &gctx)
{
    if (rowIdx >= gctx.curTileK || coverageGatherBRowIssued[rowIdx]) {
        return false;
    }

    const Addr rowBytes =
        static_cast<Addr>(gctx.curTileN) * gctx.elemBytes;
    const Addr rowAddr = gctx.baseB +
        ((static_cast<Addr>(gctx.k + rowIdx) * gctx.size + gctx.j) *
         gctx.elemBytes);
    auto [reqAddr, reqBytes, reqOffset] = planReadRequest(rowAddr, rowBytes);
    uint8_t *dmaDst =
        coverageGatherBBounceBuffer.data() + rowIdx * readBouncePitch;
    if (reqBytes != rowBytes || reqOffset != 0) {
        coverageGatherBBounceActive[rowIdx] = true;
        coverageGatherBBounceReqAddr[rowIdx] = reqAddr;
        coverageGatherBBounceReqBytes[rowIdx] = reqBytes;
        coverageGatherBBounceRowBytes[rowIdx] = rowBytes;
        coverageGatherBBounceOffset[rowIdx] = reqOffset;
    } else {
        coverageGatherBBounceActive[rowIdx] = false;
    }

    coverageGatherBRowGeneration[rowIdx] = activeCoverageGatherGeneration;
    coverageGatherBRowIssued[rowIdx] = true;
    coverageGatherAgeScore[rowIdx] = 0;
    ++coverageGatherReqsIssuedB;
    if (coverage2DGatherNoStarvationV3Mode() ||
        coverage2DGatherMinGuaranteeFirstCutMode() ||
        coverage2DGatherPingPongFirstCutMode()) {
        ++coverageGatherBudgetUsedThisTile;
        ++coverageGatherConsecutiveIssueStreak;
    }
    if (coverage2DGatherMinGuaranteeFirstCutMode() &&
        coverageGatherMinIssueBudgetConfig > 0 &&
        coverageGatherMinBudgetGrantedThisTile <
            coverageGatherMinIssueBudgetConfig) {
        ++coverageGatherMinBudgetGrantedThisTile;
        ++stats.gatherBudgetGrantedCount;
    }
    ++stats.coverageGatherIssueCount;
    stats.totalDmaBytesRead += reqBytes;
    dmaPort.dmaAction(MemCmd::ReadReq, reqAddr, reqBytes,
                      &coverageGatherBRowEvents[rowIdx], dmaDst, 0);
    return true;
}

void
MatrixFlowEngine::trySendMoreCoverageGatherB()
{
    if (!coverage2DGatherMode() || !coverageGatherValid ||
        !coverageShadowPoolEnabled()) {
        return;
    }

    const GemmContext &gctx =
        coverage2DGatherPingPongFirstCutMode() ? coverageGatherCtx : ctx;
    const uint32_t current_rows_b = gctx.curTileK;
    const uint32_t outstanding_limit =
        (coverage2DGatherNoStarvationV3Mode() ||
         coverage2DGatherMinGuaranteeFirstCutMode() ||
         coverage2DGatherPingPongFirstCutMode()) ? 4u :
        (coverage2DGatherV2Mode() ? 8u : 2u);
    uint32_t issued_this_call = 0;
    while (coverageGatherReqsIssuedB < coverageGatherTargetB &&
           sharedPrefetchBCredits() > 0 &&
           satOutstanding(coverageGatherReqsIssuedB,
                          coverageGatherRowsBCompleted) < outstanding_limit) {
        if (coverage2DGatherNoStarvationV3Mode() ||
            coverage2DGatherPingPongFirstCutMode()) {
            if (coverageGatherBudgetUsedThisTile >=
                kCoverageGatherNoStarvationBudgetRows) {
                ++stats.gatherBudgetExhaustedCount;
                ageUpCoverageGatherCandidates();
                break;
            }
            if (coverageGatherConsecutiveIssueStreak >=
                kCoverageGatherNoStarvationMaxConsecutiveIssues ||
                issued_this_call >=
                    kCoverageGatherNoStarvationMaxConsecutiveIssues) {
                ++stats.gatherMaxConsecutiveIssueHitsCount;
                ageUpCoverageGatherCandidates();
                break;
            }
        }
        if (coverage2DGatherMinGuaranteeFirstCutMode()) {
            if (coverageGatherConsecutiveIssueStreak >=
                kCoverageGatherNoStarvationMaxConsecutiveIssues ||
                issued_this_call >=
                    kCoverageGatherNoStarvationMaxConsecutiveIssues) {
                ++stats.gatherMaxConsecutiveIssueHitsCount;
                ageUpCoverageGatherCandidates();
                break;
            }
        }

        int selected_row = -1;
        int best_score = std::numeric_limits<int>::min();
        uint8_t selected_age = 0;
        for (uint32_t row = 0; row < current_rows_b; ++row) {
            if (coverageGatherBRowIssued[row]) {
                continue;
            }
            const Addr rowAddr = gctx.baseB +
                ((static_cast<Addr>(gctx.k + row) * gctx.size + gctx.j) *
                 gctx.elemBytes);
            const bool target =
                coverage2DGatherPingPongFirstCutMode()
                    ? isSmartCoverageTargetForContext(gctx, row)
                    : isCoverageBlindspotTargetPattern(
                          row, BProtectionClass::None, rowAddr);
            if (!target) {
                continue;
            }
            if (coverage2DGatherMinGuaranteeFirstCutMode()) {
                int score = 0;
                if (currentBRowState[row] == BRowState::Empty) {
                    score += 1000;
                } else if (currentBRowState[row] == BRowState::InflightFromNormal ||
                           currentBRowState[row] == BRowState::ReadyFromNormal) {
                    score += 100;
                }
                score += static_cast<int>(coverageGatherAgeScore[row]);
                score += static_cast<int>(current_rows_b - row);
                if (score > best_score) {
                    best_score = score;
                    selected_row = static_cast<int>(row);
                }
                continue;
            }
            if (!(coverage2DGatherNoStarvationV3Mode() ||
                  coverage2DGatherPingPongFirstCutMode())) {
                selected_row = static_cast<int>(row);
                break;
            }
            const uint8_t age = coverageGatherAgeScore[row];
            if (selected_row < 0 || age > selected_age) {
                selected_row = static_cast<int>(row);
                selected_age = age;
            }
        }
        if (selected_row < 0) {
            break;
        }

        const uint32_t row = static_cast<uint32_t>(selected_row);
        if (!issueCoverageGatherRow(row, gctx)) {
            break;
        }
        ++issued_this_call;
    }
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
        uint32_t r = nextPrefetchReqsIssuedB;
        if (smartPatternPrefetchMode()) {
            const int smartRow = selectSmartNextPrefetchBRow();
            if (smartRow < 0) {
                break;
            }
            r = static_cast<uint32_t>(smartRow);
        }
        if (r >= nextCtx.curTileK || nextPrefetchBRowIssued[r]) {
            break;
        }
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
        nextPrefetchBRowIssued[r] = true;
        stats.nextKPrefetchRowsIssued++;
        if (smartPatternPrefetchMode()) {
            stats.smartPrefetchBNextKPriorityIssueCount++;
        }
        if (phase == Phase::WriteC && writeCOverlapWindowActive) {
            ++writeCBRowsIssued;
            stats.bRowsIssuedDuringWriteC++;
        }
    }
}

void
MatrixFlowEngine::trySendMoreNextOutputB()
{
    if (protectedBSchedulerMode() ||
        hierarchicalProtectedBSchedulerMode()) {
        while (issueOneNextOutputProtectedB()) {
        }
        return;
    }

    const Addr rowBytes =
        static_cast<Addr>(nextOutputCtx.curTileN) * nextOutputCtx.elemBytes;

    while (nextOutputPrefetchReqsIssuedB < nextOutputPrefetchTargetB) {
        if (phase == Phase::WriteC && writeCOverlapWindowActive &&
            writeCBRowsIssued >= writeCOverlapBIssueBudgetRowsConfig) {
            break;
        }
        if (nextOutputFirstIssueTick == 0) {
            nextOutputFirstIssueTick = curTick();
        }
        uint32_t r = nextOutputPrefetchReqsIssuedB;
        if (smartPatternPrefetchMode()) {
            const int smartRow = selectSmartNextOutputPrefetchBRow();
            if (smartRow < 0) {
                break;
            }
            r = static_cast<uint32_t>(smartRow);
        }
        if (r >= nextOutputCtx.curTileK || nextOutputPrefetchBRowIssued[r]) {
            break;
        }
        if (mhotAllowsMainlineBypass() &&
            tryServeNextOutputRowFromMHot(r, BProtectionClass::NextOutput)) {
            nextOutputPrefetchBRowIssued[r] = true;
            if (smartPatternPrefetchMode()) {
                stats.smartPrefetchBNextOutputPriorityIssueCount++;
                if (isSmartCoverageTargetForContext(nextOutputCtx, r)) {
                    stats.smartPrefetchBNextOutputCoverageTargetIssueCount++;
                }
            }
            continue;
        }
        if (sharedPrefetchBCredits() == 0) {
            break;
        }
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
        nextOutputPrefetchBRowIssued[r] = true;
        stats.nextOutputPrefetchRowsIssued++;
        if (smartPatternPrefetchMode()) {
            stats.smartPrefetchBNextOutputPriorityIssueCount++;
            if (isSmartCoverageTargetForContext(nextOutputCtx, r)) {
                stats.smartPrefetchBNextOutputCoverageTargetIssueCount++;
            }
        }
        setNextOutputFutureClaim(r, BProtectionClass::NextOutput);
        if (phase == Phase::WriteC && writeCOverlapWindowActive) {
            ++writeCBRowsIssued;
            stats.bRowsIssuedDuringWriteC++;
        }
    }
}

void
MatrixFlowEngine::arbitratePrefetchBIssues()
{
    if (coverage2DGatherNoStarvationV3Mode() ||
        coverage2DGatherMinGuaranteeFirstCutMode()) {
        coverageGatherConsecutiveIssueStreak = 0;
    }

    if (phase == Phase::WriteC && writeCOverlapWindowActive) {
        const bool next_output_needs_new_rows =
            nextOutputPrefetchValid &&
            !nextOutputPrefetchReadyB &&
            nextOutputPrefetchReqsIssuedB < nextOutputPrefetchTargetB;
        const bool next_output_needs_remote_rows =
            next_output_needs_new_rows && !nextOutputPendingRowCanUseMHot();
        const bool next_k_needs_new_rows =
            nextPrefetchValid &&
            !nextPrefetchIsOutputTile &&
            !nextPrefetchReadyB &&
            nextPrefetchReqsIssuedB < nextPrefetchTargetB;
        const bool budget_exhausted =
            writeCOverlapBIssueBudgetRowsConfig == 0 ||
            writeCBRowsIssued >= writeCOverlapBIssueBudgetRowsConfig;

        if (budget_exhausted &&
            (next_output_needs_remote_rows || next_k_needs_new_rows)) {
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
    const bool gather_pending =
        coverage2DGatherMode() && coverageGatherValid &&
        coverageGatherReqsIssuedB < coverageGatherTargetB;

    if (next_output_needs_rows) {
        if (sharedPrefetchBCredits() == 0 &&
            !nextOutputPendingRowCanUseMHot()) {
            stats.nextOutputPrefetchDeferCount++;
            return;
        }
        const uint32_t issued_before = nextOutputPrefetchReqsIssuedB;
        trySendMoreNextOutputB();
        if (nextOutputPrefetchReqsIssuedB > issued_before) {
            coverageGatherConsecutiveIssueStreak = 0;
        }
    }

    const bool next_k_needs_rows =
        nextPrefetchValid &&
        !nextPrefetchIsOutputTile &&
        !nextPrefetchReadyB &&
        nextPrefetchReqsIssuedB < nextPrefetchTargetB;

    if (sharedPrefetchBCredits() > 0 && next_k_needs_rows) {
        const uint32_t issued_before = nextPrefetchReqsIssuedB;
        trySendMoreNextKBBudgeted();
        if (nextPrefetchReqsIssuedB > issued_before) {
            coverageGatherConsecutiveIssueStreak = 0;
        }
    }

    if (coverage2DGatherMinGuaranteeFirstCutMode() && gather_pending &&
        coverageGatherMinIssueBudgetConfig > 0 &&
        coverageGatherMinBudgetGrantedThisTile <
            coverageGatherMinIssueBudgetConfig &&
        sharedPrefetchBCredits() > 0) {
        trySendMoreCoverageGatherB();
    }

    if ((coverage2DGatherNoStarvationV3Mode() ||
         coverage2DGatherMinGuaranteeFirstCutMode() ||
         coverage2DGatherPingPongFirstCutMode()) && gather_pending) {
        const bool mainline_grace = coverageGatherMainlineGraceActive();
        const bool mainline_still_needs_rows =
            (nextOutputPrefetchValid &&
             !nextOutputPrefetchReadyB &&
             nextOutputPrefetchReqsIssuedB < nextOutputPrefetchTargetB) ||
            (nextPrefetchValid &&
             !nextPrefetchIsOutputTile &&
             !nextPrefetchReadyB &&
             nextPrefetchReqsIssuedB < nextPrefetchTargetB);

        if (mainline_grace) {
            ++stats.gatherBlockedByMainlineGraceWindowCount;
            ++stats.gatherDeferredByMainlineCount;
            ++stats.mainlineCreditReservedCount;
            ageUpCoverageGatherCandidates();
            return;
        }
        if (mainline_still_needs_rows) {
            ++stats.gatherDeferredByMainlineCount;
            ++stats.mainlineCreditReservedCount;
            ageUpCoverageGatherCandidates();
            return;
        }
        if (sharedPrefetchBCredits() > 0) {
            trySendMoreCoverageGatherB();
        }
        return;
    }

    if (coverage2DGatherV2Mode() && sharedPrefetchBCredits() > 0) {
        trySendMoreCoverageGatherB();
    }
    if (coverage2DGatherMode() &&
        !(coverage2DGatherV2Mode() || coverage2DGatherNoStarvationV3Mode()) &&
        sharedPrefetchBCredits() > 0) {
        trySendMoreCoverageGatherB();
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
MatrixFlowEngine::updateABParallelOverlapTracking()
{
    const uint32_t aInflight = satOutstanding(reqsIssuedA, reqsCompletedA);
    const uint32_t bInflight = satOutstanding(reqsIssuedB, reqsCompletedB);
    updateDualRxQueueStats();

    if (aInflight > aInflightPeakObserved) {
        aInflightPeakObserved = aInflight;
        stats.aRowsInflightPeak = aInflightPeakObserved;
    }
    if (bInflight > bInflightPeakObserved) {
        bInflightPeakObserved = bInflight;
        stats.bRowsInflightPeak = bInflightPeakObserved;
    }

    if (!parallelABSchedulerMode()) {
        closeABParallelOverlapTracking();
        return;
    }

    const bool overlapNow = aInflight > 0 && bInflight > 0;
    if (overlapNow && !abParallelOverlapActive) {
        abParallelOverlapActive = true;
        abParallelOverlapStartTick = curTick();
    } else if (!overlapNow && abParallelOverlapActive) {
        if (curTick() >= abParallelOverlapStartTick) {
            stats.abParallelFetchOverlapCycles +=
                (curTick() - abParallelOverlapStartTick) / clockPeriod();
        }
        abParallelOverlapActive = false;
        abParallelOverlapStartTick = 0;
    }
}

void
MatrixFlowEngine::closeABParallelOverlapTracking()
{
    if (!abParallelOverlapActive) {
        return;
    }

    if (curTick() >= abParallelOverlapStartTick) {
        stats.abParallelFetchOverlapCycles +=
            (curTick() - abParallelOverlapStartTick) / clockPeriod();
    }
    abParallelOverlapActive = false;
    abParallelOverlapStartTick = 0;
}

void
MatrixFlowEngine::tryLaunchComputeTile()
{
    if (computeDoneEvent.scheduled() || !inputsReadyForCompute()) {
        return;
    }

    stats.aRowsReadyBeforeCompute += reqsCompletedA;
    closeABParallelOverlapTracking();
    launchComputeTile();
}

void
MatrixFlowEngine::issueWriteCTile()
{
    phase = Phase::WriteC;
    if (hierarchicalProtectedBSchedulerMode()) {
        warn("%s: hier-trace issueWriteC tile i=%u j=%u k=%u dims=(%u,%u,%u)\n",
             name(), ctx.i, ctx.j, ctx.k,
             ctx.curTileM, ctx.curTileN, ctx.curTileK);
    }

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
    if (hierarchicalProtectedBSchedulerMode()) {
        warn("%s: hier-trace issueWriteFlag flag=%#llx value=%llu\n",
             name(), static_cast<unsigned long long>(ctx.flagAddr),
             static_cast<unsigned long long>(completionFlagValue));
    }

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
    stats.rxAReadyCount++;
    if (hierarchicalProtectedBSchedulerMode() &&
        reqsCompletedA == targetReqsA) {
        warn("%s: hier-trace current tile A ready %u/%u at i=%u j=%u k=%u\n",
             name(), reqsCompletedA, targetReqsA, ctx.i, ctx.j, ctx.k);
    }
    if (reqsCompletedB < targetReqsB) {
        stats.aFetchProgressDuringBFetch++;
    }
    updateABParallelOverlapTracking();
    const Addr oracleRowAddr = ctx.baseA +
        ((static_cast<Addr>(ctx.i + rowIdx) * ctx.size + ctx.k) *
         ctx.elemBytes);
    recordCurrentAOracleOutcome(oracleRowAddr, false);

    if (shouldAdmitVipRescueARow(rowIdx)) {
        const Addr rowBytes =
            static_cast<Addr>(ctx.curTileK) * ctx.elemBytes;
        const Addr rowAddr = ctx.baseA +
            ((static_cast<Addr>(ctx.i + rowIdx) * ctx.size + ctx.k) *
             ctx.elemBytes);
        const uint32_t future_reuse = rescueFutureJTileReuseCount();
        const bool short_next_use = rescueANextUseIsShort();
        const VipAdmitClass admit =
            (short_next_use && future_reuse > 1)
                ? VipAdmitClass::Strong
                : VipAdmitClass::Weak;
        if (tryInsertVipRow(rowAddr, rowBytes,
                            tileABuffer.data() + rowIdx * rowBytes,
                            VipSourceClass::RescueA,
                            BProtectionClass::None, admit)) {
            if (short_next_use) {
                stats.vipRescueAInsertShortNextUseCount++;
            }
            if (future_reuse > 1) {
                stats.vipRescueAInsertMultiFutureUseCount++;
            }
        }
    }

    if (reqsCompletedA == targetReqsA &&
        nextPrefetchTrigger == "a_ready") {
        maybePrefetchNextTile();
        maybePrefetchNextOutputTile();
    }
    if (baselineSchedulerMode()) {
        if (reqsCompletedA < targetReqsA) {
            trySendMoreA();
        } else {
            issueFetchBTile();
        }
        return;
    }

    serviceParallelFetchAB();
    if (reqsCompletedA == targetReqsA &&
        reqsCompletedB < targetReqsB &&
        !aWaitedForBThisTile) {
        stats.aPathStallWaitingForB++;
        aWaitedForBThisTile = true;
    }
    tryLaunchComputeTile();
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
    const bool hadFutureClaim =
        claimBasedHoleFillingMode() && currentBFutureClaimed[rowIdx];
    clearCurrentFutureClaimRow(rowIdx, false, false, true);
    if (hadFutureClaim) {
        currentBProtectionClass[rowIdx] = BProtectionClass::None;
    }
    recordProtectedBReady(currentBProtectionClass[rowIdx]);
    ++reqsCompletedB;
    stats.rxBReadyCount++;
    if (vipRescueMode()) {
        const Addr rowBytes =
            static_cast<Addr>(ctx.curTileN) * ctx.elemBytes;
        const Addr rowAddr = ctx.baseB +
            ((static_cast<Addr>(ctx.k + rowIdx) * ctx.size + ctx.j) *
             ctx.elemBytes);
        const uint32_t repeat_count = ++bFallbackVictimCount[rowAddr];
        const uint32_t future_reuse = rescueFutureITileReuseCount();
        const bool short_next_use = rescueNextUseIsShort();
        const bool critical = criticalRowOrWindow(rowIdx);
        const bool rescue_near = nextRescueOpportunityNear(rowIdx);
        const bool rescue_high =
            (vipRescueRecurrenceAwareV2Mode() ||
             vipRescueRecurrenceAwareV3Mode())
                ? rescueCriticalityHigh(rowIdx, rowAddr)
                : critical;
        if (repeat_count >= 2) {
            stats.repeatVictimCountTotal++;
        }
        if (critical) {
            stats.criticalWindowVictimCount++;
        }
        if (rescue_near) {
            stats.nextRescueOpportunityNearCount++;
        }
        if (rescue_high) {
            stats.rescueCriticalityHighCount++;
        }

        VipAdmitClass admit = VipAdmitClass::Reject;
        if (vipRescueRecurrenceAwareV2Mode() ||
            vipRescueRecurrenceAwareV3Mode()) {
            const auto src = currentBOracleSource[rowIdx];
            const auto dist = oracleDistanceBucket(rowIdx, targetReqsB);
            const uint32_t tile_steps = rescueTileStepsToNextUse();
            const bool long_span = tile_steps > 4;
            const bool weak_admit =
                vipRescueRecurrenceAwareV3Mode()
                    ? (long_span ? (future_reuse > 1 && rescue_near)
                                 : (future_reuse > 0 || rescue_near))
                    : (future_reuse > 0 || rescue_near);
            const bool first_seen_eligible =
                repeat_count == 1 && short_next_use &&
                weak_admit;
            if (src == OracleSourceClass::NextOutput &&
                (dist == OracleDistanceBucket::Immediate ||
                 dist == OracleDistanceBucket::Near)) {
                stats.vipRescueRejectNonFallbackCount++;
            } else if (future_reuse == 0 && !rescue_near) {
                stats.vipRescueRejectSingleUseCount++;
            } else if (!short_next_use) {
                stats.vipRescueRejectNotShortUseCount++;
            } else {
                if (first_seen_eligible) {
                    stats.eligibleRescueVictimsFirstSeenCount++;
                }
                if (!rescue_high) {
                    stats.vipRescueRejectNotRescueCriticalCount++;
                } else if (repeat_count >= 2) {
                    admit = VipAdmitClass::Strong;
                    stats.strongRescuePromotedOnRecurrenceCount++;
                } else if (weak_admit) {
                    admit = VipAdmitClass::Weak;
                    stats.weakRescuePromotedOnFirstFallbackCount++;
                    stats.eligibleRescueVictimsPromotedEarlyCount++;
                } else {
                    stats.vipRescueRejectNonRepeatCount++;
                }
            }
        } else if (vipRescueAntiDeadBlockMode()) {
            const auto src = currentBOracleSource[rowIdx];
            const auto dist = oracleDistanceBucket(rowIdx, targetReqsB);
            if (src == OracleSourceClass::NextOutput &&
                (dist == OracleDistanceBucket::Immediate ||
                 dist == OracleDistanceBucket::Near)) {
                stats.vipRescueRejectNonFallbackCount++;
            } else if (future_reuse == 0) {
                stats.vipRescueRejectSingleUseCount++;
            } else if (!short_next_use) {
                stats.vipRescueRejectNotShortUseCount++;
            } else {
                admit = classifyVipRescueAdmission(rowIdx, rowAddr);
                if (admit == VipAdmitClass::Reject) {
                    if (future_reuse > 1 || repeat_count >= 2) {
                        stats.vipRescueRejectNotCriticalCount++;
                    } else {
                        stats.vipRescueRejectNonRepeatCount++;
                    }
                }
            }
        } else if (shouldAdmitVipRescueRow(rowIdx)) {
            admit = (short_next_use && future_reuse > 1)
                ? VipAdmitClass::Strong
                : VipAdmitClass::Weak;
        }

        if (admit != VipAdmitClass::Reject &&
            tryInsertVipRow(rowAddr, rowBytes,
                            tileBBuffer.data() + rowIdx * rowBytes,
                            VipSourceClass::Rescue,
                            BProtectionClass::None, admit,
                            repeat_count, rescue_high,
                            short_next_use, rescue_near,
                            future_reuse)) {
            stats.vipRescueInsertAfterFallbackCount++;
            if (short_next_use) {
                stats.vipRescueInsertShortNextUseCount++;
            }
            if (future_reuse > 1) {
                stats.vipRescueInsertMultiFutureUseCount++;
            }
        }
    }
    if (mhotGapAwareNextCutMode() || coverageShadowMode() ||
        mhotCoverageBlindspotCandidateMode()) {
        const Addr rowBytes =
            static_cast<Addr>(ctx.curTileN) * ctx.elemBytes;
        const Addr rowAddr = ctx.baseB +
            ((static_cast<Addr>(ctx.k + rowIdx) * ctx.size + ctx.j) *
             ctx.elemBytes);
        OracleSourceClass hotOracleSrc = currentBOracleSource[rowIdx];
        if (hotOracleSrc == OracleSourceClass::Normal) {
            hotOracleSrc = oracleSourceFromProtectionClass(
                currentBProtectionClass[rowIdx]);
        }

        VipSourceClass hotSrc = VipSourceClass::None;
        switch (hotOracleSrc) {
          case OracleSourceClass::CurrentWindow:
            hotSrc = VipSourceClass::CurrentWindow;
            break;
          case OracleSourceClass::Normal:
            hotSrc = VipSourceClass::Normal;
            break;
          default:
            hotSrc = VipSourceClass::None;
            break;
        }

        const auto admit = classifyMHotAdmission(
            rowIdx, targetReqsB, hotSrc, currentBProtectionClass[rowIdx]);
        if (hotSrc != VipSourceClass::None &&
            admit != VipAdmitClass::Reject) {
            tryInsertMHotRow(rowAddr, rowBytes,
                             tileBBuffer.data() + rowIdx * rowBytes,
                             hotSrc, currentBProtectionClass[rowIdx],
                             admit,
                             oracleDistanceBucket(rowIdx, targetReqsB));
        }
    }
    if (mhotCoverageBlindspotCandidateMode()) {
        const Addr rowBytes =
            static_cast<Addr>(ctx.curTileN) * ctx.elemBytes;
        const Addr rowAddr = ctx.baseB +
            ((static_cast<Addr>(ctx.k + rowIdx) * ctx.size + ctx.j) *
             ctx.elemBytes);
        if (isCoverageBlindspotNormalCandidate(
                rowIdx, currentBProtectionClass[rowIdx], rowAddr)) {
            const uint64_t patternKey =
                coverageBlindspotPatternKey(rowIdx,
                                            currentBProtectionClass[rowIdx],
                                            rowAddr);
            auto &meta = bCoverageBlindspotPattern[patternKey];
            const uint32_t future_reuse = rescueFutureITileReuseCount();
            const uint32_t tile_steps = rescueTileStepsToNextUse();
            const auto dist = oracleDistanceBucket(rowIdx, targetReqsB);
            const auto reuse_bucket =
                coverageBlindspotFutureReuseBucket(future_reuse);
            const auto tile_band =
                coverageBlindspotTileStepBand(tile_steps);
            const bool short_next_use = tile_steps <= 12;
            const bool narrow_pattern =
                isCoverageBlindspotTargetPattern(
                    rowIdx, currentBProtectionClass[rowIdx], rowAddr);
            const Tick short_window_ticks = std::max<Tick>(
                clockPeriod(),
                static_cast<Tick>(estimateTileCycles(
                    ctx.curTileM, ctx.curTileN, ctx.curTileK)) *
                    clockPeriod() * 2);
            const Tick promote_cooldown_ticks =
                std::max<Tick>(short_window_ticks, clockPeriod() * 4);
            const bool short_window_repeat =
                meta.lastFallbackTick != 0 &&
                curTick() >= meta.lastFallbackTick &&
                (curTick() - meta.lastFallbackTick) <= short_window_ticks;
            const bool promote_cooldown_elapsed =
                meta.lastPromoteTick == 0 ||
                (curTick() >= meta.lastPromoteTick &&
                 (curTick() - meta.lastPromoteTick) >=
                     promote_cooldown_ticks);

            stats.mhotCoverageBlindspotSeenCount++;
            stats.coverageBlindspotSeenByDistance[static_cast<int>(dist)] += 1;
            stats.coverageBlindspotSeenByReuseBucket[
                static_cast<int>(reuse_bucket)] += 1;
            stats.coverageBlindspotSeenByTileBand[
                static_cast<int>(tile_band)] += 1;
            if (future_reuse > 1) {
                stats.coverageBlindspotFutureReuseGt1Count++;
            }
            if (short_next_use) {
                stats.coverageBlindspotShortNextUseCount++;
            }

            meta.seenCount += 1;
            meta.lastFallbackTick = curTick();

            VipAdmitClass admit = VipAdmitClass::Reject;
            if (mhotCoverageBlindspotCandidateV2Mode()) {
                if (!narrow_pattern) {
                    if (tile_band != 1) {
                        stats.mhotCoverageBlindspotRejectLongDistanceCount++;
                    }
                    if (reuse_bucket != 2) {
                        stats.mhotCoverageBlindspotRejectLowReuseCount++;
                    }
                } else if (meta.seenCount < 2) {
                    stats.mhotCoverageBlindspotRejectSingleCount++;
                } else {
                    stats.mhotCoverageBlindspotRepeatCount++;
                    stats.repeatCoverageVictimCountTotal++;
                    if (short_window_repeat && promote_cooldown_elapsed) {
                        admit = VipAdmitClass::Strong;
                    } else if (!short_window_repeat) {
                        stats.mhotCoverageBlindspotRejectLongDistanceCount++;
                    }
                }
            } else if (meta.seenCount == 1) {
                stats.mhotCoverageBlindspotRejectSingleCount++;
                if (!short_next_use) {
                    stats.mhotCoverageBlindspotRejectLongDistanceCount++;
                }
                if (future_reuse <= 1 && !short_window_repeat) {
                    stats.mhotCoverageBlindspotRejectLowReuseCount++;
                }
            } else {
                stats.mhotCoverageBlindspotRepeatCount++;
                stats.repeatCoverageVictimCountTotal++;
                admit =
                    (short_next_use || future_reuse > 1 || short_window_repeat)
                        ? VipAdmitClass::Strong
                        : VipAdmitClass::Weak;
            }

            if (admit != VipAdmitClass::Reject) {
                meta.promoted = true;
                meta.lastPromoteTick = curTick();
                stats.mhotCoverageBlindspotPromotedCount++;
                stats.repeatCoverageVictimPromotedCount++;
                stats.coverageBlindspotPromotedByDistance[
                    static_cast<int>(dist)] += 1;
                stats.coverageBlindspotPromotedByReuseBucket[
                    static_cast<int>(reuse_bucket)] += 1;
                stats.coverageBlindspotPromotedByTileBand[
                    static_cast<int>(tile_band)] += 1;
                tryInsertMHotRow(rowAddr, rowBytes,
                                 tileBBuffer.data() + rowIdx * rowBytes,
                                 VipSourceClass::Normal,
                                 currentBProtectionClass[rowIdx], admit,
                                 dist,
                                 true);
            }
        }
    }
    if (hierarchicalProtectedBSchedulerMode() &&
        reqsCompletedB == targetReqsB) {
        warn("%s: hier-trace current tile B ready %u/%u at i=%u j=%u k=%u\n",
             name(), reqsCompletedB, targetReqsB, ctx.i, ctx.j, ctx.k);
    }
    if (reqsCompletedA < targetReqsA) {
        stats.bFetchProgressDuringAFetch++;
    }
    recordCurrentBOracleOutcome(rowIdx, false);
    updateABParallelOverlapTracking();

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
    if (baselineSchedulerMode()) {
        trySendMoreB();
        tryLaunchComputeTile();
        return;
    }

    serviceParallelFetchAB();
    if (reqsCompletedB == targetReqsB &&
        reqsCompletedA < targetReqsA &&
        !bWaitedForAThisTile) {
        stats.bPathStallWaitingForA++;
        bWaitedForAThisTile = true;
    }
    tryLaunchComputeTile();
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
    if (coverageShadowUsesPrefetchFeed() &&
        isSmartCoverageTargetForContext(nextCtx, rowIdx)) {
        const Addr rowBytes =
            static_cast<Addr>(nextCtx.curTileN) * nextCtx.elemBytes;
        const Addr rowAddr = nextCtx.baseB +
            ((static_cast<Addr>(nextCtx.k + rowIdx) * nextCtx.size +
              nextCtx.j) * nextCtx.elemBytes);
        tryInsertCoverageShadowRow(
            rowAddr, rowBytes,
            nextTileBBuffer.data() + rowIdx * rowBytes,
            CoverageShadowInsertSource::NextK);
    }
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
        const Addr rowBytes =
            static_cast<Addr>(ctx.curTileN) * ctx.elemBytes;
        const Addr rowAddr = ctx.baseB +
            ((static_cast<Addr>(ctx.k + rowIdx) * ctx.size + ctx.j) *
             ctx.elemBytes);
        const OracleDistanceBucket dist =
            oracleDistanceBucket(rowIdx, targetReqsB);
        bool servedFromVip = false;
        const VipAdmitClass mhotAdmit =
            classifyMHotAdmission(rowIdx, targetReqsB,
                                  VipSourceClass::CarryOver,
                                  BProtectionClass::CarryOver);
        if (mhotAdmit != VipAdmitClass::Reject) {
            tryInsertMHotRow(rowAddr, rowBytes,
                             tileBBuffer.data() + rowIdx * rowBytes,
                             VipSourceClass::CarryOver,
                             BProtectionClass::CarryOver,
                             mhotAdmit, dist);
        }
        const VipAdmitClass carryAdmit =
            classifyVipAdmission(rowIdx, targetReqsB,
                                 VipSourceClass::CarryOver,
                                 BProtectionClass::CarryOver);
        if (carryAdmit != VipAdmitClass::Reject &&
            oracleGuidedVipMode() &&
            !currentBOracleVipSelected[rowIdx]) {
            currentBOracleVipSelected[rowIdx] = true;
            stats.oracleSelectedBRowsCount++;
        }
        if (carryAdmit != VipAdmitClass::Reject &&
            tryInsertVipRow(rowAddr, rowBytes,
                            tileBBuffer.data() + rowIdx * rowBytes,
                            VipSourceClass::CarryOver,
                            BProtectionClass::CarryOver,
                            carryAdmit)) {
            recordVipOracleGuidedAdmit(
                rowIdx, targetReqsB, VipSourceClass::CarryOver);
            currentBRowState[rowIdx] = BRowState::Empty;
            currentBProtectionClass[rowIdx] =
                BProtectionClass::CarryOver;
            currentBVipBacked[rowIdx] = true;
            currentBVipSource[rowIdx] = VipSourceClass::CarryOver;
            currentBVipAdmitClass[rowIdx] = carryAdmit;
            currentBVipPrefetchedCounted[rowIdx] = false;
            servedFromVip = tryServeCurrentBRowFromVip(rowIdx, false);
        }
        if (!servedFromVip) {
            currentBRowState[rowIdx] = BRowState::ReadyFromNextOutput;
            setCurrentOracleSource(rowIdx, OracleSourceClass::CarryOver);
            recordProtectedBReady(BProtectionClass::CarryOver);
            stats.carryOverRowsConsumedPostBoundary++;
            recordOracleSelectedRowMiss(rowIdx);
        }
        if (phase == Phase::WriteC && writeCOverlapWindowActive) {
            writeCOverlapMadeProgress = true;
            stats.writeCOverlapSuccessCount++;
            stats.nextOutputProgressDuringWriteC++;
        }
        if (!servedFromVip) {
            panic_if(
                reqsCompletedB >= targetReqsB,
                "%s: carry-over completion reqsCompletedB=%u >= "
                "targetReqsB=%u\n",
                name(), reqsCompletedB, targetReqsB);
            ++reqsCompletedB;
            stats.rxBReadyCount++;
        }
        recordCurrentBOracleOutcome(rowIdx, true);
        if (carryOverBRowsCompleted == carryOverBRowsIssued) {
            carryOverBActive = false;
            carryOverBGeneration = 0;
            carryOverBRowsIssued = 0;
            carryOverBRowsCompleted = 0;
            carryOverBRowsTarget = 0;
        }
        if (reqsCompletedA < targetReqsA) {
            stats.bFetchProgressDuringAFetch++;
        }
        updateABParallelOverlapTracking();
        if (baselineSchedulerMode()) {
            trySendMoreB();
            tryLaunchComputeTile();
            return;
        }
        serviceParallelFetchAB();
        if (reqsCompletedB == targetReqsB &&
            reqsCompletedA < targetReqsA &&
            !bWaitedForAThisTile) {
            stats.bPathStallWaitingForA++;
            bWaitedForAThisTile = true;
        }
        tryLaunchComputeTile();
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

    if (claimBasedHoleFillingMode() &&
        currentBFutureClaimGeneration != 0 &&
        rowGen == currentBFutureClaimGeneration &&
        rowIdx < targetReqsB &&
        currentBFutureClaimed[rowIdx]) {
        if (nextOutputFetchBBounceActive[rowIdx]) {
            uint8_t *src = nextOutputFetchBBounceBuffer.data() +
                           rowIdx * readBouncePitch +
                           nextOutputFetchBBounceOffset[rowIdx];
            uint8_t *dst = tileBBuffer.data() +
                           rowIdx * nextOutputFetchBBounceRowBytes[rowIdx];
            std::memcpy(dst, src, nextOutputFetchBBounceRowBytes[rowIdx]);
            nextOutputFetchBBounceActive[rowIdx] = false;
        }

        if (currentBRowState[rowIdx] != BRowState::Empty) {
            clearCurrentFutureClaimRow(rowIdx, false, false, true);
            stats.nextPrefetchLateCompletionCount++;
            stats.nextOutputPrefetchLateCompletionCount++;
            arbitratePrefetchBIssues();
            return;
        }

        panic_if(reqsCompletedB >= targetReqsB,
                 "%s: claimed future completion reqsCompletedB=%u >= "
                 "targetReqsB=%u\n",
                 name(), reqsCompletedB, targetReqsB);
        const BProtectionClass claimCls = currentBFutureClaimClass[rowIdx];
        const Addr rowBytes =
            static_cast<Addr>(ctx.curTileN) * ctx.elemBytes;
        const Addr rowAddr = ctx.baseB +
            ((static_cast<Addr>(ctx.k + rowIdx) * ctx.size + ctx.j) *
             ctx.elemBytes);
        const OracleDistanceBucket dist =
            oracleDistanceBucket(rowIdx, targetReqsB);
        bool servedFromVip = false;
        const VipAdmitClass mhotAdmit =
            classifyMHotAdmission(rowIdx, targetReqsB,
                                  VipSourceClass::Claim, claimCls);
        if (mhotAdmit != VipAdmitClass::Reject) {
            tryInsertMHotRow(rowAddr, rowBytes,
                             tileBBuffer.data() + rowIdx * rowBytes,
                             VipSourceClass::Claim, claimCls,
                             mhotAdmit, dist);
        }
        const VipAdmitClass claimAdmit =
            classifyVipAdmission(rowIdx, targetReqsB,
                                 VipSourceClass::Claim, claimCls);
        if (claimAdmit != VipAdmitClass::Reject &&
            oracleGuidedVipMode() &&
            !currentBOracleVipSelected[rowIdx]) {
            currentBOracleVipSelected[rowIdx] = true;
            stats.oracleSelectedBRowsCount++;
        }
        if (claimAdmit != VipAdmitClass::Reject &&
            tryInsertVipRow(rowAddr, rowBytes,
                            tileBBuffer.data() + rowIdx * rowBytes,
                            VipSourceClass::Claim, claimCls,
                            claimAdmit)) {
            recordVipOracleGuidedAdmit(
                rowIdx, targetReqsB, VipSourceClass::Claim);
            currentBRowState[rowIdx] = BRowState::Empty;
            currentBProtectionClass[rowIdx] = claimCls;
            currentBVipBacked[rowIdx] = true;
            currentBVipSource[rowIdx] = VipSourceClass::Claim;
            currentBVipAdmitClass[rowIdx] = claimAdmit;
            currentBVipPrefetchedCounted[rowIdx] = false;
            servedFromVip = tryServeCurrentBRowFromVip(rowIdx, false);
        }
        if (!servedFromVip) {
            currentBRowState[rowIdx] = BRowState::ReadyFromNextOutput;
            setCurrentOracleSource(rowIdx, OracleSourceClass::Claim);
            recordProtectedBReady(claimCls);
            clearCurrentFutureClaimRow(rowIdx, false, true, false);
            ++reqsCompletedB;
            stats.rxBReadyCount++;
            stats.prefetchedBRowsConsumed++;
            recordOracleSelectedRowMiss(rowIdx);
        }
        recordCurrentBOracleOutcome(rowIdx, true);
        if (reqsCompletedA < targetReqsA) {
            stats.bFetchProgressDuringAFetch++;
        }
        updateABParallelOverlapTracking();
        serviceParallelFetchAB();
        tryLaunchComputeTile();
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
    stats.rxBReadyCount++;
    recordProtectedBReady(BProtectionClass::NextOutput);
    if (coverageShadowUsesPrefetchFeed() &&
        isSmartCoverageTargetForContext(nextOutputCtx, rowIdx)) {
        const Addr rowBytes =
            static_cast<Addr>(nextOutputCtx.curTileN) *
            nextOutputCtx.elemBytes;
        const Addr rowAddr = nextOutputCtx.baseB +
            ((static_cast<Addr>(nextOutputCtx.k + rowIdx) *
              nextOutputCtx.size + nextOutputCtx.j) *
             nextOutputCtx.elemBytes);
        tryInsertCoverageShadowRow(
            rowAddr, rowBytes,
            nextOutputTileBBuffer.data() + rowIdx * rowBytes,
            CoverageShadowInsertSource::NextOutput);
    }
    {
        const Addr rowBytes =
            static_cast<Addr>(nextOutputCtx.curTileN) * nextOutputCtx.elemBytes;
        const Addr rowAddr = nextOutputCtx.baseB +
            ((static_cast<Addr>(nextOutputCtx.k + rowIdx) *
              nextOutputCtx.size + nextOutputCtx.j) *
             nextOutputCtx.elemBytes);
        const OracleDistanceBucket dist =
            oracleDistanceBucket(rowIdx, nextOutputCtx.curTileK);
        const BProtectionClass vipCls =
            rowIdx < computeLaunchWindowB()
                ? BProtectionClass::NextOutput
                : BProtectionClass::FutureHoleFilling;
        const VipAdmitClass mhotAdmit =
            classifyMHotAdmission(rowIdx, nextOutputCtx.curTileK,
                                  VipSourceClass::NextOutput, vipCls);
        if (mhotAdmit != VipAdmitClass::Reject) {
            tryInsertMHotRow(rowAddr, rowBytes,
                             nextOutputTileBBuffer.data() + rowIdx * rowBytes,
                             VipSourceClass::NextOutput, vipCls,
                             mhotAdmit, dist);
        }
        const VipAdmitClass admit =
            classifyVipAdmission(rowIdx, nextOutputCtx.curTileK,
                                 VipSourceClass::NextOutput, vipCls);
        if (tryInsertVipRow(rowAddr, rowBytes,
                            nextOutputTileBBuffer.data() + rowIdx * rowBytes,
                            VipSourceClass::NextOutput, vipCls, admit)) {
            recordVipOracleGuidedAdmit(
                rowIdx, nextOutputCtx.curTileK, VipSourceClass::NextOutput);
        }
    }
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
MatrixFlowEngine::onCoverageGatherBRowComplete(uint32_t rowIdx)
{
    const uint64_t rowGen = coverageGatherBRowGeneration[rowIdx];
    if (!coverageGatherValid || rowGen == 0 ||
        rowGen != activeCoverageGatherGeneration) {
        ++stats.coverageGatherLateCompletionCount;
        return;
    }

    if (coverageGatherBBounceActive[rowIdx]) {
        uint8_t *src = coverageGatherBBounceBuffer.data() +
                       rowIdx * readBouncePitch +
                       coverageGatherBBounceOffset[rowIdx];
        uint8_t *dst = coverageGatherBBounceBuffer.data() +
                       rowIdx * readBouncePitch;
        std::memmove(dst, src, coverageGatherBBounceRowBytes[rowIdx]);
        coverageGatherBBounceActive[rowIdx] = false;
    }

    ++coverageGatherRowsBCompleted;
    ++stats.coverageGatherCompletionCount;

    bool downgraded = false;
    if (!coverage2DGatherPingPongFirstCutMode()) {
        switch (currentBRowState[rowIdx]) {
          case BRowState::Empty:
            ++stats.coverageGatherCompletionWhileRowEmptyCount;
            break;
          case BRowState::InflightFromNextOutput:
          case BRowState::InflightFromNormal:
            ++stats.coverageGatherCompletionWhileRowInflightCount;
            break;
          case BRowState::ReadyFromNextOutput:
          case BRowState::ReadyFromNormal:
            ++stats.coverageGatherCompletionWhileRowReadyCount;
            break;
          case BRowState::Consumed:
            ++stats.coverageGatherCompletionWhileRowConsumedCount;
            break;
        }
        if (coverage2DGatherMinGuaranteeFirstCutMode() &&
            currentBRowState[rowIdx] != BRowState::Empty) {
            downgraded = true;
            ++stats.coverageGatherCompletionDowngradedCount;
        }
    }

    const GemmContext &gctx =
        coverage2DGatherPingPongFirstCutMode() ? coverageGatherCtx : ctx;
    const Addr rowBytes =
        static_cast<Addr>(gctx.curTileN) * gctx.elemBytes;
    const Addr rowAddr = gctx.baseB +
        ((static_cast<Addr>(gctx.k + rowIdx) * gctx.size + gctx.j) *
         gctx.elemBytes);
    bool inserted = false;
    if (coverage2DGatherPingPongFirstCutMode()) {
        inserted = tryInsertCoverageShadowRowInBank(
            coverageShadowFillBBuffer, coverageShadowFillBSlots,
            coverageShadowFillPoolOccupancy, rowAddr, rowBytes,
            coverageGatherBBounceBuffer.data() + rowIdx * readBouncePitch,
            CoverageShadowInsertSource::Gather);
    } else if (coverage2DGatherMinGuaranteeFirstCutMode() && downgraded) {
        inserted = tryInsertCoverageShadowRowLowPriority(
            rowAddr, rowBytes,
            coverageGatherBBounceBuffer.data() + rowIdx * readBouncePitch,
            CoverageShadowInsertSource::Gather);
        if (inserted) {
            ++stats.coverageGatherCompletionInsertedLowPriorityCount;
        } else {
            ++stats.coverageGatherCompletionDiscardedCount;
        }
    } else {
        inserted = tryInsertCoverageShadowRow(
            rowAddr, rowBytes,
            coverageGatherBBounceBuffer.data() + rowIdx * readBouncePitch,
            CoverageShadowInsertSource::Gather);
    }

    // Gather completions can arrive after tile-entry prime already tagged this
    // row as a known miss. Re-open the row and try to materialize it
    // immediately so the completion still has a chance to help this tile.
    if (inserted && !coverage2DGatherPingPongFirstCutMode() &&
        rowIdx < targetReqsB &&
        currentBRowState[rowIdx] == BRowState::Empty &&
        isCoverageBlindspotTargetPattern(rowIdx, BProtectionClass::None,
                                         rowAddr)) {
        currentBCoverageShadowKnownMiss[rowIdx] = false;
        currentBCoverageShadowScanned[rowIdx] = false;
        tryPrimeCurrentBRowFromCoverageShadow(rowIdx,
                                              BProtectionClass::None);
    }

    if (coverageGatherRowsBCompleted >= coverageGatherTargetB) {
        coverageGatherValid = false;
    } else {
        trySendMoreCoverageGatherB();
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
            "readyA=%u/%u readyB=%u/%u launchWin=(%u,%u) "
            "cycles=%llu done@%llu\n",
            ctx.i, ctx.j, ctx.k, ctx.curTileM, ctx.curTileN, ctx.curTileK,
            reqsCompletedA, targetReqsA, reqsCompletedB, targetReqsB,
            computeLaunchWindowA(), computeLaunchWindowB(),
            static_cast<unsigned long long>(tileCycles),
            static_cast<unsigned long long>(doneAt));
    if (hierarchicalProtectedBSchedulerMode()) {
        warn("%s: hier-trace launchCompute i=%u j=%u k=%u "
             "readyA=%u/%u readyB=%u/%u winA=%u winB=%u cycles=%llu\n",
             name(), ctx.i, ctx.j, ctx.k,
             reqsCompletedA, targetReqsA, reqsCompletedB, targetReqsB,
             computeLaunchWindowA(), computeLaunchWindowB(),
             static_cast<unsigned long long>(tileCycles));
    }

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
        if (claimBasedHoleFillingMode() && issued_rows_total >
            rows_consumed_before_fallback) {
            stageCurrentFutureClaimsFromNextOutput(
                rows_consumed_before_fallback, issued_rows_total,
                activeNextOutputPrefetchGeneration);
        }
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
        if (mhotPoolEnabled() && rows_consumed_before_fallback > 0) {
            for (uint32_t row = 0; row < rows_consumed_before_fallback; ++row) {
                if (!nextOutputPrefetchFromMHot[row]) {
                    continue;
                }
                recordMHotMaterializeCurrent(nextOutputPrefetchMHotSource[row],
                                             row, ctx.curTileK, false);
            }
        }
        if (vipPoolEnabled() && rows_consumed_before_fallback > 0) {
            const Addr rowBytes =
                static_cast<Addr>(ctx.curTileN) * ctx.elemBytes;
            for (uint32_t row = 0; row < rows_consumed_before_fallback; ++row) {
                const VipAdmitClass admit =
                    classifyVipAdmission(row, ctx.curTileK,
                                         VipSourceClass::CarryOver,
                                         BProtectionClass::CarryOver);
                if (admit == VipAdmitClass::Reject) {
                    continue;
                }
                stageCurrentOracleVipSelectedRow(row);
                const Addr rowAddr = ctx.baseB +
                    ((static_cast<Addr>(ctx.k + row) * ctx.size + ctx.j) *
                     ctx.elemBytes);
                if (!tryInsertVipRow(rowAddr, rowBytes,
                                     tileBBuffer.data() + row * rowBytes,
                                     VipSourceClass::CarryOver,
                                     BProtectionClass::CarryOver, admit)) {
                    continue;
                }
                recordVipOracleGuidedAdmit(
                    row, ctx.curTileK, VipSourceClass::CarryOver);
                if (vipForcedDiversionMode()) {
                    stageCurrentVipBackedRow(row, VipSourceClass::CarryOver,
                                             BProtectionClass::CarryOver,
                                             admit, false);
                }
            }
        }
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
        const uint32_t issued_rows_total =
            std::min(nextOutputPrefetchReqsIssuedB, nextOutputCtx.curTileK);
        if (claimBasedHoleFillingMode() &&
            nextOutputPrefetchMatchesCurrentTile() &&
            issued_rows_total > rows_consumed_before_fallback) {
            stageCurrentFutureClaimsFromNextOutput(
                rows_consumed_before_fallback, issued_rows_total,
                activeNextOutputPrefetchGeneration);
        }
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
    hierarchicalDebugComputeDeferCount = 0;
    clearFallbackAutopsyTracking();
    if (hierarchicalProtectedBSchedulerMode()) {
        warn("%s: hier-trace startMatrixCompute accepted desc=%#llx\n",
             name(), static_cast<unsigned long long>(descriptorAddr));
    }

    DPRINTF(MatrixFlow,
            "startMatrixCompute: descriptor=%#llx\n",
            static_cast<unsigned long long>(descriptorAddr));

    issueFetchDescriptor();
}

void
MatrixFlowEngine::processComputeDone()
{
    if (!fullInputsReadyForAccumulate()) {
        if (claimBasedHoleFillingMode()) {
            expireCurrentFutureClaimsIfBlocked();
        }
        // In parallel A/B modes, compute may launch on the minimum window.
        // If all currently in-flight fetches drain before the full tile is
        // ready, there may be no more DMA completions to re-enter the fetch
        // scheduler. Re-drive it here so the remaining A/B rows can still be
        // issued and we don't livelock in computeDone deferrals forever.
        if (parallelABSchedulerMode()) {
            serviceParallelFetchAB(false);
        }

        hierarchicalDebugComputeDeferCount++;
        if (hierarchicalProtectedBSchedulerMode() &&
            (hierarchicalDebugComputeDeferCount <= 8 ||
             (hierarchicalDebugComputeDeferCount &
              (hierarchicalDebugComputeDeferCount - 1)) == 0)) {
            warn("%s: hier-trace computeDone defer #%llu i=%u j=%u k=%u "
                 "readyA=%u/%u readyB=%u/%u issuedA=%u issuedB=%u "
                 "nextOutValid=%d nextOutIssued=%u nextOutDone=%u\n",
                 name(),
                 static_cast<unsigned long long>(
                     hierarchicalDebugComputeDeferCount),
                 ctx.i, ctx.j, ctx.k,
                 reqsCompletedA, targetReqsA, reqsCompletedB, targetReqsB,
                 reqsIssuedA, reqsIssuedB,
                 nextOutputPrefetchValid,
                 nextOutputPrefetchReqsIssuedB,
                 nextOutputPrefetchRowsBCompleted);
            warn("%s: hier-trace defer-state cursor=%u issueLimit=%u "
                 "nextIssuable=%d nextProtected=%d future=%d current=%d "
                 "normal=%d canIssueA=%d carryActive=%d carryIssued=%u "
                 "carryDone=%u\n",
                 name(), nextNormalBRowCursor, normalIssueLimit(),
                 nextIssuableBRow(), nextProtectedBRow(),
                 hasFutureProtectedBPressure(),
                 hasCurrentProtectedBPressure(),
                 hasNormalBPressure(), canIssueA(), carryOverBActive,
                 carryOverBRowsIssued, carryOverBRowsCompleted);
        }
        DPRINTF(MatrixFlow,
                "processComputeDone deferred: tile i=%u j=%u k=%u "
                "readyA=%u/%u readyB=%u/%u\n",
                ctx.i, ctx.j, ctx.k,
                reqsCompletedA, targetReqsA,
                reqsCompletedB, targetReqsB);
        schedule(computeDoneEvent, curTick() + clockPeriod());
        return;
    }

    for (uint32_t r = 0; r < ctx.curTileK; ++r) {
        currentBRowState[r] = BRowState::Consumed;
    }
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
    if (hierarchicalProtectedBSchedulerMode()) {
        warn("%s: hier-trace onWriteFlagComplete flag=%#llx value=%llu\n",
             name(), static_cast<unsigned long long>(pendingFlagAddr),
             static_cast<unsigned long long>(completionFlagValue));
    }
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
