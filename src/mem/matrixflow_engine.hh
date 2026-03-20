#ifndef __MEM_MATRIXFLOW_ENGINE_HH__
#define __MEM_MATRIXFLOW_ENGINE_HH__

#include <cstdint>
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
    static constexpr uint32_t kMaxInFlight = 32;

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

        explicit EngineStats(statistics::Group *parent);
    };

    const unsigned macArraySize;
    const Cycles computeLatencyPerOp;

    Phase phase;
    GemmContext ctx;

    /** Tile buffers: up to kMaxTileDim^2 uint32_t elements each. */
    std::vector<uint8_t> tileABuffer;
    std::vector<uint8_t> tileBBuffer;
    std::vector<uint8_t> tileCBuffer;

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
    uint32_t reqsIssuedC = 0;
    uint32_t reqsCompletedC = 0;
    uint32_t targetReqsC = 0;

    EventFunctionWrapper fetchDescCompleteEvent;
    EventFunctionWrapper computeDoneEvent;
    EventFunctionWrapper writeFlagCompleteEvent;

    /** One completion event per row slot (avoids "Event already scheduled"). */
    std::vector<EventFunctionWrapper> fetchARowEvents;
    std::vector<EventFunctionWrapper> fetchBRowEvents;
    std::vector<EventFunctionWrapper> writeCRowEvents;

    EngineStats stats;

    uint64_t estimateTileCycles(uint32_t tileM, uint32_t tileN,
                                uint32_t tileK) const;
    void resetContext();
    void issueFetchDescriptor();
    void prepareOutputTile();
    void issueFetchATile();
    void issueFetchBTile();
    void issueWriteCTile();
    void trySendMoreA();
    void trySendMoreB();
    void trySendMoreC();
    void issueWriteFlag();
    void onFetchDescComplete();
    void onFetchARowComplete();
    void onFetchBRowComplete();
    void onWriteCRowComplete();
    void onWriteFlagComplete();
    void launchComputeTile();
    void accumulateCurrentTile();
    void advanceTile();
    void processComputeDone();

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
