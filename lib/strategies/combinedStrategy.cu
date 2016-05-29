#include "strategies/combinedStrategy.cuh"

__host__ CombinedStrategy::CombinedStrategy(const char *symbol, BasicDataIndexMap dataIndexMap, std::vector<Configuration*> configurations)
        : Strategy(symbol, dataIndexMap) {
    this->configurations = configurations;
}

__host__ CombinedStrategy::~CombinedStrategy() {
    delete configurations;
    delete tickPreviousDataPoint;
}

__device__ void CombinedStrategy::tick(double *dataPoint) {
    BasicDataIndexMap dataIndexMap = this->getDataIndexMap();

    if (this->tickPreviousDataPoint) {
        // Simulate expiry of and profit/loss related to positions held.
        closeExpiredPositions(this->tickPreviousDataPoint[dataIndexMap.close], dataPoint[dataIndexMap.timestamp] - 1);
    }

    this->tickPreviousDataPoint = dataPoint;
}
