#include "strategies/combinedStrategy.cuh"

__device__ __host__ CombinedStrategy::CombinedStrategy(const char *symbol, std::vector<Configuration*> configurations)
        : Strategy(symbol) {
    this->configurations = configurations;
}

__device__ __host__ void CombinedStrategy::tick(double *dataPoint) {
    if (this->tickPreviousDataPoint) {
        // Simulate expiry of and profit/loss related to positions held.
        closeExpiredPositions(this->tickPreviousDataPoint[this->configuration.close], this->tickPreviousDataPoint[this->configuration.timestamp] - 1);
    }
    this->tickPreviousDataPoint = dataPoint;
}
