#include "strategies/optimizationStrategy.cuh"

__device__ __host__ OptimizationStrategy::OptimizationStrategy(const char *symbol, int group, Configuration configuration)
        : Strategy(symbol) {
    this->group = group;
    this->configuration = configuration;
    this->tickPreviousDataPoint = nullptr;
}

__device__ __host__ int OptimizationStrategy::getGroup() {
    return this->group;
}

__device__ __host__ void OptimizationStrategy::tick(double *dataPoint) {
    if (this->tickPreviousDataPoint) {
        // Simulate expiry of and profit/loss related to positions held.
        closeExpiredPositions(this->tickPreviousDataPoint[this->configuration.close], this->tickPreviousDataPoint[this->configuration.timestamp] - 1);
    }
    this->tickPreviousDataPoint = dataPoint;
}

__device__ __host__ Configuration &OptimizationStrategy::getConfiguration() {
    return this->configuration;
}
