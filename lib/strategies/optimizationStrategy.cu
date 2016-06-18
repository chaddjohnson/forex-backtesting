#include "strategies/optimizationStrategy.cuh"

__device__ __host__ OptimizationStrategy::OptimizationStrategy(const char *symbol, Configuration configuration)
        : Strategy(symbol) {
    this->configuration = configuration;
}

__device__ __host__ Configuration &OptimizationStrategy::getConfiguration() {
    return this->configuration;
}
