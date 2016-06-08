#include "strategies/optimizationStrategy.cuh"

__device__ __host__ OptimizationStrategy::OptimizationStrategy(const char *symbol, int group, Configuration configuration)
        : Strategy(symbol) {
    this->group = group;
    this->configuration = configuration;
}

__device__ __host__ int OptimizationStrategy::getGroup() {
    return this->group;
}

__device__ __host__ Configuration &OptimizationStrategy::getConfiguration() {
    return this->configuration;
}
