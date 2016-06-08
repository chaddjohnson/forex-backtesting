#include "strategies/combinedStrategy.cuh"

__device__ __host__ CombinedStrategy::CombinedStrategy(const char *symbol, std::vector<Configuration*> configurations)
        : Strategy(symbol) {
    this->configurations = configurations;
}
