#ifndef OPTIMIZATIONSTRATEGY_H
#define OPTIMIZATIONSTRATEGY_H

#include "strategy.cuh"
#include "types/configuration.cuh"

class OptimizationStrategy : public Strategy {
    private:
        Configuration configuration;

    public:
        __device__ __host__ OptimizationStrategy(const char *symbol, Configuration configuration);
        __device__ __host__ ~OptimizationStrategy() {}
        __device__ __host__ Configuration getConfiguration();
        __device__ __host__ StrategyResult getResult();
};

#endif
