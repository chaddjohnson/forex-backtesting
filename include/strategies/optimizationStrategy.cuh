#ifndef OPTIMIZATIONSTRATEGY_H
#define OPTIMIZATIONSTRATEGY_H

#include "strategy.cuh"
#include "types/configuration.cuh"
#include "types/real.cuh"

class OptimizationStrategy : public Strategy {
    private:
        int group;
        Configuration configuration;

    public:
        __device__ __host__ OptimizationStrategy(const char *symbol, int group, Configuration configuration);
        __device__ __host__ ~OptimizationStrategy() {}
        __device__ __host__ int getGroup();
        __device__ __host__ Configuration &getConfiguration();
};

#endif
