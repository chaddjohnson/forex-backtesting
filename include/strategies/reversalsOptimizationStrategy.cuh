#ifndef REVERSALSOPTIMIZATIONSTRATEGY_H
#define REVERSALSOPTIMIZATIONSTRATEGY_H

#include "optimizationStrategy.cuh"
#include "types/configuration.cuh"
#include "types/real.cuh"

class ReversalsOptimizationStrategy : public OptimizationStrategy {
    private:
        Configuration configuration;
        bool putNextTick;
        bool callNextTick;
        int expirationMinutes;

    public:
        __device__ __host__ ReversalsOptimizationStrategy(const char *symbol, int group, Configuration configuration);
        __device__ __host__ ~ReversalsOptimizationStrategy() {}
        __device__ __host__ void backtest(Real *dataPoint, Real investment, Real profitability);
};

#endif
