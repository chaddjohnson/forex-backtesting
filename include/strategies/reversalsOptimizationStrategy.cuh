#ifndef REVERSALSOPTIMIZATIONSTRATEGY_H
#define REVERSALSOPTIMIZATIONSTRATEGY_H

#include "optimizationStrategy.cuh"
#include "types/configuration.cuh"
#include "types/basicDataIndexMap.cuh"

class ReversalsOptimizationStrategy : public OptimizationStrategy {
    private:
        Configuration configuration;
        double *previousDataPoint;
        bool putNextTick;
        bool callNextTick;
        int expirationMinutes;

    public:
        __device__ __host__ ReversalsOptimizationStrategy(const char *symbol, BasicDataIndexMap dataIndexMap, int group, Configuration configuration);
        __device__ __host__ ~ReversalsOptimizationStrategy();
        __device__ __host__ void backtest(double *dataPoint, double investment, double profitability);
};

#endif
