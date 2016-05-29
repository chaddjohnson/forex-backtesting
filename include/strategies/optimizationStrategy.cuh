#ifndef OPTIMIZATIONSTRATEGY_H
#define OPTIMIZATIONSTRATEGY_H

#include "strategy.cuh"
#include "types/configuration.cuh"
#include "types/basicDataIndexMap.cuh"

class OptimizationStrategy : public Strategy {
    private:
        int group;
        Configuration *configuration;
        double *tickPreviousDataPoint;

    protected:
        __device__ void tick(double *dataPoint);

    public:
        __host__ OptimizationStrategy(const char *symbol, BasicDataIndexMap dataIndexMap, int group, Configuration *configuration);
        __host__ virtual ~OptimizationStrategy();
        __device__ int getGroup();
        __device__ Configuration *getConfiguration();
};

#endif
