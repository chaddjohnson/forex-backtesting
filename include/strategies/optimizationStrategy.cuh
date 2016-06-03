#ifndef OPTIMIZATIONSTRATEGY_H
#define OPTIMIZATIONSTRATEGY_H

#include "strategy.cuh"
#include "types/configuration.cuh"

class OptimizationStrategy : public Strategy {
    private:
        int group;
        Configuration configuration;
        double *tickPreviousDataPoint;

    protected:
        __device__ __host__ void tick(double *dataPoint);

    public:
        __device__ __host__ OptimizationStrategy(const char *symbol, int group, Configuration configuration);
        __device__ __host__ ~OptimizationStrategy() {}
        __device__ __host__ int getGroup();
        __device__ __host__ Configuration &getConfiguration();
};

#endif
