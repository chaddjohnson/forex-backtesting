#ifndef COMBINEDSTRATEGY_H
#define COMBINEDSTRATEGY_H

#include <vector>
#include "strategy.cuh"

class CombinedStrategy : public Strategy {
    private:
        std::vector<Configuration*> configurations;
        double *tickPreviousDataPoint;

    protected:
        __device__ __host__ void tick(double *dataPoint);
        __device__ __host__ std::vector<Configuration*> getConfigurations();

    public:
        __device__ __host__ CombinedStrategy(const char *symbol, std::vector<Configuration*> configurations);
        __device__ __host__ ~CombinedStrategy() {}
};

#endif
