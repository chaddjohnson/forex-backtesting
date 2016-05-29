#ifndef COMBINEDSTRATEGY_H
#define COMBINEDSTRATEGY_H

#include <vector>
#include "strategy.cuh"
#include "types/basicDataIndexMap.cuh"

class CombinedStrategy : public Strategy {
    private:
        std::vector<Configuration*> configurations;
        double *tickPreviousDataPoint;

    protected:
        __device__ void tick(double *dataPoint);
        __device__ std::vector<Configuration*> getConfigurations();

    public:
        __host__ CombinedStrategy(const char *symbol, BasicDataIndexMap dataIndexMap, std::vector<Configuration*> configurations);
        __host__ ~CombinedStrategy();
};

#endif
