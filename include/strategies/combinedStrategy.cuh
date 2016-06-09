#ifndef COMBINEDSTRATEGY_H
#define COMBINEDSTRATEGY_H

#include <vector>
#include "strategy.cuh"
#include "types/real.cuh"

class CombinedStrategy : public Strategy {
    private:
        std::vector<Configuration*> configurations;

    protected:
        __device__ __host__ std::vector<Configuration*> getConfigurations();

    public:
        __device__ __host__ CombinedStrategy(const char *symbol, std::vector<Configuration*> configurations);
        __device__ __host__ ~CombinedStrategy() {}
};

#endif
