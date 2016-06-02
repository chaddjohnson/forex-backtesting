#ifndef STRATEGYFACTORY_H
#define STRATEGYFACTORY_H

#include <string>
#include "strategies/optimizationStrategy.cuh"
#include "strategies/reversalsOptimizationStrategy.cuh"
#include "types/configuration.cuh"
#include "types/basicDataIndexMap.cuh"

class OptimizationStrategyFactory {
    public:
        static OptimizationStrategy *create(const char *name, const char *symbol, BasicDataIndexMap dataIndexMap, int group, Configuration configuration);
};

#endif
