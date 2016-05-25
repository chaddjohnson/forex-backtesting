#ifndef STRATEGYFACTORY_H
#define STRATEGYFACTORY_H

#include <string>
#include "strategies/optimizationStrategy.cuh"
#include "strategies/reversalsOptimizationStrategy.cuh"
#include "types/configuration.cuh"

class OptimizationStrategyFactory {
    public:
        static OptimizationStrategy *create(const char *name, const char *symbol, std::map<std::string, int> *dataIndexMap, int group, Configuration *configuration);
};

#endif
