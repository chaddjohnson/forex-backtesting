#ifndef STRATEGYFACTORY_H
#define STRATEGYFACTORY_H

#include <string>
#include "strategies/optimizationStrategy.h"
#include "strategies/reversalsOptimizationStrategy.h"
#include "types/configuration.h"

class OptimizationStrategyFactory {
    public:
        static OptimizationStrategy *create(std::string name, std::string symbol, int group, Configuration *configuration);
};

#endif
