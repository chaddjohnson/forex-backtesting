#include "factories/optimizationStrategyFactory.h"

OptimizationStrategy *OptimizationStrategyFactory::create(std::string name, std::string symbol, int group, Configuration *configuration) {
    //if (name == "reversals") {
        return new ReversalsOptimizationStrategy(symbol, group, configuration);
    //}
}
