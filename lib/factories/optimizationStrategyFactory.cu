#include "factories/optimizationStrategyFactory.cuh"

OptimizationStrategy *OptimizationStrategyFactory::create(const char *name, const char *symbol, BasicDataIndexMap dataIndexMap, int group, Configuration *configuration) {
    //if (name == "reversals") {
        return new ReversalsOptimizationStrategy(symbol, dataIndexMap, group, configuration);
    //}
}
