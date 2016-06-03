#include "factories/optimizationStrategyFactory.cuh"

OptimizationStrategy *OptimizationStrategyFactory::create(const char *name, const char *symbol, int group, Configuration configuration) {
    //if (name == "reversals") {
        return new ReversalsOptimizationStrategy(symbol, group, configuration);
    //}
}
