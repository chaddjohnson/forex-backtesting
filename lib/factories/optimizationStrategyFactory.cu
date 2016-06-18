#include "factories/optimizationStrategyFactory.cuh"

OptimizationStrategy *OptimizationStrategyFactory::create(const char *name, const char *symbol, Configuration configuration) {
    //if (name == "reversals") {
        return new ReversalsOptimizationStrategy(symbol, configuration);
    //}
}
