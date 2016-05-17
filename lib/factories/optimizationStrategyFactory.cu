#include "factories/optimizationStrategyFactory.cuh"

OptimizationStrategy *OptimizationStrategyFactory::create(const char *name, const char *symbol, std::map<std::string, int> *dataIndex, int group, Configuration *configuration) {
    //if (name == "reversals") {
        return new ReversalsOptimizationStrategy(symbol, dataIndex, group, configuration);
    //}
}
