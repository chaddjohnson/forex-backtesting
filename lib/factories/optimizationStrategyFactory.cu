#include "factories/optimizationStrategyFactory.cuh"

OptimizationStrategy *OptimizationStrategyFactory::create(std::string name, std::string symbol, std::map<std::string, int> *dataIndex, int group, Configuration *configuration) {
    //if (name == "reversals") {
        return new ReversalsOptimizationStrategy(symbol, dataIndex, group, configuration);
    //}
}
