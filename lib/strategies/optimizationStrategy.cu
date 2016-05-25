#include "strategies/optimizationStrategy.cuh"

OptimizationStrategy::OptimizationStrategy(const char *symbol, std::map<std::string, int> *dataIndexMap, int group, Configuration *configuration)
        : Strategy(symbol, dataIndexMap) {
    this->group = group;
    this->configuration = configuration;
    this->tickPreviousDataPoint = nullptr;
}

OptimizationStrategy::~OptimizationStrategy() {
    delete configuration;
    delete tickPreviousDataPoint;
}

int OptimizationStrategy::getGroup() {
    return this->group;
}

void OptimizationStrategy::tick(double *dataPoint) {
    std::map<std::string, int> *dataIndexMap = this->getDataIndexMap();

    if (this->tickPreviousDataPoint) {
        // Simulate expiry of and profit/loss related to positions held.
        closeExpiredPositions(this->tickPreviousDataPoint[(*dataIndexMap)["close"]], dataPoint[(*dataIndexMap)["timestamp"]] - 1);
    }

    this->tickPreviousDataPoint = dataPoint;
}

Configuration *OptimizationStrategy::getConfiguration() {
    return this->configuration;
}
