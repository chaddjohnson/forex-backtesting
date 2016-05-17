#include "strategies/optimizationStrategy.cuh"

OptimizationStrategy::OptimizationStrategy(char *symbol, std::map<std::string, int> *dataIndex, int group, Configuration *configuration)
        : Strategy(symbol, dataIndex) {
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
    std::map<std::string, int> *dataIndex = this->getDataIndex();

    if (this->tickPreviousDataPoint) {
        // Simulate expiry of and profit/loss related to positions held.
        closeExpiredPositions(this->tickPreviousDataPoint[(*dataIndex)["close"]], dataPoint[(*dataIndex)["timestamp"]] - 1);
    }

    this->tickPreviousDataPoint = dataPoint;
}

Configuration *OptimizationStrategy::getConfiguration() {
    return this->configuration;
}
