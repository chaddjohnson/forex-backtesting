#include "strategies/optimizationStrategy.h"

OptimizationStrategy::OptimizationStrategy(std::string symbol, int group, Configuration *configuration)
        : Strategy(symbol) {
    this->group = group;
    this->configuration = configuration;
}

OptimizationStrategy::~OptimizationStrategy() {
    delete configuration;
    delete tickPreviousDataPoint;
}

int OptimizationStrategy::getGroup() {
    return this->group;
}

void OptimizationStrategy::tick(double *dataPoint) {
    if (this->tickPreviousDataPoint) {
        // Simulate expiry of and profit/loss related to positions held.
        // TODO: Timestamp stuff.
        closeExpiredPositions(this->tickPreviousDataPoint[configuration->close], dataPoint[configuration->timestamp] - 1000);
    }

    this->tickPreviousDataPoint = dataPoint;
}

Configuration *OptimizationStrategy::getConfiguration() {
    return this->configuration;
}
