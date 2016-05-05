#include "strategies/combinedStrategy.h"

CombinedStrategy::CombinedStrategy(std::string symbol, std::vector<Configuration*> configurations)
        : Strategy(symbol) {
    this->configurations = configurations;
}

CombinedStrategy::~CombinedStrategy() {
    delete configurations;
    delete tickPreviousDataPoint;
}

void CombinedStrategy::tick(double *dataPoint) {
    if (this->tickPreviousDataPoint) {
        // Simulate expiry of and profit/loss related to positions held.
        // TODO: Timestamp stuff.
        closeExpiredPositions(this->tickPreviousDataPoint[configuration->close], dataPoint[configuration->timestamp] - 1000);
    }

    this->tickPreviousDataPoint = dataPoint;
}
