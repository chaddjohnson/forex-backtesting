#include "strategies/combinedStrategy.cuh"

CombinedStrategy::CombinedStrategy(const char *symbol, std::map<std::string, int> *dataIndexMap, std::vector<Configuration*> configurations)
        : Strategy(symbol, dataIndexMap) {
    this->configurations = configurations;
}

CombinedStrategy::~CombinedStrategy() {
    delete configurations;
    delete tickPreviousDataPoint;
}

void CombinedStrategy::tick(double *dataPoint) {
    std::map<std::string, int> dataIndexMap = this->getDataIndexMap();

    if (this->tickPreviousDataPoint) {
        // Simulate expiry of and profit/loss related to positions held.
        closeExpiredPositions(this->tickPreviousDataPoint[dataIndexMap["close"]], dataPoint[dataIndexMap["timestamp"]] - 1);
    }

    this->tickPreviousDataPoint = dataPoint;
}
