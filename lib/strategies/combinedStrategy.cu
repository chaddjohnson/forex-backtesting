#include "strategies/combinedStrategy.cuh"

CombinedStrategy::CombinedStrategy(char *symbol, std::map<std::string, int> *dataIndex, std::vector<Configuration*> configurations)
        : Strategy(symbol, dataIndex) {
    this->configurations = configurations;
}

CombinedStrategy::~CombinedStrategy() {
    delete configurations;
    delete tickPreviousDataPoint;
}

void CombinedStrategy::tick(double *dataPoint) {
    std::map<std::string, int> dataIndex = this->getDataIndex();

    if (this->tickPreviousDataPoint) {
        // Simulate expiry of and profit/loss related to positions held.
        closeExpiredPositions(this->tickPreviousDataPoint[dataIndex["close"]], dataPoint[dataIndex["timestamp"]] - 1);
    }

    this->tickPreviousDataPoint = dataPoint;
}
