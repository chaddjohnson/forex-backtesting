#include "strategies/strategy.cuh"

__device__ __host__ Strategy::Strategy(const char *symbol, BasicDataIndexMap dataIndexMap) {
    this->symbol = symbol;
    this->dataIndexMap = dataIndexMap;
    this->profitLoss = 0.0;
    this->winCount = 0;
    this->loseCount = 0;
    this->consecutiveLosses = 0;
    this->maximumConsecutiveLosses = 0;
    this->minimumProfitLoss = 99999.0;

    int i = 0;

    for (i=0; i<10; i++) {
        this->openPositions[i] = nullptr;
    }
}

__device__ __host__ BasicDataIndexMap Strategy::getDataIndexMap() {
    return this->dataIndexMap;
}

__device__ __host__ const char *Strategy::getSymbol() {
    return this->symbol;
}

__device__ __host__ double Strategy::getProfitLoss() {
    return this->profitLoss;
}

__device__ __host__ void Strategy::setProfitLoss(double profitLoss) {
    this->profitLoss = profitLoss;
}

__device__ __host__ double Strategy::getWinRate() {
    if (this->winCount + this->loseCount == 0) {
        return 0;
    }
    return this->winCount / (this->winCount + this->loseCount);
}

__device__ __host__ void Strategy::closeExpiredPositions(double price, time_t timestamp) {
    int i = 0;

    for (i=0; i<10; i++) {
        if (this->openPositions[i]) {
            Position position = *this->openPositions[i];
            double positionProfitLoss = 0.0;

            if (position.getHasExpired(timestamp)) {
                // Close the position since it is open and has expired.
                position.close(price, timestamp);

                // Remove the position's investment amount from the total profit/loss for this strategy.
                this->profitLoss -= position.getInvestment();

                // Add the profit/loss for this position to the profit/loss for this strategy.
                positionProfitLoss = position.getProfitLoss();
                this->profitLoss += positionProfitLoss;

                if (positionProfitLoss > position.getInvestment()) {
                    this->winCount++;
                    this->consecutiveLosses = 0;
                }
                if (profitLoss == 0) {
                    this->loseCount++;
                    this->consecutiveLosses++;
                }

                // Update the minimum profit/loss for this strategy if applicable.
                if (this->consecutiveLosses > this->maximumConsecutiveLosses) {
                    this->maximumConsecutiveLosses = this->consecutiveLosses;
                }

                // Remove the position from the list of open positions, and free memory.
                // Delete the position.
                delete this->openPositions[i];
                this->openPositions[i] = nullptr;
            }
        }
    }
}

__device__ __host__ StrategyResults Strategy::getResults() {
    StrategyResults results;

    results.profitLoss = this->profitLoss;
    results.winCount = this->winCount;
    results.loseCount = this->loseCount;
    results.winRate = getWinRate();
    results.tradeCount = this->winCount + this->loseCount;
    results.maximumConsecutiveLosses = this->maximumConsecutiveLosses;
    results.minimumProfitLoss = this->minimumProfitLoss;

    return results;
}

__device__ __host__ void Strategy::addPosition(Position *position) {
    bool done = false;
    int i = 0;

    while (!done || i == 10) {
        // If there is an unused position slot, then use it.
        if (!this->openPositions[i]) {
            this->openPositions[i] = position;
            done = true;
        }

        i++;
    }
}
