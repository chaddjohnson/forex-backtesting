#include "strategies/strategy.cuh"

__device__ __host__ Strategy::Strategy(const char *symbol) {
    this->symbol = symbol;
    this->profitLoss = 0.0;
    this->winCount = 0;
    this->loseCount = 0;
    this->consecutiveLosses = 0;
    this->maximumConsecutiveLosses = 0;
    this->minimumProfitLoss = 99999.0;
    this->previousClose = 0.0;
    this->previousTimestamp = 0.0;

    for (int i=0; i<10; i++) {
        this->openPositions[i] = nullptr;
    }
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
    return (double)this->winCount / ((double)this->winCount + (double)this->loseCount);
}

__device__ __host__ void Strategy::tick(double *dataPoint) {
    // Simulate expiry of and profit/loss related to positions held.
    closeExpiredPositions(this->previousClose, this->previousTimestamp - 1);
}

__device__ __host__ void Strategy::closeExpiredPositions(double price, double timestamp) {
    if (!price || !timestamp) {
        return;
    }

    for (int i=0; i<10; i++) {
        if (this->openPositions[i]) {
            if (this->openPositions[i]->getHasExpired(timestamp)) {
                // Close the position since it is open and has expired.
                this->openPositions[i]->close(price, timestamp);

                double positionProfitLoss = this->openPositions[i]->getProfitLoss();
                double positionInvestment = this->openPositions[i]->getInvestment();

                // Remove the position from the list of open positions, and free memory
                // by deleting the position.
                delete this->openPositions[i];
                this->openPositions[i] = nullptr;

                // Remove the position's investment amount from the total profit/loss for this strategy.
                this->profitLoss -= positionInvestment;

                // Add the profit/loss for this position to the profit/loss for this strategy.
                this->profitLoss += positionProfitLoss;

                if (positionProfitLoss > positionInvestment) {
                    this->winCount++;
                    this->consecutiveLosses = 0;
                }
                if (positionProfitLoss == 0) {
                    this->loseCount++;
                    this->consecutiveLosses++;
                }

                // Track minimum profit/loss.
                if (this->profitLoss < this->minimumProfitLoss) {
                    this->minimumProfitLoss = this->profitLoss;
                }

                // Update the minimum profit/loss for this strategy if applicable.
                if (this->consecutiveLosses > this->maximumConsecutiveLosses) {
                    this->maximumConsecutiveLosses = this->consecutiveLosses;
                }
            }
        }
    }
}

__device__ __host__ StrategyResult Strategy::getResult() {
    StrategyResult result;

    result.profitLoss = this->profitLoss;
    result.winCount = this->winCount;
    result.loseCount = this->loseCount;
    result.winRate = getWinRate();
    result.tradeCount = this->winCount + this->loseCount;
    result.maximumConsecutiveLosses = this->maximumConsecutiveLosses;
    result.minimumProfitLoss = this->minimumProfitLoss;

    return result;
}

__device__ __host__ void Strategy::addPosition(Position *position) {
    for (int i=0; i<10; i++) {
        // If there is an unused position slot, then use it.
        if (this->openPositions[i] == nullptr) {
            this->openPositions[i] = position;
            break;
        }
    }
}
