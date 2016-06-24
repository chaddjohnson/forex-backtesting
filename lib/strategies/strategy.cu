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
}

__device__ __host__ const char *Strategy::getSymbol() {
    return this->symbol;
}

__device__ __host__ float Strategy::getProfitLoss() {
    return this->profitLoss;
}

__device__ __host__ void Strategy::setProfitLoss(float profitLoss) {
    this->profitLoss = profitLoss;
}

__device__ __host__ float Strategy::getWinRate() {
    if (this->winCount + this->loseCount == 0) {
        return 0;
    }
    return (float)this->winCount / ((float)this->winCount + (float)this->loseCount);
}

__device__ __host__ void Strategy::tick(float close, int timestamp) {
    // Simulate expiry of and profit/loss related to positions held.
    closeExpiredPutPositions(close, timestamp);
    closeExpiredCallPositions(close, timestamp);
}

__device__ __host__ void Strategy::closeExpiredPutPositions(float price, int timestamp) {
    for (int i=0; i<5; i++) {
        if (this->openPutPositions[i].getIsOpen() && this->openPutPositions[i].getHasExpired(timestamp)) {
            // Close the position since it is open and has expired.
            this->openPutPositions[i].close(price, timestamp);

            float positionProfitLoss = this->openPutPositions[i].getProfitLoss();
            float positionInvestment = this->openPutPositions[i].getInvestment();

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

__device__ __host__ void Strategy::closeExpiredCallPositions(float price, int timestamp) {
    for (int i=0; i<5; i++) {
        if (this->openCallPositions[i].getIsOpen() && this->openCallPositions[i].getHasExpired(timestamp)) {
            // Close the position since it is open and has expired.
            this->openCallPositions[i].close(price, timestamp);

            float positionProfitLoss = this->openCallPositions[i].getProfitLoss();
            float positionInvestment = this->openCallPositions[i].getInvestment();

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

__device__ __host__ StrategyResult Strategy::getResult() {
    StrategyResult result = {};

    result.profitLoss = this->profitLoss;
    result.winCount = this->winCount;
    result.loseCount = this->loseCount;
    result.winRate = getWinRate();
    result.tradeCount = this->winCount + this->loseCount;
    result.maximumConsecutiveLosses = this->maximumConsecutiveLosses;
    result.minimumProfitLoss = this->minimumProfitLoss;

    return result;
}

__device__ __host__ void Strategy::addPutPosition(const char *symbol, int timestamp, float close, float investment, float profitability, float expirationMinutes) {
    for (int i=0; i<5; i++) {
        // If there is an unused position slot, then use it.
        if (!this->openPutPositions[i].getIsOpen()) {
            this->openPutPositions[i] = PutPosition(symbol, timestamp, close, investment, profitability, expirationMinutes);
            break;
        }
    }
}

__device__ __host__ void Strategy::addCallPosition(const char *symbol, int timestamp, float close, float investment, float profitability, float expirationMinutes) {
    for (int i=0; i<5; i++) {
        // If there is an unused position slot, then use it.
        if (!this->openCallPositions[i].getIsOpen()) {
            this->openCallPositions[i] = CallPosition(symbol, timestamp, close, investment, profitability, expirationMinutes);
            break;
        }
    }
}
