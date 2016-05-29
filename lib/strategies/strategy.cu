#include "strategies/strategy.cuh"

__host__ Strategy::Strategy(const char *symbol, BasicDataIndexMap dataIndexMap) {
    this->symbol = symbol;
    this->dataIndexMap = dataIndexMap;
    this->profitLoss = 0.0;
    this->winCount = 0;
    this->loseCount = 0;
    this->consecutiveLosses = 0;
    this->maximumConsecutiveLosses = 0;
    this->minimumProfitLoss = 99999.0;
}

__device__ BasicDataIndexMap Strategy::getDataIndexMap() {
    return this->dataIndexMap;
}

__device__ const char *Strategy::getSymbol() {
    return this->symbol;
}

__device__ double Strategy::getProfitLoss() {
    return this->profitLoss;
}

__device__ void Strategy::setProfitLoss(double profitLoss) {
    this->profitLoss = profitLoss;
}

__device__ double Strategy::getWinRate() {
    if (this->winCount + this->loseCount == 0) {
        return 0;
    }
    return this->winCount / (this->winCount + this->loseCount);
}

__device__ void Strategy::closeExpiredPositions(double price, time_t timestamp) {
    // // Create a copy of the vector of open positions.
    // std::vector<Position*> tempOpenPositions(this->openPositions);

    // for (std::vector<Position*>::iterator positionIterator = tempOpenPositions.begin(); positionIterator != tempOpenPositions.end(); ++positionIterator) {
    //     double positionProfitLoss = 0.0;

    //     if ((*positionIterator)->getHasExpired(timestamp)) {
    //         // Close the position since it is open and has expired.
    //         (*positionIterator)->close(price, timestamp);

    //         // Remove the position's investment amount from the total profit/loss for this strategy.
    //         this->profitLoss -= (*positionIterator)->getInvestment();

    //         // Add the profit/loss for this position to the profit/loss for this strategy.
    //         positionProfitLoss = (*positionIterator)->getProfitLoss();
    //         this->profitLoss += positionProfitLoss;

    //         if (positionProfitLoss > (*positionIterator)->getInvestment()) {
    //             this->winCount++;
    //             this->consecutiveLosses = 0;
    //         }
    //         if (profitLoss == 0) {
    //             this->loseCount++;
    //             this->consecutiveLosses++;
    //         }

    //         // Update the minimum profit/loss for this strategy if applicable.
    //         if (this->consecutiveLosses > this->maximumConsecutiveLosses) {
    //             this->maximumConsecutiveLosses = this->consecutiveLosses;
    //         }

    //         // Remove the position from the list of open positions, and free memory.
    //         this->openPositions.erase(std::remove(this->openPositions.begin(), this->openPositions.end(), *positionIterator), this->openPositions.end());
    //         delete *positionIterator;
    //         *positionIterator = nullptr;
    //     }
    // }
}

__device__ StrategyResults Strategy::getResults() {
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

__device__ void Strategy::addPosition(Position *position) {
    // this->openPositions.push_back(position);
}
