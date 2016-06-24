#ifndef STRATEGYRESULT_H
#define STRATEGYRESULT_H

typedef struct StrategyResult {
    // Resulting stats
    float profitLoss;
    int winCount;
    int loseCount;
    int tradeCount;
    float winRate;
    int maximumConsecutiveLosses;
    int minimumProfitLoss;

    // Configuration used
    Configuration configuration;

    __device__ __host__ StrategyResult() {
        configuration = {};
    }
} StrategyResult;

#endif
