#ifndef STRATEGYRESULT_H
#define STRATEGYRESULT_H

typedef struct StrategyResult {
    // Resulting stats
    double profitLoss;
    int winCount;
    int loseCount;
    int tradeCount;
    double winRate;
    int maximumConsecutiveLosses;
    int minimumProfitLoss;

    // Configuration used
    Configuration *configuration;
} StrategyResult;

#endif
