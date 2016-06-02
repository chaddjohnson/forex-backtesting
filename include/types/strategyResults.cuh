#ifndef STRATEGYRESULTS_H
#define STRATEGYRESULTS_H

typedef struct StrategyResults {
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
} StrategyResults;

#endif
