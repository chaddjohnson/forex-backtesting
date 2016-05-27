#ifndef STRATEGYRESULTS_H
#define STRATEGYRESULTS_H

typedef struct StrategyResults {
    double profitLoss;
    int winCount;
    int loseCount;
    double winRate;
    int tradeCount;
    int maximumConsecutiveLosses;
    int minimumProfitLoss;
} StrategyResults;

#endif
