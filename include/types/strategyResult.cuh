#ifndef STRATEGYRESULT_H
#define STRATEGYRESULT_H

#include "types/real.cuh"

typedef struct StrategyResult {
    // Resulting stats
    Real profitLoss;
    int winCount;
    int loseCount;
    int tradeCount;
    Real winRate;
    int maximumConsecutiveLosses;
    int minimumProfitLoss;

    // Configuration used
    Configuration *configuration;
} StrategyResult;

#endif
