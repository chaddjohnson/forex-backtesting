#ifndef REVERSALSOPTIMIZATIONSTRATEGY_H
#define REVERSALSOPTIMIZATIONSTRATEGY_H

#include <cstdlib>
#include <string>
#include <ctime>
#include "optimizationStrategy.h"
#include "positions/callPosition.h"
#include "positions/putPosition.h"
#include "types/configuration.h"

class ReversalsOptimizationStrategy : public OptimizationStrategy {
    private:
        Configuration *configuration;
        double *previousDataPoint;
        bool putNextTick;
        bool callNextTick;
        int expirationMinutes;

    public:
        ReversalsOptimizationStrategy(std::string symbol, int group, Configuration *configuration);
        ~ReversalsOptimizationStrategy();
        void backtest(double *dataPoint, double investment, double profitability);
};

#endif
