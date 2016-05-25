#ifndef REVERSALSOPTIMIZATIONSTRATEGY_H
#define REVERSALSOPTIMIZATIONSTRATEGY_H

#include <cstdlib>
#include <string>
#include <ctime>
#include "optimizationStrategy.cuh"
#include "positions/callPosition.cuh"
#include "positions/putPosition.cuh"
#include "types/configuration.cuh"

class ReversalsOptimizationStrategy : public OptimizationStrategy {
    private:
        Configuration *configuration;
        double *previousDataPoint;
        bool putNextTick;
        bool callNextTick;
        int expirationMinutes;

    public:
        ReversalsOptimizationStrategy(const char *symbol, std::map<std::string, int> *dataIndexMap, int group, Configuration *configuration);
        ~ReversalsOptimizationStrategy();
        void backtest(double *dataPoint, double investment, double profitability);
};

#endif
