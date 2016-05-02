#ifndef REVERSALSSTRATEGY_H
#define REVERSALSSTRATEGY_H

#include <cstdlib>
#include <string>
#include "positions/callPosition.h"
#include "positions/putPosition.h"

class ReversalsStrategy : public Strategy {
    private:
        Configuration configuration;
        Tick *previousTick;
        bool putNextTick;
        bool callNextTick;
        int expirationMinutes;

    public:
        ReversalsStrategy(std::string symbol, int group, Configuration *configuration);
        void backtest(double *dataPoint, double investment, double profitability);
};

#endif
