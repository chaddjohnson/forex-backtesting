#ifndef STRATEGY_H
#define STRATEGY_H

#include <string>
#include "types/configuration.h"

class Strategy {
    public:
        Strategy(std::string symbol, Configuration *configuration);
        virtual void backtest(double *dataPoint, double investment, double profitability) = 0;
};

#endif
