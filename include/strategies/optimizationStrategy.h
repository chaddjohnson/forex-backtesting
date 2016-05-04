#ifndef OPTIMIZATIONSTRATEGY_H
#define OPTIMIZATIONSTRATEGY_H

#include <string>
#include "strategy.h"
#include "types/configuration.h"

class OptimizationStrategy : public Strategy {
    private:
        int group;
        Configuration *configuration;
        double *tickPreviousDataPoint;

    protected:
        void tick(double *dataPoint);

    public:
        OptimizationStrategy(std::string symbol, int group, Configuration *configuration);
        ~OptimizationStrategy();
        int getGroup();
        Configuration *getConfiguration();
};

#endif
