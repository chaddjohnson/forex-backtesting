#ifndef OPTIMIZATIONSTRATEGY_H
#define OPTIMIZATIONSTRATEGY_H

#include <string>
#include "strategy.cuh"
#include "types/configuration.cuh"
#include "types/basicDataIndexMap.cuh"

class OptimizationStrategy : public Strategy {
    private:
        int group;
        Configuration *configuration;
        double *tickPreviousDataPoint;

    protected:
        void tick(double *dataPoint);

    public:
        OptimizationStrategy(const char *symbol, BasicDataIndexMap dataIndexMap, int group, Configuration *configuration);
        ~OptimizationStrategy();
        int getGroup();
        Configuration *getConfiguration();
};

#endif
