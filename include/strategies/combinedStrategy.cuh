#ifndef COMBINEDSTRATEGY_H
#define COMBINEDSTRATEGY_H

#include <vector>
#include <string>
#include "strategy.cuh"
#include "types/basicDataIndexMap.cuh"

class CombinedStrategy : public Strategy {
    private:
        std::vector<Configuration*> configurations;
        double *tickPreviousDataPoint;

    protected:
        void tick(double *dataPoint);
        std::vector<Configuration*> getConfigurations();

    public:
        CombinedStrategy(const char *symbol, BasicDataIndexMap dataIndexMap, std::vector<Configuration*> configurations);
        ~CombinedStrategy();
};

#endif
