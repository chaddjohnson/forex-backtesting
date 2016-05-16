#ifndef COMBINEDSTRATEGY_H
#define COMBINEDSTRATEGY_H

#include <vector>
#include <string>
#include "strategy.cuh"

class CombinedStrategy : public Strategy {
    private:
        std::vector<Configuration*> configurations;
        double *tickPreviousDataPoint;

    protected:
        void tick(double *dataPoint);
        std::vector<Configuration*> getConfigurations();

    public:
        CombinedStrategy(std::string symbol, std::map<std::string, int> *dataIndex, std::vector<Configuration*> configurations);
        ~CombinedStrategy();
};

#endif
