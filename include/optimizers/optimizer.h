#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <stdio.h>
#include <string>
#include <vector>
#include <map>
#include "studies/study.h"
#include "types/tick.h"
#include "types/configuration.h"
#include "maginatics/threadpool/threadpool.h"

class Optimizer {
    private:
        std::string strategyName;
        std::string symbol;
        int group;

    protected:
        std::vector<Study*> studies;
        virtual void prepareStudies() = 0;

    public:
        Optimizer(std::string strategyName, std::string symbol, int group);
        void prepareData(std::vector<Tick*> data);
        std::vector<Configuration> buildConfigurations();
        void optimize(std::vector<Configuration>, double investment, double profitability);
};

#endif
