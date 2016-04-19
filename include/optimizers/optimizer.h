#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <stdio.h>
#include <string>
#include <vector>
#include <map>
#include "studies/study.h"
#include "types/tick.h"
#include "types/configuration.h"

class Optimizer {
    private:
        std::string strategyName;
        std::string symbol;
        int group;

    protected:
        std::vector<Study*> studies;

    public:
        Optimizer(std::string strategyName, std::string symbol, int group);
        virtual void prepareStudies() = 0;
        void prepareData(std::vector<Tick*> data);
        void optimize(std::vector<Configuration>, double investment, double profitability);
        std::vector<Configuration> buildConfigurations();
};

#endif
