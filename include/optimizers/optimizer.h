#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <vector>
#include <string>
#include <studies/study.h>

class Optimizer {
    private:
        std::string strategyName;
        std::string symbol;
        int group;

    protected:
        std::vector<Study*> studies;

    public:
        Optimizer(std::string strategyName, std::string symbol, int group) {}
        virtual void prepareStudies() = 0;
        void optimize(std::string strategyName, std::string symbol, int group);
        void buildConfigurations();
};

#endif
