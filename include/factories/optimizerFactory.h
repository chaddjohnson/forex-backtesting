#ifndef OPTIMIZERFACTORY_H
#define OPTIMIZERFACTORY_H

#include <string>
#include "optimizers/optimizer.h"
#include "optimizers/reversalsOptimizer.h"

class OptimizerFactory {
    public:
        static Optimizer *create(std::string name, std::string symbol, int group);
};

#endif
