#include "optimizers/optimizerFactory.h"

Optimizer *OptimizerFactory::create(std::string name, std::string symbol, int group) {
    //if (name == "reversals") {
        return new ReversalsOptimizer(symbol, group);
    //}
}
