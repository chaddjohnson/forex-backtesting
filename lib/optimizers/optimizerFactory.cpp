#include "optimizers/optimizerFactory.h"

Optimizer *OptimizerFactory::create(std::string name, std::string symbol, int group) {
    //if (name == "reversals" || name == "Reversals") {
        return new ReversalsOptimizer(symbol, group);
    //}
}
