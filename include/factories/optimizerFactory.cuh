#ifndef OPTIMIZERFACTORY_H
#define OPTIMIZERFACTORY_H

#include <string>
#include <mongoc.h>
#include "optimizers/optimizer.cuh"
#include "optimizers/reversalsOptimizer.cuh"

class OptimizerFactory {
    public:
        static Optimizer *create(std::string name, mongoc_client_t *dbClient, std::string symbol, int group = 0);
};

#endif
