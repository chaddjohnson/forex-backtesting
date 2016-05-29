#ifndef OPTIMIZERFACTORY_H
#define OPTIMIZERFACTORY_H

#include <string>
#include <mongoc.h>
#include "optimizers/optimizer.cuh"
#include "optimizers/reversalsOptimizer.cuh"

class OptimizerFactory {
    public:
        static Optimizer create(const char *name, mongoc_client_t *dbClient, const char *symbol, int group);
};

#endif
