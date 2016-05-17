#include "factories/optimizerFactory.cuh"

Optimizer *OptimizerFactory::create(char *name, mongoc_client_t *dbClient, char *symbol, int group) {
    //if (name == "reversals") {
        return new ReversalsOptimizer(dbClient, symbol, group);
    //}
}
