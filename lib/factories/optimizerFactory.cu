#include "factories/optimizerFactory.cuh"

Optimizer OptimizerFactory::create(const char *name, mongoc_client_t *dbClient, const char *symbol, int group) {
    //if (name == "reversals") {
        return ReversalsOptimizer::ReversalsOptimizer(dbClient, symbol, group);
    //}
}
