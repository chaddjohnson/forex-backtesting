#include "factories/optimizerFactory.cuh"

Optimizer *OptimizerFactory::create(std::string name, mongoc_client_t *dbClient, std::string symbol, int group) {
    //if (name == "reversals") {
        return new ReversalsOptimizer(dbClient, symbol, group);
    //}
}
