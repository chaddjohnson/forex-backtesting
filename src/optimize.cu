#include <string>
#include <iostream>
#include <mongoc.h>
#include <thrust/host_vector.h>
#include "optimizers/optimizer.cuh"
#include "factories/optimizerFactory.cuh"
#include "types/configuration.cuh"

int main(int argc, char *argv[]) {
    // Optimizer settings and objects.
    char *optimizerName = "reversals";
    char *symbol = "AUDJPY";
    int group = 1;
    Optimizer *optimizer;
    thrust::host_vector<Configuration*> configurations;

    // Connect to the database
    mongoc_init();
    mongoc_client_t *dbClient = mongoc_client_new("mongodb://localhost:27017");

    // Perform optimization.
    try {
        optimizer = OptimizerFactory::create(optimizerName, dbClient, symbol, group);
        optimizer->loadData();
        configurations = optimizer->buildConfigurations(optimizer->getConfigurationOptions());
        optimizer->optimize(configurations, 1000.0, 0.76);
    }
    catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
    }

    // Clean up.
    // TODO
    // delete optimizer;
    mongoc_cleanup();

    return 0;
}
