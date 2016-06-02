#include <string>
#include <iostream>
#include <mongoc.h>
#include <vector>
#include "optimizers/optimizer.cuh"
#include "factories/optimizerFactory.cuh"
#include "types/configuration.cuh"

int main(int argc, char *argv[]) {
    // Optimizer settings and objects.
    const char *optimizerName = "reversals";
    const char *symbol = "AUDJPY";
    double investment = 1000.0;
    double profitability = 0.76;
    int group = 1;
    std::vector<Configuration*> configurations;
    int returnValue = 0;

    // Connect to the database
    mongoc_init();
    mongoc_client_t *dbClient = mongoc_client_new("mongodb://localhost:27017");

    Optimizer *optimizer = OptimizerFactory::create(optimizerName, dbClient, symbol, group);

    // Perform optimization.
    try {
        configurations = optimizer->buildConfigurations(optimizer->getConfigurationOptions());
        optimizer->optimize(configurations, investment, profitability);
    }
    catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        returnValue = 1;
    }

    // Clean up.
    mongoc_cleanup();

    return returnValue;
}
