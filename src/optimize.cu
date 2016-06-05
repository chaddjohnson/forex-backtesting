#include <string>
#include <iostream>
#include <cstdlib>
#include <mongoc.h>
#include <vector>
#include "optimizers/optimizer.cuh"
#include "factories/optimizerFactory.cuh"
#include "types/configuration.cuh"

int main(int argc, char *argv[]) {
    std::vector<Configuration*> configurations;
    int returnValue = 0;
    int i = 0;

    // Settings
    std::string symbol;
    std::string type;
    int group;
    std::string optimizerName;
    double investment;
    double profitability;

    // Parse command line arguments.
    for (i=0; i<argc; i++) {
        std::string arg = std::string(argv[i]);

        if (arg == "--symbol") {
            if (i + 1 < argc) {
                symbol = std::string(argv[i + 1]);
            }
            else {
                std::cerr << "--symbol option requires one argument.";
                return 1;
            }
        }
        if (arg == "--type") {
            if (i + 1 < argc) {
                type = std::string(argv[i + 1]);
            }
            else {
                std::cerr << "--type option requires one argument.";
                return 1;
            }
        }
        if (arg == "--group") {
            if (i + 1 < argc) {
                group = atoi(argv[i + 1]);
            }
            else {
                std::cerr << "--group option requires one argument.";
                return 1;
            }
        }
        if (arg == "--optimizer") {
            if (i + 1 < argc) {
                optimizerName = std::string(argv[i + 1]);
            }
            else {
                std::cerr << "--optimizer option requires one argument.";
                return 1;
            }
        }
        if (arg == "--investment") {
            if (i + 1 < argc) {
                investment = atof(argv[i + 1]);
            }
            else {
                std::cerr << "--investment option requires one argument.";
                return 1;
            }
        }
        if (arg == "--profitability") {
            if (i + 1 < argc) {
                profitability = atof(argv[i + 1]);
            }
            else {
                std::cerr << "--profitability option requires one argument.";
                return 1;
            }
        }
    }

    // Connect to the database
    mongoc_init();
    mongoc_client_t *dbClient = mongoc_client_new("mongodb://localhost:27017");

    Optimizer *optimizer = OptimizerFactory::create(optimizerName, dbClient, symbol, group);
    optimizer->setType(type);

    // Perform optimization.
    try {
        optimizer->optimize(investment, profitability);
    }
    catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        returnValue = 1;
    }

    // Clean up.
    mongoc_cleanup();

    return returnValue;
}
