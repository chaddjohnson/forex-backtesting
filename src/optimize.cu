#include <string>
#include <vector>
#include <iostream>
#include <mongoc.h>
#include "optimizers/optimizer.cuh"
#include "factories/optimizerFactory.cuh"
#include "types/configuration.cuh"

int main(int argc, char *argv[]) {
    int returnValue = 0;
    int i = 0;

    // Settings
    std::string symbol;
    std::string type;
    int group;
    std::string optimizerName;
    float investment;
    float profitability;

    // Parse command line arguments.
    for (i=0; i<argc; i++) {
        std::string arg = std::string(argv[i]);

        if (arg == "--symbol") {
            if (i + 1 < argc) {
                symbol = std::string(argv[i + 1]);
            }
            else {
                std::cerr << "--symbol option requires one argument." << std::endl;
                return 1;
            }
        }
        if (arg == "--type") {
            if (i + 1 < argc) {
                type = std::string(argv[i + 1]);
            }
            else {
                std::cerr << "--type option requires one argument." << std::endl;
                return 1;
            }
        }
        if (arg == "--group") {
            if (i + 1 < argc) {
                group = atoi(argv[i + 1]);
            }
            else {
                std::cerr << "--group option requires one argument." << std::endl;
                return 1;
            }
        }
        if (arg == "--optimizer") {
            if (i + 1 < argc) {
                optimizerName = std::string(argv[i + 1]);
            }
            else {
                std::cerr << "--optimizer option requires one argument." << std::endl;
                return 1;
            }
        }
        if (arg == "--investment") {
            if (i + 1 < argc) {
                investment = atof(argv[i + 1]);
            }
            else {
                std::cerr << "--investment option requires one argument." << std::endl;
                return 1;
            }
        }
        if (arg == "--profitability") {
            if (i + 1 < argc) {
                profitability = atof(argv[i + 1]);
            }
            else {
                std::cerr << "--profitability option requires one argument." << std::endl;
                return 1;
            }
        }
    }

    if (argc < 12) {
        std::cerr << "Too few arguments provided." << std::endl;
        return 0;
    }

    // Connect to the database
    mongoc_init();
    mongoc_client_t *dbClient = mongoc_client_new("mongodb://localhost:27017");

    try {
        // Initialize the optimizer.
        Optimizer *optimizer = OptimizerFactory::create(optimizerName, dbClient, symbol, Optimizer::getTypeId(type), group);

        // Perform optimization.
        optimizer->optimize(investment, profitability);
    }
    catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        returnValue = 1;
    }

    // Clean up.
    mongoc_client_destroy(dbClient);
    mongoc_cleanup();

    return returnValue;
}
