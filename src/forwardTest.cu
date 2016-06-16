#include <string>
#include <vector>
#include <iostream>
#include <mongoc.h>
#include "dataParsers/dataParser.cuh"
#include "optimizers/optimizer.cuh"
#include "factories/dataParserFactory.cuh"
#include "factories/optimizerFactory.cuh"
#include "types/configuration.cuh"

int main(int argc, char *argv[]) {
    std::vector<Tick*> ticks;
    int returnValue = 0;
    int i = 0;

    // Settings
    std::string symbol;
    std::string filePath;
    std::string parserName;
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
                std::cerr << "--symbol option requires one argument." << std::endl;
                return 1;
            }
        }
        if (arg == "--parser") {
            if (i + 1 < argc) {
                parserName = std::string(argv[i + 1]);
            }
            else {
                std::cerr << "--parser option requires one argument." << std::endl;
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
        if (arg == "--file") {
            if (i + 1 < argc) {
                filePath = std::string(argv[i + 1]);
            }
            else {
                std::cerr << "--file option requires one argument." << std::endl;
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
        // Parse the data file.
        // DataParser *dataParser = DataParserFactory::create(parserName, filePath, DataParser::types::FORWARDTEST);
        // ticks = dataParser->parse();

        // Initialize the optimizer.
        Optimizer *optimizer = OptimizerFactory::create(optimizerName, dbClient, symbol, Optimizer::types::FORWARDTEST);

        // Prepare the data.
        // optimizer->prepareData(ticks);

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
