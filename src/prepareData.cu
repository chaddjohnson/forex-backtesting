#include <string>
#include <vector>
#include <map>
#include <iterator>
#include <iostream>
#include <mongoc.h>
#include "dataParsers/dataParser.cuh"
#include "optimizers/optimizer.cuh"
#include "factories/dataParserFactory.cuh"
#include "factories/optimizerFactory.cuh"

int main(int argc, char *argv[]) {
    std::vector<Tick*> ticks;
    int i = 0;

    // Settings
    std::string symbol;
    std::string filePath;
    std::string parserName;
    std::string optimizerName;

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

    if (argc < 8) {
        std::cerr << "Too few arguments provided." << std::endl;
        return 0;
    }

    // Connect to the database
    mongoc_init();
    mongoc_client_t *dbClient = mongoc_client_new("mongodb://localhost:27017");

    // Parse the data file.
    DataParser *dataParser = DataParserFactory::create(parserName, filePath);
    ticks = dataParser->parse();

    // Initialize the optimizer.
    Optimizer *optimizer = OptimizerFactory::create(optimizerName, dbClient, symbol);

    // Prepare the data.
    optimizer->prepareData(ticks);

    // Clean up.
    mongoc_cleanup();

    return 0;
}
