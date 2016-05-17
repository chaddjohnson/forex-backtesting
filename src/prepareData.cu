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
    // Data parser settings and objects.
    const char *dataParserName = "oanda";
    std::string dataFilePath = "/Users/chad/development/desktop/forex-backtesting/data/oanda/k-fold/combined/AUDJPY.csv";
    DataParser *dataParser;
    Optimizer *optimizer;
    std::vector<Tick*> ticks;

    // Optimizer settings.
    const char *optimizerName = "reversals";
    const char *symbol = "AUDJPY";
    int group = 1;

    // Connect to the database
    mongoc_init();
    mongoc_client_t *dbClient = mongoc_client_new("mongodb://localhost:27017");

    // Parse the data file.
    dataParser = DataParserFactory::create(dataParserName, dataFilePath);
    ticks = dataParser->parse();

    // Initialize the optimizer.
    optimizer = OptimizerFactory::create(optimizerName, dbClient, symbol, group);

    // Prepare the data.
    optimizer->prepareData(ticks);

    // Clean up.
    // TODO
    delete dataParser;
    // delete optimizer;
    mongoc_cleanup();

    return 0;
}
