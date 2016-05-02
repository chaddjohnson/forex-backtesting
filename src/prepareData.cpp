#include <string>
#include <vector>
#include <map>
#include <iterator>
#include <iostream>
#include <mongoc.h>
#include "dataParsers/dataParser.h"
#include "optimizers/optimizer.h"
#include "factories/dataParserFactory.h"
#include "factories/optimizerFactory.h"

int main(int argc, char *argv[]) {
    // Data parser settings and objects.
    std::string dataParserName = "oanda";
    std::string dataFilePath = "/Users/chad/development/desktop/forex-backtesting/data/oanda/k-fold/combined/AUDJPY.csv";
    DataParser *dataParser;
    std::vector<Tick*> ticks;

    // Optimizer settings.
    std::string optimizerName = "reversals";
    std::string symbol = "AUDJPY";
    int group = 1;

    // Connect to the database
    mongoc_init();
    mongoc_client_t *dbClient = mongoc_client_new("mongodb://localhost:27017");

    // Parse the data file.
    dataParser = DataParserFactory::create(dataParserName, dataFilePath);
    ticks = dataParser->parse();

    // Initialize the optimizer.
    Optimizer *optimizer = OptimizerFactory::create(optimizerName, dbClient, symbol, group);

    // Prepare the data.
    optimizer->prepareData(ticks);

    // Clean up.
    // TODO

    return 0;
}
