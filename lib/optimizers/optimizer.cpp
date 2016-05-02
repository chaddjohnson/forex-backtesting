#include "optimizers/optimizer.h"

Optimizer::Optimizer(mongoc_client_t *dbClient, std::string strategyName, std::string symbol, int group) {
    this->dbClient = dbClient;
    this->strategyName = strategyName;
    this->symbol = symbol;
    this->group = group;
    this->dataCount = 0;

    // Prepare studies for use.
    prepareStudies();
}

void Optimizer::prepareData(std::vector<Tick*> ticks) {
    double percentage;
    int tickCount = ticks.size();
    std::vector<Tick*> cumulativeTicks;
    std::vector<Tick*> tempcumulativeTicks;
    int threadCount = std::thread::hardware_concurrency();
    int i = 0;
    int j = 0;
    maginatics::ThreadPool pool(1, threadCount, 5000);

    // If there is a significant gap, save the current data points, and start over with recording.
    // TODO

    // Reserve space in advance for better performance
    cumulativeTicks.reserve(tickCount);

    printf("Preparing data...");

    // Go through the data and run studies for each data item.
    for (std::vector<Tick*>::iterator dataIterator = ticks.begin(); dataIterator != ticks.end(); ++dataIterator) {
        percentage = (++i / (double)tickCount) * 100.0;
        printf("\rPreparing data...%0.4f%%", percentage);

        // Append to the cumulative data.
        cumulativeTicks.push_back(*dataIterator);

        for (std::vector<Study*>::iterator studyIterator = studies.begin(); studyIterator != studies.end(); ++studyIterator) {
            // Update the data for the study.
            (*studyIterator)->setData(&cumulativeTicks);

            // Use a thread pool so that all CPU cores can be used.
            pool.execute([&]() {
                // Source: http://stackoverflow.com/a/7854596/83897
                auto functor = [=]() {
                    // Process the latest data for the study.
                    (*studyIterator)->tick();
                };
            });
        }

        // Block until all tasks for the current data point to complete.
        pool.drain();

        // Periodically free up memory.
        if (cumulativeTicks.size() >= 2000) {
            for (j=0; j<1000; j++) {
                delete cumulativeTicks[j];
            }

            // Extract the last 1000 elements into a new vector.
            std::vector<Tick*> tempcumulativeTicks(cumulativeTicks.begin() + (cumulativeTicks.size() - 1000), cumulativeTicks.end());

            // Release memory for the old vector.
            std::vector<Tick*>().swap(cumulativeTicks);
            //cumulativeTicks.shrink_to_fit();

            // Set the original to be the new vector.
            cumulativeTicks = tempcumulativeTicks;
        }
    }

    printf("\n");
}

std::vector<Configuration*> Optimizer::buildConfigurations() {
    std::vector<Configuration*> configurations;



    // Built a flat key/value list of configurations.
    // ...

    return configurations;
}

// double *Optimizer::convertTickToArray(Tick *tick) {
//     double *convertedTick = (double*) malloc(getStudies.size() * sizeof(double));
//     int index;

//     for (Tick::iterator iterator = tick.begin(); iterator != tick.end(); ++iterator) {
//         // Get the current property index.
//         index = std::distance(tick.begin(), iterator);

//         // Add the current property value to the array.
//         convertedTick[index] = iterator->second;
//     }

//     return convertedTick;
// }

int Optimizer::getDataPropertyCount() {
    int count = 0;

    for (std::vector<Study*>::iterator iterator = this->studies.begin(); iterator != this->studies.end(); ++iterator) {
        count += (*iterator)->getOutputMap().size();
    }

    return 0;
}

void Optimizer::loadData() {
    int propertyIndex = 0;
    mongoc_collection_t *collection;
    bson_t *countQuery;
    bson_t *query;
    mongoc_cursor_t *cursor;
    const bson_t *document;
    bson_iter_t iterator;
    bson_iter_t close;
    bson_error_t error;

    // Get a reference to the database collection.
    collection = mongoc_client_get_collection(this->dbClient, "forex-backtesting", "datapoints");

    // Query for the number of data points.
    countQuery = BCON_NEW("$query", "{", "symbol", BCON_UTF8(this->symbol), "}");
    this->dataCount = mongoc_collection_count(collection, MONGOC_QUERY_NONE, countQuery, 0, 0, NULL, &error);

    if (this->dataCount < 0) {
        // No data points found.
        throw std::exception(error.message);
    }

    // Allocate memory for the data.
    this->data = (double** data) malloc(this->dataCount * sizeof(double*));

    // Query the database.
    query = BCON_NEW(
        "$query", "{", "symbol", BCON_UTF8(this->symbol), "}",
        "$orderby", "{", "data.timestamp", BCON_INT32(1) "}",
        "$hint", "{", "data.timestamp", BCON_INT32(1), "}"
    );
    cursor = mongoc_collection_find(collection, MONGOC_QUERY_NONE, 0, 0, 1000, query, NULL, NULL);

    // Go through query results, and convert each document into an array.
    while (mongoc_cursor_next(cursor, &document)) {
        propertyIndex = 0;

        // Allocate memory for the data point.
        this->data[this->dataCount] = (double*) malloc(this->getDataPropertyCount() * sizeof(double));

        if (bson_iter_init(&iterator, document)) {
            // TODO
            bson_iter_find_descendant(&iterator, "data.?", &close);
            // ...

            this->data[this->dataCount][propertyIndex] = bson_iter_double(&value);
            propertyIndex++;
        }

        // Keep track of the number of ticks.
        this->dataCount++;
    }

    // Cleanup.
    bson_destroy(query);
    mongoc_cursor_destroy(cursor);
    mongoc_collection_destroy(collection);
}

void Optimizer::optimize(std::vector<Configuration*> configurations, double investment, double profitability) {
    int threadCount = std::thread::hardware_concurrency();
    std::vector<Strategy*> strategies;
    int i = 0;

    // Load tick data from the database.
    loadData();

    // Set up a threadpool so all CPU cores and their threads can be used.
    maginatics::ThreadPool pool(1, threadCount, 5000);

    // Set up one strategy instance per configuration.
    for (std::vector<Configuration*>::iterator iterator = configurations.begin(); iterator != configurations.end(); ++iterator) {
        i = std::distance(configurations.begin(), iterator);
        strategies[i] = StrategyFactory::create(this->strategyName, this->symbol, this->group, *iterator);
    }

    // Iterate over data ticks.
    for (i=0; i<this->dataCount; i++) {
        // Loop through all strategies/configurations.
        for (std::vector<Strategy*>::iterator strategyIterator = strategies.begin(); strategyIterator != strategies.end(); ++ strategyIterator) {
            // Use a thread pool so that all CPU cores can be used.
            pool.execute([&]() {
                // Source: http://stackoverflow.com/a/7854596/83897
                auto functor = [=]() {
                    // Process the latest data for the study.
                    (*strategyIterator)->backtest(this->data[i], investment, profitability);
                };
            });
        }

        // Block until all tasks for the current data point to complete.
        pool.drain();
    }

    // Unload data.
    // TODO
}
