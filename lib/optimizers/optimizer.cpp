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

Optimizer::~Optimizer() {
    free(data);
}

bson_t *Optimizer::convertTickToBson(Tick *tick) {
    bson_t *document;
    bson_t dataDocument;

    document = bson_new();
    BSON_APPEND_UTF8(document, "symbol", this->symbol.c_str());
    BSON_APPEND_DOCUMENT_BEGIN(document, "data", &dataDocument);

    // Add tick properties to document.
    for (Tick::iterator propertyIterator = tick->begin(); propertyIterator != tick->end(); ++propertyIterator) {
        bson_append_double(&dataDocument, propertyIterator->first.c_str(), propertyIterator->first.length(), propertyIterator->second);
    }

    bson_append_document_end(document, &dataDocument);

    return document;
}

void Optimizer::saveTicks(std::vector<Tick*> ticks) {
    mongoc_collection_t *collection;
    mongoc_bulk_operation_t *bulkOperation;
    bson_t *document;
    bson_t bulkOperationReply;
    bson_error_t bulkOperationError;

    // Get a reference to the database collection.
    collection = mongoc_client_get_collection(this->dbClient, "forex-backtesting", "datapoints");

    // Begin a bulk operation.
    bulkOperation = mongoc_collection_create_bulk_operation(collection, true, NULL);

    // Reference: http://api.mongodb.org/c/current/bulk.html
    for (std::vector<Tick*>::iterator insertionIterator = ticks.begin(); insertionIterator != ticks.end(); ++insertionIterator) {
        document = convertTickToBson(*insertionIterator);
        mongoc_bulk_operation_insert(bulkOperation, document);
        bson_destroy(document);
    }

    // Execute the bulk operation.
    mongoc_bulk_operation_execute(bulkOperation, &bulkOperationReply, &bulkOperationError);

    // Cleanup.
    mongoc_collection_destroy(collection);
    mongoc_bulk_operation_destroy(bulkOperation);
    bson_destroy(&bulkOperationReply);
    delete document;
}

void Optimizer::prepareData(std::vector<Tick*> ticks) {
    double percentage;
    int tickCount = ticks.size();
    std::vector<Tick*> cumulativeTicks;
    int threadCount = std::thread::hardware_concurrency();
    maginatics::ThreadPool pool(1, threadCount, 5000);
    std::vector<Study*> studies = this->getStudies();
    int i = 0;
    int j = 0;

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

        // Periodically save tick data to the database and free up memory.
        if (cumulativeTicks.size() >= 2000) {
            // Extract the first ~1000 ticks to be inserted.
            std::vector<Tick*> firstCumulativeTicks(cumulativeTicks.begin(), cumulativeTicks.begin() + ((cumulativeTicks.size() - 1000) - 1));

            // Write ticks to database.
            saveTicks(cumulativeTicks);

            for (j=0; j<1000; j++) {
                delete cumulativeTicks[j];
            }

            // Extract the last 1000 elements into a new vector.
            std::vector<Tick*> lastCumulativeTicks(cumulativeTicks.begin() + (cumulativeTicks.size() - 1000), cumulativeTicks.end());

            // Release memory for the old vector.
            std::vector<Tick*>().swap(cumulativeTicks);

            // Set the original to be the new vector.
            cumulativeTicks = lastCumulativeTicks;
        }
    }

    printf("\n");
}

// double *Optimizer::convertTickToArray(Tick *tick) {
//     double *convertedTick = (double*)malloc(this->getStudies().size() * sizeof(double));
//     int index;

//     for (Tick::iterator iterator = tick->begin(); iterator != tick->end(); ++iterator) {
//         // Get the current property index.
//         index = std::distance(tick->begin(), iterator);

//         // Add the current property value to the array.
//         convertedTick[index] = iterator->second;
//     }

//     return convertedTick;
// }

int Optimizer::getDataPropertyCount() {
    std::vector<Study*> studies = this->getStudies();
    int propertyCount = 0;

    for (std::vector<Study*>::iterator iterator = studies.begin(); iterator != studies.end(); ++iterator) {
        propertyCount += (*iterator)->getOutputMap().size();
    }

    return propertyCount;
}

void Optimizer::loadData() {
    int dataPointIndex = 0;
    int propertyIndex = 0;
    mongoc_collection_t *collection;
    mongoc_cursor_t *cursor;
    bson_t *countQuery;
    bson_t *query;
    const bson_t *document;
    bson_iter_t documentIterator;
    bson_iter_t dataIterator;
    bson_error_t error;
    const char *propertyName;
    const bson_value_t *propertyValue;

    // Get a reference to the database collection.
    collection = mongoc_client_get_collection(this->dbClient, "forex-backtesting", "datapoints");

    // Query for the number of data points.
    countQuery = BCON_NEW("$query", "{", "symbol", BCON_UTF8(this->symbol.c_str()), "}");
    this->dataCount = mongoc_collection_count(collection, MONGOC_QUERY_NONE, countQuery, 0, 0, NULL, &error);

    if (this->dataCount < 0) {
        // No data points found.
        throw std::runtime_error(error.message);
    }

    // Allocate memory for the data.
    this->data = (double**)malloc(this->dataCount * sizeof(double*));

    // Query the database.
    query = BCON_NEW(
        "$query", "{", "symbol", BCON_UTF8(this->symbol.c_str()), "}",
        "$orderby", "{", "data.timestamp", BCON_INT32(1), "}",
        "$hint", "{", "data.timestamp", BCON_INT32(1), "}"
    );
    cursor = mongoc_collection_find(collection, MONGOC_QUERY_NONE, 0, 0, 1000, query, NULL, NULL);

    // Go through query results, and convert each document into an array.
    while (mongoc_cursor_next(cursor, &document)) {
        propertyIndex = 0;

        // Allocate memory for the data point.
        this->data[dataPointIndex] = (double*)malloc(this->getDataPropertyCount() * sizeof(double));

        if (bson_iter_init(&documentIterator, document)) {
            // Find the "data" subdocument.
            if (bson_iter_init_find(&documentIterator, document, "data") &&
                BSON_ITER_HOLDS_DOCUMENT(&documentIterator) &&
                bson_iter_recurse(&documentIterator, &dataIterator)) {

                // Iterate through the data properties.
                while (bson_iter_next(&dataIterator)) {
                    propertyValue = bson_iter_value(&dataIterator);

                    // Add the data property value to the data store.
                    this->data[dataPointIndex][propertyIndex] = propertyValue->value.v_double;

                    // For the first data point only (only need to do this once), build an
                    // index of data item positions.
                    if (this->dataCount == 0) {
                        // Get the property name.
                        propertyName = bson_iter_key(&dataIterator);

                        // Add to the data index map.
                        this->dataIndex[propertyName] = propertyIndex;
                    }

                    propertyIndex++;
                }
            }
        }

        // Keep track of the number of ticks.
        dataPointIndex++;
    }

    // Cleanup.
    bson_destroy(countQuery);
    bson_destroy(query);
    mongoc_cursor_destroy(cursor);
    mongoc_collection_destroy(collection);
    delete document;
}

std::vector<Configuration*> Optimizer::buildConfigurations() {
    std::vector<Configuration*> configurations;

    // Built a flat key/value list of configurations.
    // ...

    return configurations;
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
    for (std::vector<Configuration*>::iterator configurationIterator = configurations.begin(); configurationIterator != configurations.end(); ++configurationIterator) {
        i = std::distance(configurations.begin(), configurationIterator);
        strategies[i] = OptimizationStrategyFactory::create(this->strategyName, this->symbol, this->group, *configurationIterator);
    }

    // Iterate over data ticks.
    for (i=0; i<this->dataCount; i++) {
        // Loop through all strategies/configurations.
        for (std::vector<Strategy*>::iterator strategyIterator = strategies.begin(); strategyIterator != strategies.end(); ++strategyIterator) {
            // Use a thread pool so that all CPU cores can be used.
            pool.execute([&]() {
                // Reference: http://stackoverflow.com/a/7854596/83897
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
