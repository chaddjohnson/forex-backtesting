#include "optimizers/optimizer.h"

Optimizer::Optimizer(mongoc_client_t *dbClient, std::string strategyName, std::string symbol, int group) {
    this->dbClient = dbClient;
    this->strategyName = strategyName;
    this->symbol = symbol;
    this->group = group;
    this->dataCount = 0;
}

Optimizer::~Optimizer() {
    free(data);
}

bson_t *Optimizer::convertTickToBson(Tick *tick) {
    bson_t *document;
    bson_t dataDocument;

    document = bson_new();
    BSON_APPEND_UTF8(document, "symbol", this->symbol.c_str());
    BSON_APPEND_INT32(document, "testingGroups", tick->at("testingGroups"));
    BSON_APPEND_INT32(document, "validationGroups", tick->at("validationGroups"));
    BSON_APPEND_DOCUMENT_BEGIN(document, "data", &dataDocument);

    // Remove group keys as they are no longer needed.
    tick->erase("testingGroups");
    tick->erase("validationGroups");

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
    collection = mongoc_client_get_collection(this->dbClient, "forex-backtesting-test", "datapoints");

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
}

void Optimizer::prepareData(std::vector<Tick*> ticks) {
    double percentage;
    int tickCount = ticks.size();
    std::vector<Tick*> cumulativeTicks;
    Tick *tick = nullptr;
    Tick *previousTick = nullptr;
    int threadCount = std::thread::hardware_concurrency();
    maginatics::ThreadPool pool(1, threadCount, 5000);
    std::vector<Study*> studies = this->getStudies();
    int i = 0;
    int j = 0;

    // Reserve space in advance for better performance
    cumulativeTicks.reserve(tickCount);

    printf("Preparing data...");

    // Go through the data and run studies for each data item.
    for (std::vector<Tick*>::iterator tickIterator = ticks.begin(); tickIterator != ticks.end(); ++tickIterator) {
        // Show progress.
        percentage = (++i / (double)tickCount) * 100.0;
        printf("\rPreparing data...%0.4f%%", percentage);

        tick = *tickIterator;

        // If the previous tick's minute was not the previous minute, then save the current
        // ticks, and start over with recording.
        if (previousTick && ((*tick)["timestamp"] - (*previousTick)["timestamp"]) > 60) {
            // Save and then remove the current cumulative ticks.
            saveTicks(cumulativeTicks);
            std::vector<Tick*>().swap(cumulativeTicks);
        }

        // Append to the cumulative data.
        cumulativeTicks.push_back(tick);

        for (std::vector<Study*>::iterator studyIterator = studies.begin(); studyIterator != studies.end(); ++studyIterator) {
            // Update the data for the study.
            (*studyIterator)->setData(&cumulativeTicks);

            // Use a thread pool so that all CPU cores can be used.
            pool.execute([studyIterator]() {
                // Process the latest data for the study.
                (*studyIterator)->tick();
            });
        }

        // Block until all tasks for the current data point to complete.
        pool.drain();

        // Merge tick output values from the studies into the current tick.
        for (std::vector<Study*>::iterator studyIterator = studies.begin(); studyIterator != studies.end(); ++studyIterator) {
            std::map<std::string, double> studyOutputs = (*studyIterator)->getTickOutputs();

            for (std::map<std::string, double>::iterator outputIterator = studyOutputs.begin(); outputIterator != studyOutputs.end(); ++outputIterator) {
                (*tick)[outputIterator->first] = outputIterator->second;
            }

            (*studyIterator)->resetTickOutputs();
        }

        previousTick = tick;

        // Periodically save tick data to the database and free up memory.
        if (cumulativeTicks.size() >= 2000) {
            // Extract the first ~1000 ticks to be inserted.
            std::vector<Tick*> firstCumulativeTicks(cumulativeTicks.begin(), cumulativeTicks.begin() + (cumulativeTicks.size() - 1000));

            // Write ticks to database.
            saveTicks(firstCumulativeTicks);

            // Release memory.
            std::vector<Tick*>().swap(firstCumulativeTicks);
            for (j=0; j<1000; j++) {
                delete cumulativeTicks[j];
            }

            // Keep only the last 1000 elements.
            std::vector<Tick*>(cumulativeTicks.begin() + (cumulativeTicks.size() - 1000), cumulativeTicks.end()).swap(cumulativeTicks);
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
    collection = mongoc_client_get_collection(this->dbClient, "forex-backtesting-test", "datapoints");

    // Query for the number of data points.
    countQuery = BCON_NEW("symbol", BCON_UTF8(this->symbol.c_str()));
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
}

std::vector<Configuration*> Optimizer::buildConfigurations() {
    std::vector<Configuration*> configurations;

    // Built a flat key/value list of configurations.
    // ...

    return configurations;
}

void Optimizer::optimize(std::vector<Configuration*> configurations, double investment, double profitability) {
    double percentage;
    int threadCount = std::thread::hardware_concurrency();
    std::vector<Strategy*> strategies;
    int i = 0;

    // Load tick data from the database.
    loadData();

    // Set up a threadpool so all CPU cores and their threads can be used.
    maginatics::ThreadPool pool(1, threadCount, 5000);

    printf("Preparing configurations...");

    // Set up one strategy instance per configuration.
    for (std::vector<Configuration*>::iterator configurationIterator = configurations.begin(); configurationIterator != configurations.end(); ++configurationIterator) {
        i = std::distance(configurations.begin(), configurationIterator);
        strategies[i] = OptimizationStrategyFactory::create(this->strategyName, this->symbol, this->dataIndex, this->group, *configurationIterator);
    }

    printf("%i configurations prepared\n", strategies.size());
    printf("Optimizing...");

    // Iterate over data ticks.
    for (i=0; i<this->dataCount; i++) {
        // Loop through all strategies/configurations.
        for (std::vector<Strategy*>::iterator strategyIterator = strategies.begin(); strategyIterator != strategies.end(); ++strategyIterator) {
            // Use a thread pool so that all CPU cores can be used.
            pool.execute([this, strategyIterator, i, investment, profitability]() {
                // Process the latest data for the study.
                (*strategyIterator)->backtest(this->data[i], investment, profitability);
            });
        }

        // Block until all tasks for the current data point to complete.
        pool.drain();

        // Show progress.
        percentage = (++i / (double)this->dataCount) * 100.0;
        printf("\rOptimizing...%0.4f%%", percentage);
    }

    // Unload data.
    // TODO
}
