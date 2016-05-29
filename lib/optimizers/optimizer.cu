#include "optimizers/optimizer.cuh"

__global__ void optimizer_backtest(double *data, Strategy *strategies, int strategyCount, double investment, double profitability) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < strategyCount) {
        // strategies[i]->backtest(data[dataPointIndex], investment, profitability);

        // TODO: Remove this.
        int j = 0;
        int k = 0;
        for (j=0; j<50; j++) {
            k++;
        }
    }
}

Optimizer::Optimizer(mongoc_client_t *dbClient, const char *strategyName, const char *symbol, int group) {
    this->dbClient = dbClient;
    this->strategyName = strategyName;
    this->symbol = symbol;
    this->group = group;
    this->dataPropertyCount = 0;
    this->dataIndexMap = new std::map<std::string, int>();
}

bson_t *Optimizer::convertTickToBson(Tick *tick) {
    bson_t *document;
    bson_t dataDocument;

    document = bson_new();
    BSON_APPEND_UTF8(document, "symbol", this->symbol);
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
    if (ticks.size() == 0) {
        return;
    }

    mongoc_collection_t *collection;
    mongoc_bulk_operation_t *bulkOperation;
    bson_t bulkOperationReply;
    bson_error_t bulkOperationError;

    // Get a reference to the database collection.
    collection = mongoc_client_get_collection(this->dbClient, "forex-backtesting-test", "datapoints");

    // Begin a bulk operation.
    bulkOperation = mongoc_collection_create_bulk_operation(collection, true, NULL);

    // Reference: http://api.mongodb.org/c/current/bulk.html
    for (std::vector<Tick*>::iterator insertionIterator = ticks.begin(); insertionIterator != ticks.end(); ++insertionIterator) {
        bson_t *document = convertTickToBson(*insertionIterator);
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
    int cumulativeTickCount;
    int threadCount = std::thread::hardware_concurrency();
    maginatics::ThreadPool pool(1, threadCount, 5000);
    std::vector<Study*> studies = this->getStudies();
    int i = 0;
    int j = 0;

    // Reserve space in advance for better performance.
    cumulativeTicks.reserve(tickCount);

    printf("Preparing data...");

    // Go through the data and run studies for each data item.
    for (std::vector<Tick*>::iterator tickIterator = ticks.begin(); tickIterator != ticks.end(); ++tickIterator) {
        // Show progress.
        percentage = (++i / (double)tickCount) * 100.0;
        printf("\rPreparing data...%0.4f%%", percentage);

        Tick *tick = *tickIterator;
        Tick *previousTick = nullptr;

        if (cumulativeTicks.size() > 0) {
            previousTick = cumulativeTicks.back();
        }

        // If the previous tick's minute was not the previous minute, then save the current
        // ticks, and start over with recording.
        if (previousTick && ((*tick)["timestamp"] - (*previousTick)["timestamp"]) > 60) {
            previousTick = nullptr;

            // Save and then remove the current cumulative ticks.
            saveTicks(cumulativeTicks);

            // Release memory.
            cumulativeTickCount = cumulativeTicks.size();
            for (j=0; j<cumulativeTickCount; j++) {
                delete cumulativeTicks[j];
                cumulativeTicks[j] = nullptr;
            }
            std::vector<Tick*>().swap(cumulativeTicks);
        }

        previousTick = tick;

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

        // Block until all tasks for the current data point complete.
        pool.drain();

        // Merge tick output values from the studies into the current tick.
        for (std::vector<Study*>::iterator studyIterator = studies.begin(); studyIterator != studies.end(); ++studyIterator) {
            std::map<std::string, double> studyOutputs = (*studyIterator)->getTickOutputs();

            for (std::map<std::string, double>::iterator outputIterator = studyOutputs.begin(); outputIterator != studyOutputs.end(); ++outputIterator) {
                (*tick)[outputIterator->first] = outputIterator->second;
            }
        }

        // Periodically save tick data to the database and free up memory.
        if (cumulativeTicks.size() >= 2000) {
            // Extract the first ~1000 ticks to be inserted.
            std::vector<Tick*> firstCumulativeTicks(cumulativeTicks.begin(), cumulativeTicks.begin() + (cumulativeTicks.size() - 1000));

            // Write ticks to database.
            saveTicks(firstCumulativeTicks);

            // Release memory.
            for (j=0; j<1000; j++) {
                delete cumulativeTicks[j];
                cumulativeTicks[j] = nullptr;
            }
            std::vector<Tick*>().swap(firstCumulativeTicks);

            // Keep only the last 1000 elements.
            std::vector<Tick*>(cumulativeTicks.begin() + (cumulativeTicks.size() - 1000), cumulativeTicks.end()).swap(cumulativeTicks);
        }

        tick = nullptr;
        previousTick = nullptr;
    }

    printf("\n");
}

int Optimizer::getDataPropertyCount() {
    if (this->dataPropertyCount) {
        return this->dataPropertyCount;
    }

    std::vector<Study*> studies = this->getStudies();
    this->dataPropertyCount = 5;

    for (std::vector<Study*>::iterator iterator = studies.begin(); iterator != studies.end(); ++iterator) {
        this->dataPropertyCount += (*iterator)->getOutputMap().size();
    }

    return this->dataPropertyCount;
}

std::map<std::string, int> *Optimizer::getDataIndexMap() {
    if (this->dataIndexMap->size() > 0) {
        return this->dataIndexMap;
    }

    std::vector<std::string> properties;

    // Add basic properties.
    properties.push_back("timestamp");
    properties.push_back("timestampHour");
    properties.push_back("timestampMinute");
    properties.push_back("open");
    properties.push_back("high");
    properties.push_back("low");
    properties.push_back("close");

    std::vector<Study*> studies = this->getStudies();

    for (std::vector<Study*>::iterator iterator = studies.begin(); iterator != studies.end(); ++iterator) {
        std::map<std::string, std::string> outputMap = (*iterator)->getOutputMap();

        for (std::map<std::string, std::string>::iterator outputMapIterator = outputMap.begin(); outputMapIterator != outputMap.end(); ++outputMapIterator) {
            properties.push_back(outputMapIterator->second);
        }
    }

    std::sort(properties.begin(), properties.end());

    for (std::vector<std::string>::iterator propertyIterator = properties.begin(); propertyIterator != properties.end(); ++propertyIterator) {
        (*this->dataIndexMap)[*propertyIterator] = std::distance(properties.begin(), propertyIterator);
    }

    return this->dataIndexMap;
}

BasicDataIndexMap Optimizer::getBasicDataIndexMap() {
    std::map<std::string, int> *dataIndexMap = this->getDataIndexMap();
    BasicDataIndexMap basicDataIndexMap;

    basicDataIndexMap.timestamp = (*dataIndexMap)["timestamp"];
    basicDataIndexMap.open = (*dataIndexMap)["open"];
    basicDataIndexMap.high = (*dataIndexMap)["high"];
    basicDataIndexMap.low = (*dataIndexMap)["low"];
    basicDataIndexMap.close = (*dataIndexMap)["close"];

    return basicDataIndexMap;
}

double *Optimizer::loadData(int offset, int chunkSize) {
    mongoc_collection_t *collection;
    mongoc_cursor_t *cursor;
    bson_t *countQuery;
    bson_t *query;
    const bson_t *document;
    bson_iter_t documentIterator;
    bson_iter_t dataIterator;
    bson_error_t error;
    const bson_value_t *propertyValue;
    int dataPointCount;
    int dataPointIndex = 0;
    int propertyIndex = 0;
    std::map<std::string, int> *dataIndexMap = this->getDataIndexMap();

    // Get a reference to the database collection.
    collection = mongoc_client_get_collection(this->dbClient, "forex-backtesting-test", "datapoints");

    // Query for the number of data points.
    countQuery = BCON_NEW("symbol", BCON_UTF8(this->symbol));
    dataPointCount = mongoc_collection_count(collection, MONGOC_QUERY_NONE, countQuery, offset, chunkSize, NULL, &error);

    // Allocate memory for the flattened data store.
    uint64_t dataChunkBytes = dataPointCount * this->getDataPropertyCount() * sizeof(double);
    double *data = (double*)malloc(dataChunkBytes);

    if (dataPointCount < 0) {
        // No data points found.
        throw std::runtime_error(error.message);
    }

    // Query the database.
    query = BCON_NEW(
        "$query", "{", "symbol", BCON_UTF8(this->symbol), "}",
        "$orderby", "{", "data.timestamp", BCON_INT32(1), "}",
        "$hint", "{", "data.timestamp", BCON_INT32(1), "}"
    );
    cursor = mongoc_collection_find(collection, MONGOC_QUERY_NONE, offset, chunkSize, 1000, query, NULL, NULL);

    // Go through query results, and convert each document into an array.
    while (mongoc_cursor_next(cursor, &document)) {
        propertyIndex = 0;

        if (bson_iter_init(&documentIterator, document)) {
            // Find the "data" subdocument.
            if (bson_iter_init_find(&documentIterator, document, "data") &&
                BSON_ITER_HOLDS_DOCUMENT(&documentIterator) &&
                bson_iter_recurse(&documentIterator, &dataIterator)) {

                // Iterate through the data properties.
                while (bson_iter_next(&dataIterator)) {
                    propertyValue = bson_iter_value(&dataIterator);

                    // Add the data property value to the flattened data store.
                    data[dataPointIndex * this->getDataPropertyCount() + propertyIndex] = propertyValue->value.v_double;

                    propertyIndex++;
                }

                // Add additional timestamp-related data.
                time_t utcTime = data[dataPointIndex * this->getDataPropertyCount() + (*dataIndexMap)["timestamp"]];
                struct tm *localTime = localtime(&utcTime);
                data[dataPointIndex * this->getDataPropertyCount() + (*dataIndexMap)["timestampHour"]] = localTime->tm_hour;
                data[dataPointIndex * this->getDataPropertyCount() + (*dataIndexMap)["timestampMinute"]] = localTime->tm_min;
            }
        }

        dataPointIndex++;
    }

    // Cleanup.
    bson_destroy(countQuery);
    bson_destroy(query);
    mongoc_cursor_destroy(cursor);
    mongoc_collection_destroy(collection);

    // Return the pointer to the data.
    return data;
}

std::vector<MapConfiguration*> *Optimizer::buildMapConfigurations(
    std::map<std::string, ConfigurationOption> options,
    int optionIndex,
    std::vector<MapConfiguration*> *results,
    MapConfiguration *current
) {
    std::vector<std::string> allKeys;
    std::string optionKey;
    ConfigurationOption configurationOptions;
    std::map<std::string, int> *dataIndexMap = this->getDataIndexMap();

    // Get all options keys.
    for (std::map<std::string, ConfigurationOption>::iterator optionsIterator = options.begin(); optionsIterator != options.end(); ++optionsIterator) {
        allKeys.push_back(optionsIterator->first);
    }

    optionKey = allKeys[optionIndex];
    configurationOptions = options[optionKey];

    for (ConfigurationOption::iterator configurationOptionsIterator = configurationOptions.begin(); configurationOptionsIterator != configurationOptions.end(); ++configurationOptionsIterator) {
        // Iterate through configuration option values.
        for (std::map<std::string, boost::variant<std::string, double>>::iterator valuesIterator = configurationOptionsIterator->begin(); valuesIterator != configurationOptionsIterator->end(); ++valuesIterator) {
            if (valuesIterator->second.type() == typeid(std::string)) {
                if (boost::get<std::string>(valuesIterator->second).length() > 0) {
                    // Value points to a key.
                    (*current)[valuesIterator->first] = (*dataIndexMap)[boost::get<std::string>(valuesIterator->second)];
                }
            }
            else {
                // Value is an actual value.
                (*current)[valuesIterator->first] = boost::get<double>(valuesIterator->second);
            }
        }

        if (optionIndex + 1 < allKeys.size()) {
            buildMapConfigurations(options, optionIndex + 1, results, current);
        }
        else {
            results->push_back(current);
        }
    }

    return results;
}

std::vector<Configuration*> Optimizer::buildConfigurations(std::map<std::string, ConfigurationOption> options) {
    printf("Building configurations...");

    std::map<std::string, int> *dataIndexMap = this->getDataIndexMap();
    std::vector<MapConfiguration*> *mapConfigurations = buildMapConfigurations(options);
    std::vector<Configuration*> configurations;
    Configuration *configuration = nullptr;

    // Reserve space in advance for better performance.
    configurations.reserve(mapConfigurations->size());

    // Convert map representations of maps into structs of type Configuration.
    for (std::vector<MapConfiguration*>::iterator mapConfigurationIterator = mapConfigurations->begin(); mapConfigurationIterator != mapConfigurations->end(); ++mapConfigurationIterator) {
        // Set up a new, empty configuration.
        configuration = new Configuration();

        // Set basic properties.
        configuration->timestamp = (*dataIndexMap)["timestamp"];
        configuration->timestampHour = (*dataIndexMap)["timestampHour"];
        configuration->timestampMinute = (*dataIndexMap)["timestampMinute"];
        configuration->open = (*dataIndexMap)["open"];
        configuration->high = (*dataIndexMap)["high"];
        configuration->low = (*dataIndexMap)["low"];
        configuration->close = (*dataIndexMap)["close"];

        // Set index mappings.
        if ((*mapConfigurationIterator)->find("sma13") != (*mapConfigurationIterator)->end()) {
            configuration->sma13 = boost::get<int>((**mapConfigurationIterator)["sma13"]);
        }
        if ((*mapConfigurationIterator)->find("ema50") != (*mapConfigurationIterator)->end()) {
            configuration->ema50 = boost::get<int>((**mapConfigurationIterator)["ema50"]);
        }
        if ((*mapConfigurationIterator)->find("ema100") != (*mapConfigurationIterator)->end()) {
            configuration->ema100 = boost::get<int>((**mapConfigurationIterator)["ema100"]);
        }
        if ((*mapConfigurationIterator)->find("ema200") != (*mapConfigurationIterator)->end()) {
            configuration->ema200 = boost::get<int>((**mapConfigurationIterator)["ema200"]);
        }
        if ((*mapConfigurationIterator)->find("ema250") != (*mapConfigurationIterator)->end()) {
            configuration->ema250 = boost::get<int>((**mapConfigurationIterator)["ema250"]);
        }
        if ((*mapConfigurationIterator)->find("ema300") != (*mapConfigurationIterator)->end()) {
            configuration->ema300 = boost::get<int>((**mapConfigurationIterator)["ema300"]);
        }
        if ((*mapConfigurationIterator)->find("ema350") != (*mapConfigurationIterator)->end()) {
            configuration->ema350 = boost::get<int>((**mapConfigurationIterator)["ema350"]);
        }
        if ((*mapConfigurationIterator)->find("ema400") != (*mapConfigurationIterator)->end()) {
            configuration->ema400 = boost::get<int>((**mapConfigurationIterator)["ema400"]);
        }
        if ((*mapConfigurationIterator)->find("ema450") != (*mapConfigurationIterator)->end()) {
            configuration->ema450 = boost::get<int>((**mapConfigurationIterator)["ema450"]);
        }
        if ((*mapConfigurationIterator)->find("ema500") != (*mapConfigurationIterator)->end()) {
            configuration->ema500 = boost::get<int>((**mapConfigurationIterator)["ema500"]);
        }
        if ((*mapConfigurationIterator)->find("rsi") != (*mapConfigurationIterator)->end()) {
            configuration->rsi = boost::get<int>((**mapConfigurationIterator)["rsi"]);
        }
        if ((*mapConfigurationIterator)->find("stochasticD") != (*mapConfigurationIterator)->end()) {
            configuration->stochasticD = boost::get<int>((**mapConfigurationIterator)["stochasticD"]);
        }
        if ((*mapConfigurationIterator)->find("stochasticK") != (*mapConfigurationIterator)->end()) {
            configuration->stochasticK = boost::get<int>((**mapConfigurationIterator)["stochasticK"]);
        }
        if ((*mapConfigurationIterator)->find("prChannelUpper") != (*mapConfigurationIterator)->end()) {
            configuration->prChannelUpper = boost::get<int>((**mapConfigurationIterator)["prChannelUpper"]);
        }
        if ((*mapConfigurationIterator)->find("prChannelLower") != (*mapConfigurationIterator)->end()) {
            configuration->prChannelLower = boost::get<int>((**mapConfigurationIterator)["prChannelLower"]);
        }

        // Set values.
        if ((*mapConfigurationIterator)->find("rsiOverbought") != (*mapConfigurationIterator)->end()) {
            configuration->rsiOverbought = boost::get<double>((**mapConfigurationIterator)["rsiOverbought"]);
        }
        if ((*mapConfigurationIterator)->find("rsiOversold") != (*mapConfigurationIterator)->end()) {
            configuration->rsiOversold = boost::get<double>((**mapConfigurationIterator)["rsiOversold"]);
        }
        if ((*mapConfigurationIterator)->find("stochasticOverbought") != (*mapConfigurationIterator)->end()) {
            configuration->stochasticOverbought = boost::get<double>((**mapConfigurationIterator)["stochasticOverbought"]);
        }
        if ((*mapConfigurationIterator)->find("stochasticOversold") != (*mapConfigurationIterator)->end()) {
            configuration->stochasticOversold = boost::get<double>((**mapConfigurationIterator)["stochasticOversold"]);
        }

        configurations.push_back(configuration);
    }

    printf("%i configurations built\n", (int)configurations.size());

    return configurations;
}

void Optimizer::optimize(std::vector<Configuration*> &configurations, double investment, double profitability) {
    printf("Optimizing...");

    double percentage;
    mongoc_collection_t *collection;
    bson_t *countQuery;
    bson_error_t error;
    int dataPointCount;
    int configurationCount = configurations.size();
    int dataChunkSize = 1000000;
    int dataOffset = 0;
    std::map<std::string, int> *dataIndexMap = this->getDataIndexMap();
    int chunkNumber = 0;
    int dataPointIndex = 0;
    int i = 0;

    // GPU settings.
    int blockCount = 32;
    int threadsPerBlock = 1024;

    // Host data.
    Strategy *strategies = (Strategy*)malloc(configurationCount * sizeof(Strategy));

    // GPU data.
    Strategy *devStrategies;

    // Get a count of all data points for the symbol.
    collection = mongoc_client_get_collection(this->dbClient, "forex-backtesting-test", "datapoints");
    countQuery = BCON_NEW("symbol", BCON_UTF8(this->symbol));
    dataPointCount = mongoc_collection_count(collection, MONGOC_QUERY_NONE, countQuery, 0, 0, NULL, &error);

    // Set up one strategy instance per configuration.
    for (i=0; i<configurationCount; i++) {
        strategies[i] = OptimizationStrategyFactory::create(this->strategyName, this->symbol, getBasicDataIndexMap(), this->group, configurations[i]);
    }

    cudaSetDevice(0);

    // Allocate memory on the GPU.
    cudaMalloc((void**)&devStrategies, configurationCount * sizeof(Strategy));

    // Copy strategies and data to the GPU.
    cudaMemcpy(devStrategies, strategies, configurationCount * sizeof(Strategy), cudaMemcpyHostToDevice);

    while (dataOffset < dataPointCount) {
        // Calculate the next chunk's size.
        int nextChunkSize = (chunkNumber * dataChunkSize < dataPointCount) ? dataChunkSize : ((chunkNumber * dataChunkSize) - dataPointCount);

        // Calculate the number of bytes needed for the next chunk.
        uint64_t dataChunkBytes = nextChunkSize * this->getDataPropertyCount() * sizeof(double);

        // Load another chunk of data.
        double *data = loadData(dataOffset, nextChunkSize);
        double *devData;

        // Allocate memory for the data on the GPU.
        cudaMalloc((void**)&devData, dataChunkBytes);

        // Copy a chunk of data points to the GPU.
        cudaMemcpy(devData, data, dataChunkBytes, cudaMemcpyHostToDevice);

        // Backtest all strategies against the current data point.
        // TODO? Loop through all data points in the chunk?
        for (i=0; i<nextChunkSize; i++) {
            // Show progress.
            percentage = (++dataPointIndex / (double)dataPointCount) * 100.0;
            printf("\rOptimizing...%0.4f%%", percentage);

            optimizer_backtest<<<blockCount, threadsPerBlock>>>(devData, devStrategies, configurationCount, investment, profitability);
        }

        // TODO: Determine if this is actually necessary.
        // cudaDeviceSynchronize();

        // Free GPU and host memory;
        cudaFree(devData);
        free(data);

        dataOffset += nextChunkSize;
        chunkNumber++;
    }

    // Copy strategies from the GPU to the host.
    cudaMemcpy(strategies, devStrategies, configurationCount * sizeof(Strategy), cudaMemcpyDeviceToHost);

    // Save results.
    // TODO

    // Free memory on the GPU memory.
    cudaFree(devStrategies);

    // Free host memory.
    free(strategies);

    printf("\n");
}
