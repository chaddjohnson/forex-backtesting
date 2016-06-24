#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <iterator>
#include <exception>
#include <ctime>
#include <cmath>
#include <mongoc.h>
#include <bson.h>
#include <bcon.h>
#include <boost/variant.hpp>
#include "maginatics/threadpool/threadpool.h"
#include "dataParsers/dataParser.cuh"
#include "studies/study.cuh"
#include "strategies/reversalsOptimizationStrategy.cuh"
#include "types/tick.cuh"
#include "types/configuration.cuh"
#include "types/mapConfiguration.cuh"
#include "types/configurationOption.cuh"
#include "types/strategyResult.cuh"

class Optimizer {
    private:
        mongoc_client_t *dbClient;
        std::string strategyName;
        std::string symbol;
        int type;
        std::string groupFilter;
        int group;
        std::map<std::string, int> *dataIndexMap;
        int dataPropertyCount;
        int getDataPropertyCount();
        bson_t *convertTickToBson(Tick *tick);
        float *loadData(int lastTimestamp, int chunkSize);
        void saveTicks(std::vector<Tick*> ticks);
        void saveResults(std::vector<StrategyResult> &results);

    protected:
        mongoc_client_t *getDbClient();
        std::string getStrategyName();
        std::string getSymbol();
        int getType();
        int getGroup();
        std::map<std::string, int> *getDataIndexMap();
        std::string findDataIndexMapKeyByValue(int value);
        virtual std::vector<Study*> getStudies() = 0;
        std::vector<MapConfiguration> *buildMapConfigurations(
            std::map<std::string, ConfigurationOption> options,
            int optionIndex = 0,
            std::vector<MapConfiguration> *results = new std::vector<MapConfiguration>(),
            MapConfiguration *current = new MapConfiguration()
        );
        virtual std::map<std::string, ConfigurationOption> getConfigurationOptions() = 0;
        virtual std::vector<Configuration*> buildBaseConfigurations() = 0;
        virtual std::vector<Configuration*> buildGroupConfigurations() = 0;
        virtual std::vector<Configuration*> buildSavedConfigurations() = 0;
        virtual std::vector<Configuration*> loadConfigurations(const char *collectionName, bson_t *query) = 0;
        virtual bson_t *convertResultToBson(StrategyResult &result) = 0;

    public:
        Optimizer(mongoc_client_t *dbClient, std::string strategyName, std::string symbol, int type = 0, int group = 0);
        virtual ~Optimizer() {}
        void prepareData(std::vector<Tick*> ticks);
        void optimize(float investment, float profitability);
        static int getTypeId(std::string name);
        enum types { TEST, VALIDATION, FORWARDTEST };
};

#endif
