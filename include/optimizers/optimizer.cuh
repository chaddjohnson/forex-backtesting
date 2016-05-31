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
#include <mongoc.h>
#include <bson.h>
#include <bcon.h>
#include <boost/variant.hpp>
#include "maginatics/threadpool/threadpool.h"
#include "studies/study.cuh"
#include "strategies/reversalsOptimizationStrategy.cuh"
#include "types/tick.cuh"
#include "types/configuration.cuh"
#include "types/mapConfiguration.cuh"
#include "types/configurationOption.cuh"

// CUDA kernel headers.
__global__ void optimizer_backtest(double *data, ReversalsOptimizationStrategy *strategies, int strategyCount, double investment, double profitability);

class Optimizer {
    private:
        mongoc_client_t *dbClient;
        const char *strategyName;
        const char *symbol;
        int group;
        std::map<std::string, int> *dataIndexMap;
        int dataPropertyCount;
        int getDataPropertyCount();
        bson_t *convertTickToBson(Tick *tick);
        std::map<std::string, int> *getDataIndexMap();
        BasicDataIndexMap getBasicDataIndexMap();
        double *loadData(int offset, int chunkSize);
        void saveTicks(std::vector<Tick*> ticks);
        std::vector<MapConfiguration*> *buildMapConfigurations(
            std::map<std::string, ConfigurationOption> options,
            int optionIndex = 0,
            std::vector<MapConfiguration*> *results = new std::vector<MapConfiguration*>(),
            MapConfiguration *current = new MapConfiguration()
        );

    protected:
        virtual std::vector<Study*> getStudies() {
            return std::vector<Study*>();
        }

    public:
        Optimizer(mongoc_client_t *dbClient, const char *strategyName, const char *symbol, int group);
        virtual ~Optimizer() {}
        void prepareData(std::vector<Tick*> ticks);
        virtual std::map<std::string, ConfigurationOption> getConfigurationOptions() {
            return std::map<std::string, ConfigurationOption>();
        }
        std::vector<Configuration*> buildConfigurations(std::map<std::string, ConfigurationOption> options);
        void optimize(std::vector<Configuration*> &configurations, double investment, double profitability);
};

#endif
