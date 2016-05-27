#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>
#include <map>
#include <iterator>
#include <exception>
#include <mongoc.h>
#include <bson.h>
#include <bcon.h>
#include <boost/variant.hpp>
#include "maginatics/threadpool/threadpool.h"
#include "studies/study.cuh"
#include "strategies/strategy.cuh"
#include "factories/optimizationStrategyFactory.cuh"
#include "types/tick.cuh"
#include "types/configuration.cuh"
#include "types/mapConfiguration.cuh"
#include "types/configurationOption.cuh"

// CUDA kernel headers.
__global__ void optimizer_initialize(Strategy *strategies, Configuration *configurations, int configurationCount);
__global__ void optimizer_backtest(double *data, Strategy *strategies, int dataPointIndex, int configurationCount, double investment, double profitability);

class Optimizer {
    private:
        mongoc_client_t *dbClient;
        const char *strategyName;
        const char *symbol;
        int group;
        std::map<std::string, int> *dataIndexMap;
        int propertyCount;
        int getDataPropertyCount();
        bson_t *convertTickToBson(Tick *tick);
        std::map<std::string, int> *getDataIndexMap();
        double *loadData(int offset, int chunkSize);
        void saveTicks(std::vector<Tick*> ticks);
        std::vector<MapConfiguration*> *buildMapConfigurations(
            std::map<std::string, ConfigurationOption> options,
            int optionIndex = 0,
            std::vector<MapConfiguration*> *results = new std::vector<MapConfiguration*>(),
            MapConfiguration *current = new MapConfiguration()
        );

    protected:
        virtual std::vector<Study*> getStudies() = 0;

    public:
        Optimizer(mongoc_client_t *dbClient, const char *strategyName, const char *symbol, int group);
        virtual ~Optimizer() {}
        void prepareData(std::vector<Tick*> ticks);
        virtual std::map<std::string, ConfigurationOption> getConfigurationOptions() = 0;
        std::vector<Configuration*> buildConfigurations(std::map<std::string, ConfigurationOption> options);
        void optimize(std::vector<Configuration*> &configurations, double investment, double profitability);
};

#endif
