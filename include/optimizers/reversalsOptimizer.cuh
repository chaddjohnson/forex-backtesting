#ifndef REVERSALSOPTIMIZER_H
#define REVERSALSOPTIMIZER_H

#include <string>
#include <vector>
#include <map>
#include <mongoc.h>
#include <boost/variant.hpp>
#include "optimizers/optimizer.cuh"
#include "studies/smaStudy.cuh"
#include "studies/emaStudy.cuh"
#include "studies/rsiStudy.cuh"
#include "studies/stochasticOscillatorStudy.cuh"
#include "studies/polynomialRegressionChannelStudy.cuh"

class ReversalsOptimizer : public Optimizer {
    private:
        std::vector<Study*> studies;

    protected:
        std::vector<Study*> getStudies();

    public:
        ReversalsOptimizer(mongoc_client_t *dbClient, char *symbol, int group)
            : Optimizer(dbClient, "ReversalsOptimization", symbol, group) {}
        ~ReversalsOptimizer() {}
        std::map<std::string, ConfigurationOption> getConfigurationOptions();
};

#endif
