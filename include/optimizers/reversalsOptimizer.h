#ifndef REVERSALSOPTIMIZER_H
#define REVERSALSOPTIMIZER_H

#include <string>
#include <vector>
#include "optimizers/optimizer.h"
#include "studies/smaStudy.h"
#include "studies/emaStudy.h"
#include "studies/rsiStudy.h"
#include "studies/stochasticOscillatorStudy.h"
#include "studies/polynomialRegressionChannelStudy.h"

class ReversalsOptimizer : public Optimizer {
    private:
        std::vector<Study*> studies;

    protected:
        std::map<std::string, std::vector<std::map<std::string, boost::variant<std::string, double>>>> getConfigurationOptions();
        std::vector<Study*> getStudies();

    public:
        ReversalsOptimizer(mongoc_client_t *dbClient, std::string symbol, int group)
            : Optimizer(dbClient, "ReversalsOptimization", symbol, group) {}
        void prepareStudies();
};

#endif
