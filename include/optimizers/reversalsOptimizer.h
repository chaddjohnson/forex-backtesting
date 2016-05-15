#ifndef REVERSALSOPTIMIZER_H
#define REVERSALSOPTIMIZER_H

#include <string>
#include <vector>
#include <map>
#include <mongoc.h>
#include <boost/variant.hpp>
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
        std::vector<Study*> getStudies();

    public:
        ReversalsOptimizer(mongoc_client_t *dbClient, std::string symbol, int group)
            : Optimizer(dbClient, "ReversalsOptimization", symbol, group) {}
        ~ReversalsOptimizer() {}
        std::map<std::string, std::vector<std::map<std::string, boost::variant<std::string, double, bool>>>> getConfigurationOptions();
};

#endif
