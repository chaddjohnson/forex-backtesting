#ifndef REVERSALSOPTIMIZER_H
#define REVERSALSOPTIMIZER_H

#include <string>
#include "optimizers/optimizer.h"
#include "studies/smaStudy.h"
#include "studies/emaStudy.h"
#include "studies/rsiStudy.h"
#include "studies/stochasticOscillatorStudy.h"
#include "studies/polynomialRegressionChannelStudy.h"

class ReversalsOptimizer : public Optimizer {
    public:
        ReversalsOptimizer(mongoc_client_t *dbClient, std::string symbol, int group)
            : Optimizer(dbClient, "Reversals", symbol, group) {}
        void prepareStudies();
};

#endif
