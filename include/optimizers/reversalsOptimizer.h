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
        ReversalsOptimizer(std::string symbol, int group)
            : Optimizer("Reversals", symbol, group) {}
        void prepareStudies();
};

#endif
