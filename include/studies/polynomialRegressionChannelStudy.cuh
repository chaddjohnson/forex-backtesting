#ifndef POLYNOMIALREGRESSIONCHANNELSTUDY_H
#define POLYNOMIALREGRESSIONCHANNELSTUDY_H

#include <vector>
#include <map>
#include <string>
#include <gsl/gsl_multifit.h>
#include <cmath>
#include "types/tick.cuh"
#include "study.cuh"
#include "types/real.cuh"

class PolynomialRegressionChannelStudy : public Study {
    private:
        std::vector<Real> pastPrices;
        std::vector<Real> pastRegressions;

    public:
        PolynomialRegressionChannelStudy(std::map<std::string, Real> inputs, std::map<std::string, std::string> outputMap)
            : Study(inputs, outputMap) {}
        Real calculateRegression(std::vector<Real> &values, int degree);
        Real calculateStandardDeviation(std::vector<Real> &values);
        void tick();
};

#endif
