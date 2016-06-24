#ifndef POLYNOMIALREGRESSIONCHANNELSTUDY_H
#define POLYNOMIALREGRESSIONCHANNELSTUDY_H

#include <vector>
#include <map>
#include <string>
#include <gsl/gsl_multifit.h>
#include <cmath>
#include "types/tick.cuh"
#include "study.cuh"

class PolynomialRegressionChannelStudy : public Study {
    private:
        std::vector<float> pastPrices;
        std::vector<float> pastRegressions;

    public:
        PolynomialRegressionChannelStudy(std::map<std::string, float> inputs, std::map<std::string, std::string> outputMap)
            : Study(inputs, outputMap) {}
        float calculateRegression(std::vector<float> &values, int degree);
        float calculateStandardDeviation(std::vector<float> &values);
        void tick();
};

#endif
