#ifndef POLYNOMIALREGRESSIONCHANNELSTUDY_H
#define POLYNOMIALREGRESSIONCHANNELSTUDY_H

#include <vector>
#include <map>
#include <string>
#include <gsl/gsl_multifit.h>
#include <math.h>
#include "types/tick.h"
#include "study.h"

class PolynomialRegressionChannelStudy : public Study {
    private:
        std::vector<double> pastPrices;
        std::vector<double> pastRegressions;

    public:
        PolynomialRegressionChannelStudy(std::map<std::string, double> inputs, std::map<std::string, std::string> outputMap)
            : Study(inputs, outputMap) {}
        double calculateRegression(std::vector<double> &values, int degree);
        double calculateStandardDeviation(std::vector<double> &values);
        void tick();
};

#endif
