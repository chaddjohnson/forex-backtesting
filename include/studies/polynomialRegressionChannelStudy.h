#ifndef SMASTUDY_H
#define SMASTUDY_H

#include <vector>
#include <map>
#include <string>
#include <gsl/gsl_multifit.h>
#include <math.h>
#include "types/tick.h"
#include "study.h"

class PolynomialRegressionChannelStudy : public Study {
    public:
        PolynomialRegressionChannelStudy(std::map<std::string, double> &inputs, std::map<std::string, std::string> &outputMap)
            : Study(inputs, outputMap) {}
        double calculateRegression(std::vector<double> &values, int degree);
        double calculateStandardDeviation(std::vector<double> &values);
        std::map<std::string, double> tick();
};

#endif
