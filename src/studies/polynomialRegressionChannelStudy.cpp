#include "polynomialRegressionChannelStudy.h"

// Sources: https://rosettacode.org/wiki/Polynomial_regression#C, http://stackoverflow.com/a/36524956/83897
double PolynomialRegressionChannelStudy::calculateRegression(std::vector<double> &values, int degree) {
    gsl_multifit_linear_workspace *ws;
    gsl_matrix *cov, *X;
    gsl_vector *y, *c;
    double chisq;
    int obs;
    std::vector<double> coefficients;
    double point;

    int i, j;

    obs = values.size();

    X = gsl_matrix_alloc(obs, degree);
    y = gsl_vector_alloc(obs);
    c = gsl_vector_alloc(degree);
    cov = gsl_matrix_alloc(degree, degree);

    for (i=0; i<obs; i++) {
        for (j=0; j<degree; j++) {
            gsl_matrix_set(X, i, j, pow(i, j));
        }
        gsl_vector_set(y, i, values[i]);
    }

    ws = gsl_multifit_linear_alloc(obs, degree);
    gsl_multifit_linear(X, y, c, cov, &chisq, ws);

    // Get coefficients.
    for (i=0; i<degree; i++) {
        coefficients.push_back(gsl_vector_get(c, i));
    }

    gsl_multifit_linear_free(ws);
    gsl_matrix_free(X);
    gsl_matrix_free(cov);
    gsl_vector_free(y);
    gsl_vector_free(c);

    // Calculate the last data point in the series (that's all that is needed for
    // the purposes of this study).
    point = coefficients[0];
    for (i=1; i<degree; i++) {
        point += pow(obs - 1, i) * coefficients[i];
    }

    return point;
}

double PolynomialRegressionChannelStudy::calculateStandardDeviation(std::vector<double> &values) {
    double sum = 0;
    double squaredSum = 0;
    double mean;
    double variance;
    int valueCount = values.size();
    int i;

    if (valueCount == 0) {
        return 0.0;
    }

    for (i=0; i<valueCount; ++i) {
        sum += values[i];
        squaredSum += values[i] * values[i];
    }

    mean = sum / valueCount;
    variance = squaredSum / valueCount - mean * mean;

    return sqrt(variance);
}

std::map<std::string, double> PolynomialRegressionChannelStudy::tick() {
    std::map<std::string, double> valueMap;

    // ...

    return valueMap;
}
