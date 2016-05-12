#include "studies/polynomialRegressionChannelStudy.h"

// Sources: https://rosettacode.org/wiki/Polynomial_regression#C, http://stackoverflow.com/a/36524956/83897
double PolynomialRegressionChannelStudy::calculateRegression(std::vector<double> &values, int degree) {
    gsl_multifit_linear_workspace *ws;
    gsl_matrix *cov, *X;
    gsl_vector *y, *c;
    double chisq;
    int obs;
    std::vector<double> coefficients;
    double point;

    int i = 0;
    int j = 0;

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
    int i = 0;

    if (valueCount == 0) {
        return 0.0;
    }

    for (i=0; i<valueCount; ++i) {
        sum += values[i];
        squaredSum += values[i] * values[i];
    }

    mean = sum / valueCount;
    variance = (squaredSum / valueCount) - (mean * mean);

    return sqrt(variance);
}

void PolynomialRegressionChannelStudy::tick() {
    Tick *lastTick = getLastTick();
    std::vector<Tick*> *dataSegment = new std::vector<Tick*>();
    int dataSegmentLength = 0;
    std::string regressionOutputName = getOutputMapping("regression");
    double regression;
    double regressionStandardDeviation;

    dataSegment = getDataSegment(getInput("length"));
    dataSegmentLength = dataSegment->size();

    // Record another past price.
    pastPrices.push_back(lastTick->at("close"));

    if (dataSegmentLength < getInput("length")) {
        return;
    }

    // Keep the correct number of past prices.
    while (pastPrices.size() > getInput("length")) {
        pastPrices.erase(pastPrices.begin());
    }

    // Calculate the regression.
    regression = calculateRegression(pastPrices, getInput("degree"));

    // Record another past regression.
    pastRegressions.push_back(regression);

    // Keep the correct number of past regressions.
    while (pastRegressions.size() > getInput("length")) {
        pastRegressions.erase(pastRegressions.begin());
    }

    // Calculate the standard deviation from the regressions.
    regressionStandardDeviation = calculateStandardDeviation(pastRegressions);

    setTickOutput(regressionOutputName, regression);

    // Calculate the upper and lower values.
    setTickOutput(getOutputMapping("upper"), regression + (regressionStandardDeviation * getInput("deviations")));
    setTickOutput(getOutputMapping("lower"), regression - (regressionStandardDeviation * getInput("deviations")));

    // Free memory.
    delete dataSegment;
}
