var Base = require('./base');
var _ = require('underscore');
var regression = require('regression');

function PolynomialRegressionChannel(inputs, outputMap) {
    this.constructor = PolynomialRegressionChannel;
    Base.call(this, inputs, outputMap);

    if (!inputs.length) {
        throw 'No length input parameter provided to study.';
    }
}

// Create a copy of the Base "class" prototype for use in this "class."
PolynomialRegressionChannel.prototype = Object.create(Base.prototype);

PolynomialRegressionChannel.prototype.calculateRegression = function(values, degree) {
    var data = [];
    var regressionOutput;

    // Format the data in a way the regression library expects.
    values.forEach(function(value, index) {
        data.push([index, value]);
    });

    // Use the library to calculate the regression.
    regressionOutput = regression('polynomial', data, degree);

    // Return the last regression data point.
    return regressionOutput.points[regressionOutput.points.length - 1][1];
};

PolynomialRegressionChannel.prototype.calculateStandardDeviation = function(values) {
    var average = _(values).reduce(function(total, value) {
        return total + value;
    }) / values.length;

    var squaredDeviations = _(values).reduce(function(total, value) {
        var deviation = value - average;
        var deviationSquared = deviation * deviation;

        return total + deviationSquared;
    }, 0);

    return Math.sqrt(squaredDeviations / values.length);
};

PolynomialRegressionChannel.prototype.tick = function() {
    var self = this;
    var dataSegment = self.getDataSegment(self.getInput('length'));
    var regressionValue = 0.0;
    var regressionStandardDeviation = 0.0;
    var upperValue = 0.0;
    var lowerValue = 0.0;
    var pastPrices = [];
    var pastRegressions = [];
    var returnValue = {};

    if (dataSegment.length < self.getInput('length')) {
        return returnValue;
    }

    // Calculate the regression.
    pastPrices = _(dataSegment).pluck('close');
    regressionValue = self.calculateRegression(pastPrices, self.getInput('degree'));

    // Calculate the standard deviations of the regression. If there is no regression data
    // available, then skip
    if (dataSegment[dataSegment.length - 2][self.getOutputMapping('regression')]) {
        // Build an array of regression data using only points that actually have regression data.
        dataSegment.forEach(function(dataPoint) {
            var dataPointRegression = dataPoint[self.getOutputMapping('regression')];
            if (dataPointRegression) {
                pastRegressions.push(dataPointRegression);
            }
        });

        // Calculate the standard deviation from the regressions.
        regressionStandardDeviation = self.calculateStandardDeviation(pastRegressions);

        // Calculate the upper and lower values. These should be 1.618 standard deviations
        // in distance from the regression line.
        upperValue = regressionValue + (regressionStandardDeviation * self.getInput('deviations'));
        lowerValue = regressionValue - (regressionStandardDeviation * self.getInput('deviations'));
    }
    else {
        upperValue = '';
        lowerValue = '';
    }

    returnValue[self.getOutputMapping('regression')] = regressionValue;
    returnValue[self.getOutputMapping('upper')] = upperValue;
    returnValue[self.getOutputMapping('lower')] = lowerValue;

    return returnValue;
};

module.exports = PolynomialRegressionChannel;
