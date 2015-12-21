var Base = require('./Base');
var _ = require('lodash');
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
    var valuesCount = values.length;
    var i = 0;

    // Format the data in a way the regression library expects.
    for (i = 0; i < valuesCount; i++) {
        data[i] = [i, values[i]];
    }

    // Use the library to calculate the regression.
    regressionOutput = regression('polynomial', data, degree);

    // Return the last regression data point.
    return regressionOutput.points[regressionOutput.points.length - 1][1];
};

// Source: http://www.strchr.com/standard_deviation_in_one_pass
PolynomialRegressionChannel.prototype.calculateStandardDeviation = function(values) {
    var valuesCount = values.length;
    var sum = 0;
    var squaredSum = 0;
    var mean = 0.0;
    var variance = 0.0;
    var i = 0;

    if (valuesCount === 0) {
        return 0.0;
    }

    for (i = 0; i < valuesCount; ++i) {
       sum += values[i];
       squaredSum += values[i] * values[i];
    }

    mean = sum / valuesCount;
    variance = squaredSum / valuesCount - mean * mean;

    return Math.sqrt(variance);
};

PolynomialRegressionChannel.prototype.tick = function() {
    var self = this;
    var dataSegment = self.getDataSegment(self.getInput('length'));
    var dataSegmentLength = dataSegment.length;
    var regressionValue = 0.0;
    var regressionStandardDeviation = 0.0;
    var upperValue = 0.0;
    var lowerValue = 0.0;
    var upperValue2 = 0.0;
    var lowerValue2 = 0.0;
    var pastPrices = [];
    var pastRegressions = [];
    var returnValue = {};
    var dataPointRegression;
    var i = 0;
    var j = 0;
    var regressionOutputName = self.getOutputMapping('regression');

    if (dataSegmentLength < self.getInput('length')) {
        return returnValue;
    }

    // Calculate the regression.
    pastPrices = _.map(dataSegment, function(dataPoint) {
        return dataPoint.close;
    });
    regressionValue = self.calculateRegression(pastPrices, self.getInput('degree'));

    // Calculate the standard deviations of the regression. If there is no regression data
    // available, then skip.
    if (self.getInput('deviations') && dataSegment[dataSegmentLength - 2][regressionOutputName]) {
        // Build an array of regression data using only points that actually have regression data.
        for (i = 0; i < dataSegmentLength; i++) {
            if (i === dataSegmentLength - 1) {
                // Current regression.
                dataPointRegression = regressionValue;
            }
            else {
                // A previous regression.
                dataPointRegression = dataSegment[i][regressionOutputName];
            }

            if (dataPointRegression) {
                pastRegressions[j++] = dataPointRegression;
            }
        }

        // Calculate the standard deviation from the regressions.
        regressionStandardDeviation = self.calculateStandardDeviation(pastRegressions);

        // Calculate the upper and lower values.
        upperValue = regressionValue + (regressionStandardDeviation * self.getInput('deviations'));
        lowerValue = regressionValue - (regressionStandardDeviation * self.getInput('deviations'));
        upperValue2 = regressionValue + (regressionStandardDeviation * (self.getInput('deviations') + 0.382));
        lowerValue2 = regressionValue - (regressionStandardDeviation * (self.getInput('deviations') + 0.382));
    }
    else {
        upperValue = '';
        lowerValue = '';
        upperValue2 = '';
        lowerValue2 = '';
    }

    returnValue[regressionOutputName] = regressionValue;

    if (self.getInput('deviations')) {
        returnValue[self.getOutputMapping('upper')] = upperValue;
        returnValue[self.getOutputMapping('lower')] = lowerValue;
        returnValue[self.getOutputMapping('upper2')] = upperValue2;
        returnValue[self.getOutputMapping('lower2')] = lowerValue2;
    }

    return returnValue;
};

module.exports = PolynomialRegressionChannel;
