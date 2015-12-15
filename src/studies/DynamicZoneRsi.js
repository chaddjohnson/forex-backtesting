var Base = require('./Base');
var _ = require('lodash');

function DynamicZoneRsi(inputs, outputMap) {
    this.constructor = DynamicZoneRsi;
    Base.call(this, inputs, outputMap);

    if (!inputs.length) {
        throw 'No length input parameter provided to study.';
    }

    this.previousAverageGain = null;
    this.previousAverageLoss = null;
    this.previousRsiValues = [];
    this.previousRsiMovingAverageValues = [];
}

// Create a copy of the Base "class" prototype for use in this "class."
DynamicZoneRsi.prototype = Object.create(Base.prototype);

DynamicZoneRsi.prototype.calculateInitialAverageGain = function(initialDataPoint) {
    var previousDataPoint = initialDataPoint;

    return _.reduce(this.dataSegment, function(memo, dataPoint) {
        var gain = dataPoint.close > previousDataPoint.close ? dataPoint.close - previousDataPoint.close : 0;

        previousDataPoint = dataPoint;

        return memo + gain;
    }, 0) / this.dataSegmentLength;
};

DynamicZoneRsi.prototype.calculateInitialAverageLoss = function(initialDataPoint) {
    var previousDataPoint = initialDataPoint;

    return _.reduce(this.dataSegment, function(memo, dataPoint) {
        var loss = dataPoint.close < previousDataPoint.close ? previousDataPoint.close - dataPoint.close : 0;

        previousDataPoint = dataPoint;

        return memo + loss;
    }, 0) / this.dataSegmentLength;
};

DynamicZoneRsi.prototype.calculateRsi = function() {
    var previousDataPoint = this.getPrevious();
    var lastDataPoint = this.getLast();
    var averageGain = 0.0;
    var averageLoss = 0.0;
    var currentGain = 0.0;
    var currentLoss = 0.0;
    var RS = 0.0;

    // Calculate the current gain and the current loss.
    currentGain = lastDataPoint.close > previousDataPoint.close ? lastDataPoint.close - previousDataPoint.close : 0;
    currentLoss = lastDataPoint.close < previousDataPoint.close ? previousDataPoint.close - lastDataPoint.close : 0;

    if (!this.previousAverageGain || !this.previousAverageLoss) {
        averageGain = this.previousAverageGain = this.calculateInitialAverageGain(lastDataPoint);
        averageLoss = this.previousAverageLoss = this.calculateInitialAverageLoss(lastDataPoint);
    }
    else {
        averageGain = this.previousAverageGain = ((this.previousAverageGain * (this.getInput('length') - 1)) + currentGain) / this.getInput('length');
        averageLoss = this.previousAverageLoss = ((this.previousAverageLoss * (this.getInput('length') - 1)) + currentLoss) / this.getInput('length');
    }

    RS = averageLoss > 0 ? averageGain / averageLoss : 0;

    return 100 - (100 / (1 + RS));
};

DynamicZoneRsi.prototype.tick = function() {
    var self = this;
    var rsi = 0.0;
    var previousRsiValuesCount = 0;
    var rsiMovingAverage = 0.0;
    var rsiMovingAverageStandardDeviation = 0.0;
    var returnValue = {};

    self.dataSegment = self.getDataSegment(self.getInput('length'));
    self.dataSegmentLength = self.dataSegment.length;

    if (self.dataSegmentLength < self.getInput('length')) {
        return returnValue;
    }

    // Calculate the normal RSI.
    rsi = self.calculateRsi();
    returnValue[self.getOutputMapping('rsi')] = rsi;

    // Track the RSI as a previous one.
    previousRsiValuesCount = self.previousRsiValues.push(rsi);

    // Track only the necessary number of previous RSI values.
    if (previousRsiValuesCount > self.getInput('bandsLength')) {
        self.previousRsiValues = self.previousRsiValues.slice(previousRsiValuesCount - self.getInput('bandsLength'), previousRsiValuesCount);
    }

    if (self.previousRsiValues.length < self.getInput('bandsLength')) {
        return returnValue;
    }

    // Calculate a moving average of the RSI.
    rsiMovingAverage = _.reduce(self.previousRsiValues, function(memo, previousRsi) {
        return memo + previousRsi;
    }, 0) / self.getInput('bandsLength');

    // Track the moving average as a previous one.
    self.previousRsiMovingAverageValues.push(rsiMovingAverage);

    // Calculate the standard deviation of the moving average.
    rsiMovingAverageStandardDeviation = self.calculateStandardDeviation(self.previousRsiMovingAverageValues);

    // Calculate the upper band using the deviation factor.
    returnValue[self.getOutputMapping('upper')] = rsiMovingAverage + (self.getInput('deviations') * rsiMovingAverageStandardDeviation);

    // Calculate the lower band using the deviation factor.
    returnValue[self.getOutputMapping('lower')] = rsiMovingAverage - (self.getInput('deviations') * rsiMovingAverageStandardDeviation);

    return returnValue;
};

module.exports = DynamicZoneRsi;
