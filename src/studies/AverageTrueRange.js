var _ = require('lodash');
var Base = require('./Base');

function AverageTrueRange(inputs, outputMap) {
    this.constructor = AverageTrueRange;
    Base.call(this, inputs, outputMap);

    this.previousTrValues = [];
    this.previousTrValuesCount = 0;

    if (!inputs.length) {
        throw 'No length input parameter provided to study.';
    }
}

// Create a copy of the Base "class" prototype for use in this "class."
AverageTrueRange.prototype = Object.create(Base.prototype);

AverageTrueRange.prototype.tick = function() {
    var self = this;
    var dataSegment = self.getDataSegment(self.getInput('length'));
    var dataSegmentLength = dataSegment.length;
    var previousDataPoint = self.getPrevious();
    var lastDataPoint = self.getLast();
    var tr = 0.0;
    var atr = 0.0;
    var returnValue = {};

    if (dataSegmentLength < self.getInput('length')) {
        return returnValue;
    }

    if (self.previousAtr) {
        // Calculate TR and ATR.
        tr = Math.max(
            lastDataPoint.high - lastDataPoint.low,
            Math.abs(lastDataPoint.high - previousDataPoint.close),
            Math.abs(lastDataPoint.low - previousDataPoint.close)
        );
        atr = ((self.previousAtr * (self.getInput('length') - 1)) + tr) / self.getInput('length');

        if (self.previousTrValues) {
            self.previousTrValues = [];
            self.previousTrValuesCount = 0;
        }
    }
    else {
        // Calculate and track the TR along with the previous ones.
        self.previousTrValues.push(lastDataPoint.high - lastDataPoint.low);
        self.previousTrValuesCount++;

        // Restrict the number of previous TR values tracked.
        if (self.previousTrValuesCount > self.getInput('length')) {
            self.previousTrValues = self.previousTrValues.slice(self.previousTrValuesCount - self.getInput('length'), self.previousTrValuesCount);
            self.previousTrValuesCount = self.getInput('length');
        }

        // Calculate the initial ATR if there are enough previous TR values.
        if (self.previousTrValuesCount === self.getInput('length')) {
            atr = _.reduce(self.previousTrValues, function(memo, previousTr) {
                return memo + previousTr;
            }, 0) / self.previousTrValuesCount;
        }
    }

    self.previousAtr = atr;
    returnValue[self.getOutputMapping('atr')] = self.previousAtr ? self.previousAtr : '';

    return returnValue;
};

module.exports = AverageTrueRange;
