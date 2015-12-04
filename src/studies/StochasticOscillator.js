var _ = require('lodash');
var Base = require('./Base');

function StochasticOscillator(inputs, outputMap) {
    this.constructor = StochasticOscillator;
    Base.call(this, inputs, outputMap);

    if (!inputs.length) {
        throw 'No length input parameter provided to study.';
    }
}

// Create a copy of the Base "class" prototype for use in this "class."
StochasticOscillator.prototype = Object.create(Base.prototype);

StochasticOscillator.prototype.tick = function() {
    var self = this;
    var dataSegment = self.getDataSegment(self.getInput('length'));
    var dataSegmentLength = dataSegment.length;
    var averageLengthDataSegment = [];
    var lastDataPoint = self.getLast();
    var low = 0.0;
    var high = 0.0;
    var highLowDifference = 0.0;
    var K = 0.0;
    var D = 0.0;
    var KOutputName = self.getOutputMapping('K');
    var returnValue = {};

    if (dataSegmentLength < self.getInput('length')) {
        return returnValue;
    }

    averageLengthDataSegment = dataSegment.slice(dataSegmentLength - self.getInput('averageLength'), dataSegmentLength);

    low = _.min(_.map(dataSegment, function(dataPoint) {
        return dataPoint.low;
    }));
    high = _.max(_.map(dataSegment, function(dataPoint) {
        return dataPoint.high;
    }));
    highLowDifference = high - low;
    K = highLowDifference > 0 ? 100 * ((lastDataPoint.close - low) / highLowDifference) : 0;
    D = _.reduce(averageLengthDataSegment, function(memo, dataPoint) {
        if (typeof dataPoint[KOutputName] === 'number') {
            return memo + dataPoint[KOutputName];
        }
        else {
            // Use the current K value for the last data point.
            return memo + K;
        }
    }, 0) / averageLengthDataSegment.length;

    returnValue[KOutputName] = K;
    returnValue[self.getOutputMapping('D')] = D;

    return returnValue;
};

module.exports = StochasticOscillator;
