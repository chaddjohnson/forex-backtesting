var Base = require('./Base');
var _ = require('lodash');

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
    var dataSegment = this.getDataSegment(this.getInput('length'));
    var dataSegmentLength = dataSegment.length;
    var averageLengthDataSegment = this.getDataSegment(this.getInput('averageLength') + 1).slice(0, this.getInput('averageLength'));
    var lastDataPoint = this.getLast();
    var low = 0.0;
    var high = 0.0;
    var K = 0.0;
    var D = 0.0;
    var returnValue = {};

    if (dataSegmentLength < this.getInput('length')) {
        return returnValue;
    }

    low = _.min(_.pluck(dataSegment, 'low'));
    high = _.max(_.pluck(dataSegment, 'high'));
    K = 100 * ((lastDataPoint.close - low) / (high - low));
    D = _.reduce(averageLengthDataSegment, function(memo, dataPoint) {
        if (typeof dataPoint[self.getOutputMapping('K')] === 'number') {
            return memo + dataPoint[self.getOutputMapping('K')];
        }
        else {
            return memo + K;
        }
    }, 0) / averageLengthDataSegment.length;

    returnValue[this.getOutputMapping('K')] = K;
    returnValue[this.getOutputMapping('D')] = D;

    return returnValue;
};

module.exports = StochasticOscillator;
