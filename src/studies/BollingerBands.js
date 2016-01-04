var Base = require('./Base');
var _ = require('lodash');

function BollingerBands(inputs, outputMap) {
    this.constructor = BollingerBands;
    Base.call(this, inputs, outputMap);

    if (!inputs.length) {
        throw 'No length input parameter provided to study.';
    }
}

// Create a copy of the Base "class" prototype for use in this "class."
BollingerBands.prototype = Object.create(Base.prototype);

BollingerBands.prototype.tick = function() {
    var self = this;
    var returnValue = {};
    var middle = 0.0;
    var middleStandardDeviation = 0.0;

    var dataSegment = self.getDataSegment(self.getInput('length'));
    var dataSegmentLength = dataSegment.length;

    if (dataSegmentLength < self.getInput('length')) {
        return returnValue;
    }

    middle = _.reduce(dataSegment, function(memo, dataPoint) {
        return memo + dataPoint.close;
    }, 0) / dataSegmentLength;

    middleStandardDeviation = self.calculateStandardDeviation(_.pluck(dataSegment, 'close'));

    returnValue[self.getOutputMapping('middle')] = middle;

    // Calculate the upper band using the deviation factor.
    returnValue[self.getOutputMapping('upper')] = middle + (self.getInput('deviations') * middleStandardDeviation);

    // Calculate the lower band using the deviation factor.
    returnValue[self.getOutputMapping('lower')] = middle - (self.getInput('deviations') * middleStandardDeviation);

    return returnValue;
};

module.exports = BollingerBands;