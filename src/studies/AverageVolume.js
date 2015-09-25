var Base = require('./Base');
var _ = require('underscore');

function AverageVolume(inputs, outputMap) {
    this.constructor = AverageVolume;
    Base.call(this, inputs, outputMap);

    if (!inputs.length) {
        throw 'No length input parameter provided to study.';
    }
}

// Create a copy of the Base "class" prototype for use in this "class."
AverageVolume.prototype = Object.create(Base.prototype);

AverageVolume.prototype.tick = function() {
    var dataSegment = this.getDataSegment(this.getInput('length'));
    var average = 0.0;
    var returnValue = {};

    if (dataSegment.length < this.getInput('length')) {
        return returnValue;
    }

    average = _(dataSegment).reduce(function(memo, dataPoint) {
        return memo + dataPoint.volume;
    }, 0) / dataSegment.length;

    returnValue[this.getOutputMapping('average')] = average;

    return returnValue;
};

module.exports = AverageVolume;
