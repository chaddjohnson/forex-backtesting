var Base = require('./base');
var _ = require('underscore');

function Sma(inputs, outputMap) {
    this.constructor = Sma;
    Base.call(this, inputs, outputMap);

    if (!inputs.length) {
        throw 'No length input parameter provided to study.';
    }
}

// Create a copy of the Base "class" prototype for use in this "class."
Sma.prototype = Object.create(Base.prototype);

Sma.prototype.tick = function() {
    var dataSegment = this.getDataSegment(this.getInput('length'));
    var sma = 0.0;
    var returnValue = {};

    if (dataSegment.length < this.getInput('length')) {
        return '';
    }

    sma = _(dataSegment).reduce(function(memo, dataPoint) {
        return memo + dataPoint.close;
    }, 0) / dataSegment.length;

    returnValue[this.getOutputMapping('sma')] = sma;

    return returnValue;
};

module.exports = Sma;
