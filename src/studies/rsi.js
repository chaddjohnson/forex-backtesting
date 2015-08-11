var Base = require('./base');
var underscore = require('underscore');

function Rsi(name, data, inputs) {
    this.constructor = Rsi;
    Base.call(this, name, data, inputs);

    if (!inputs.length) {
        throw 'No length input parameter provided to study.';
    }
}

// Create a copy of the Base "class" prototype for use in this "class."
Rsi.prototype = Object.create(Base.prototype);

Rsi.prototype.tick = function() {
    var dataSegment = this.getDataSegment();
    var averageGain = 0.0;
    var averageLoss = 0.0;
    var rs = 0.0;
    var rsi = 0.0

    if (dataSegment.length < this.getInput('length')) {
        return null;
    }

    if (!inputs.previousAverageGain || !inputs.previousAverageLoss) {
        // Default previousEma to 0.
        inputs.previousEma = 0;
    }
    else {
        // Calculate the average gain.
        averageGain = _(dataSegment).reduce(function(memo, dataPoint) {
            return memo + dataPoint.price;
        }, 0) / dataSegment.length;

        // Calculate the average loss.
        averageLoss = _(dataSegment).reduce(function(memo, dataPoint) {
            return memo + dataPoint.price;
        }, 0) / dataSegment.length;
    }

    rs = averageGain / averageLoss;

    // Calculate RSI.
    rsi = 100 - (100 / (1 + RS));

    return rsi;
};

module.exports = Rsi;
