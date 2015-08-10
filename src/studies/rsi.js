var Base = require('./base');
var underscore = require('_');

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
    var RS = 0.0;

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
            return memo + dataPoint.ask;
        }, 0) / dataSegment.length;

        // Calculate the average loss.
        averageLoss = _(dataSegment).reduce(function(memo, dataPoint) {
            return memo + dataPoint.ask;
        }, 0) / dataSegment.length;
    }

    RS = averageGain / averageLoss;

    // Calculate RSI.
    return 100 - (100 / (1 + RS));
};

module.exports = Rsi;
