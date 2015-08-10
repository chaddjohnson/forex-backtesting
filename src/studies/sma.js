var Base = require('./base');
var underscore = require('_');

function Sma(name, data, inputs) {
    this.constructor = Sma;
    Base.call(this, name, data, inputs);

    if (!inputs.length) {
        throw 'No length input parameter provided to study.';
    }
}

// Create a copy of the Base "class" prototype for use in this "class."
Sma.prototype = Object.create(Base.prototype);

Sma.prototype.tick = function() {
    var dataSegment = this.getDataSegment();

    // Calculate SMA.
    return _(dataSegment).reduce(function(memo, dataPoint) {
        return memo + dataPoint.ask;
    }, 0) / dataSegment.length;
};

module.exports = Sma;
