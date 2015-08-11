var Base = require('./base');
var _ = require('underscore');

function Sma(name, inputs) {
    this.constructor = Sma;
    Base.call(this, name, inputs);

    if (!inputs.length) {
        throw 'No length input parameter provided to study.';
    }
}

// Create a copy of the Base "class" prototype for use in this "class."
Sma.prototype = Object.create(Base.prototype);

Sma.prototype.tick = function() {
    var dataSegment = this.getDataSegment(this.getInput('length'));

    if (dataSegment.length < this.getInput('length')) {
        return '';
    }

    // Calculate SMA.
    return _(dataSegment).reduce(function(memo, dataPoint) {
        return memo + dataPoint.price;
    }, 0) / dataSegment.length;
};

module.exports = Sma;
