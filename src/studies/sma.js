var Base = require('./base');
var underscore = require('underscore');

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
    var dataSegment = this.getDataSegment(this.getInput('length'));

    // Calculate SMA.
    var sma = _(dataSegment).reduce(function(memo, dataPoint) {
        return memo + dataPoint.price;
    }, 0) / dataSegment.length;

    return sma;
};

module.exports = Sma;
