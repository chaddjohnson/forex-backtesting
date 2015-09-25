var Base = require('./Base');

function Ema(inputs, outputMap) {
    this.constructor = Ema;
    Base.call(this, inputs, outputMap);

    if (!inputs.length) {
        throw 'No length input parameter provided to study.';
    }
}

// Create a copy of the Base "class" prototype for use in this "class."
Ema.prototype = Object.create(Base.prototype);

Ema.prototype.tick = function() {
    var lastDataPoint = this.getLast();
    var K = 0.0;
    var ema = 0.0;
    var returnValue = {};

    if (!this.previousEma) {
        // Use the last data item as the first previous EMA value.
        this.previousEma = lastDataPoint.close;
    }

    K = 2 / (1 + this.getInput('length'));
    ema = (lastDataPoint.close * K) + (this.previousEma * (1 - K));

    // Set the new EMA just calculated as the previous EMA.
    this.previousEma = ema;

    returnValue[this.getOutputMapping('ema')] = ema;

    return returnValue;
};

module.exports = Ema;
