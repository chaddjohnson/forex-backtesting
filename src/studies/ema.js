var Base = require('./base');

function Ema(name, inputs) {
    this.constructor = Ema;
    Base.call(this, name, inputs);

    if (!inputs.length) {
        throw 'No length input parameter provided to study.';
    }
}

// Create a copy of the Base "class" prototype for use in this "class."
Ema.prototype = Object.create(Base.prototype);

Ema.prototype.tick = function() {
    var lastDataPoint = this.getLast();
    var K;
    var ema;

    if (!this.previousEma) {
        // Use the last data item as the first previous EMA value.
        this.previousEma = lastDataPoint.price;
    }

    K = 2 / (1 + this.getInput('length'));
    ema = (lastDataPoint.price * K) + (this.previousEma * (1 - K));

    // Set the new EMA just calculated as the previous EMA.
    this.previousEma = ema;

    return ema;
};

module.exports = Ema;
