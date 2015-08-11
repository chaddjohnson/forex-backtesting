var Base = require('./base');

function Ema(name, inputs) {
    this.constructor = Ema;
    Base.call(this, name, inputs);

    if (!inputs.length) {
        throw 'No length input parameter provided to study.';
    }

    // Use the last data item as the first previous EMA value.
    this.previousEma = this.getLast().price;
}

// Create a copy of the Base "class" prototype for use in this "class."
Ema.prototype = Object.create(Base.prototype);

Ema.prototype.tick = function() {
    var K = 2 / (1 + this.getInput('length'));
    var ema = (this.getLast().price * K) + (this.previousEma * (1 - K));

    // Set the new EMA just calculated as the previous EMA.
    this.previousEma = ema;

    return ema;
};

module.exports = Ema;
