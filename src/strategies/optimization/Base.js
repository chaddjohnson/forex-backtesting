var StrategyBase = require('../base');

function Base() {
    this.constructor = Base;
    StrategyBase.call(this);
}

// Create a copy of the Base "class" prototype for use in this "class."
Base.prototype = Object.create(StrategyBase.prototype);

// Override tick() to work with prepared data.
Base.prototype.tick = function() {
    // ...
};

module.exports = Base;
