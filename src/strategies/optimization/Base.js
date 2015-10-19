var StrategyBase = require('../Base');

function Base(symbol, configuration) {
    this.constructor = Base;
    StrategyBase.call(this, symbol);

    this.configuration = configuration;
}

// Create a copy of the Base "class" prototype for use in this "class."
Base.prototype = Object.create(StrategyBase.prototype);

Base.prototype.tick = function(dataPoint) {
    // Simulate expiry of and profit/loss related to positions held.
    this.closeExpiredPositions(dataPoint.open, dataPoint.timestamp);
};

Base.prototype.getConfiguration = function() {
    return this.configuration;
};

module.exports = Base;
