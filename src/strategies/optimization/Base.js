var StrategyBase = require('../Base');

function Base(symbol, configuration) {
    this.constructor = Base;
    StrategyBase.call(this, symbol);

    this.configuration = configuration;
}

// Create a copy of the Base "class" prototype for use in this "class."
Base.prototype = Object.create(StrategyBase.prototype);

Base.prototype.tick = function(dataPoint) {
    if (this.tickPreviousDataPoint) {
        // Simulate expiry of and profit/loss related to positions held.
        this.closeExpiredPositions(this.tickPreviousDataPoint.close, dataPoint.timestamp - 1000);
    }
    this.tickPreviousDataPoint = dataPoint;
};

Base.prototype.getConfiguration = function() {
    return this.configuration;
};

module.exports = Base;
