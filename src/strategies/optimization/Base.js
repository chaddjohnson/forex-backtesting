var StrategyBase = require('../Base');

function Base() {
    this.constructor = Base;
    StrategyBase.call(this);
}

// Create a copy of the Base "class" prototype for use in this "class."
Base.prototype = Object.create(StrategyBase.prototype);

Base.prototype.tick = function(dataPoint) {
    if (this.previousDataPoint) {
        // Simulate expiry of and profit/loss related to positions held.
        this.closeExpiredPositions(this.previousDataPoint.close, dataPoint.timestamp);
    }
    this.previousDataPoint = dataPoint;
};

module.exports = Base;
