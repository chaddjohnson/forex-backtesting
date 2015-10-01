var StrategyBase = require('../Base');

function Base() {
    this.constructor = Base;
    StrategyBase.call(this);

    this.cumulativeData = [];
}

// Create a copy of the Base "class" prototype for use in this "class."
Base.prototype = Object.create(StrategyBase.prototype);

// Override tick() to work with prepared data.
Base.prototype.tick = function(dataPoint) {
    var previousDataPoint = this.cumulativeData[this.cumulativeData.length - 1];

    // Add the data point to the cumulative data.
    this.cumulativeData.push(dataPoint);

    if (previousDataPoint) {
        // Simulate expiry of and profit/loss related to positions held.
        this.closeExpiredPositions(previousDataPoint.close, dataPoint.timestamp);
    }
};

module.exports = Base;
