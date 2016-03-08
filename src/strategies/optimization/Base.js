var _ = require('lodash');
var StrategyBase = require('../Base');
var PositionModel = require('../../models/Position');
var uuid = require('node-uuid');

function Base(symbol, group, configuration, dataPointCount) {
    this.constructor = Base;
    StrategyBase.call(this, symbol);

    this.group = group;
    this.uuid = uuid.v4();
    this.configuration = configuration;
    this.dataPointCount = dataPointCount;
}

// Create a copy of the Base "class" prototype for use in this "class."
Base.prototype = Object.create(StrategyBase.prototype);

Base.prototype.getGroup = function() {
    return this.group;
};

Base.prototype.getUuid = function() {
    return this.uuid;
};

Base.prototype.tick = function(dataPoint, index, callback) {
    var self = this;

    if (self.tickPreviousDataPoint) {
        // Simulate expiry of and profit/loss related to positions held.
        self.closeExpiredPositions(self.tickPreviousDataPoint.close, dataPoint.timestamp - 1000);

        self.tickPreviousDataPoint = dataPoint;

        callback();
    }
    else {
        self.tickPreviousDataPoint = dataPoint;
        callback();
    }
};

Base.prototype.getConfiguration = function() {
    return this.configuration;
};

module.exports = Base;
