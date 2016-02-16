var _ = require('lodash');
var StrategyBase = require('../Base');
var PositionModel = require('../../models/Position');
var uuid = require('node-uuid');

function Base(symbol, configuration, dataPointCount) {
    this.constructor = Base;
    StrategyBase.call(this, symbol);

    this.uuid = uuid.v4();
    this.configuration = configuration;
    this.dataPointCount = dataPointCount;
}

// Create a copy of the Base "class" prototype for use in this "class."
Base.prototype = Object.create(StrategyBase.prototype);

Base.expiredPositionsPool = [];

Base.prototype.getUuid = function() {
    return this.uuid;
};

Base.prototype.tick = function(dataPoint) {
    var self = this;
    var expiredPositions = [];

    if (self.tickPreviousDataPoint) {
        // Simulate expiry of and profit/loss related to positions held.
        expiredPositions = self.closeExpiredPositions(self.tickPreviousDataPoint.close, dataPoint.timestamp - 1000);

        expiredPositions.forEach(function(position) {
            Base.expiredPositionsPool.push(position);
        });

        self.tickPreviousDataPoint = dataPoint;
    }
    else {
        self.tickPreviousDataPoint = dataPoint;
    }
};

Base.prototype.getConfiguration = function() {
    return this.configuration;
};

Base.saveExpiredPositionsPool = function(callback) {
    if (Base.expiredPositionsPool.length === 0) {
        callback();
        return;
    }

    var self = this;
    var expiredPositionsBuffer = [];

    expiredPositionsBuffer = _.map(Base.expiredPositionsPool, function(position) {
        return {
            symbol: position.getSymbol(),
            strategyUuid: position.getStrategyUuid(),
            transactionType: position.getTransactionType(),
            timestamp: position.getTimestamp(),
            price: position.getPrice(),
            investment: position.getInvestment(),
            profitability: position.getProfitability(),
            closePrice: position.getClosePrice(),
            expirationTimestamp: position.getExpirationTimestamp(),
            closeTimestamp: position.getCloseTimestamp(),
            profitLoss: position.getProfitLoss()
        };
    });

    Base.expiredPositionsPool.forEach(function(position, index) {
        Base.expiredPositionsPool[index] = null;
    });

    Base.expiredPositionsPool = [];

    if (expiredPositionsBuffer.length === 0) {
        callback();
        return;
    }

    PositionModel.collection.insert(expiredPositionsBuffer, function() {
        expiredPositionsBuffer = [];
        callback();
    });
};

module.exports = Base;
