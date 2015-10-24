var _ = require('underscore');
var StrategyBase = require('../Base');
var PositionModel = require('../../models/Position');
var uuid = require('node-uuid');

// Static and available for all instances.
var expiredPositionsBuffer = [];

function Base(symbol, configuration) {
    this.constructor = Base;
    StrategyBase.call(this, symbol);

    this.uuid = uuid.v4();
    this.configuration = configuration;
}

// Create a copy of the Base "class" prototype for use in this "class."
Base.prototype = Object.create(StrategyBase.prototype);

Base.prototype.getUuid = function() {
    return this.uuid;
};

Base.prototype.tick = function(dataPoint) {
    var self = this;
    var expiredPositions = [];
    var i = 0;
    var expiredPositionsCount = 0;

    if (self.tickPreviousDataPoint) {
        // Simulate expiry of and profit/loss related to positions held.
        expiredPositions = self.closeExpiredPositions(self.tickPreviousDataPoint.close, dataPoint.timestamp - 1000);

        expiredPositions.forEach(function(position) {
            expiredPositionsBuffer.push({
                symbol: position.getSymbol(),
                strategyUuid: self.uuid,
                transactionType: position.getTransactionType(),
                timestamp: position.getTimestamp(),
                price: position.getPrice(),
                investment: position.getInvestment(),
                profitability: position.getProfitability(),
                closePrice: position.getClosePrice(),
                expirationTimestamp: position.getExpirationTimestamp(),
                closeTimestamp: position.getCloseTimestamp(),
                profitLoss: position.getProfitLoss()
            });
        });

        expiredPositionsCount = expiredPositions.length;

        for (i = 0; i < count; i++) {
            expiredPositions[i] = null;
        }
        expiredPositions.length = 0;

        // Periodically save the static buffer to the database.
        if (expiredPositionsBuffer.length >= 2000) {
            Base.saveExpiredPositionsBuffer(expiredPositionsBuffer.slice(0, 1000));
        }
    }

    self.tickPreviousDataPoint = dataPoint;
};

Base.prototype.getConfiguration = function() {
    return this.configuration;
};

// Static function.
Base.saveExpiredPositionsBuffer = function(expiredPositions) {
    if (expiredPositions.length === 0) {
        return;
    }
    PositionModel.collection.insert(expiredPositions, function() {
        var i = 0;
        var count = expiredPositions.length;

        for (i = 0; i < count; i++) {
            expiredPositions[i] = null;
        }
        expiredPositions.length = 0;
    });
};

module.exports = Base;
