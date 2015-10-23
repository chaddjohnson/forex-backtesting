var _ = require('underscore');
var StrategyBase = require('../Base');
var PositionModel = require('../../models/Position');
var uuid = require('node-uuid');

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

    if (self.tickPreviousDataPoint) {
        // Simulate expiry of and profit/loss related to positions held.
        expiredPositions = self.closeExpiredPositions(self.tickPreviousDataPoint.close, dataPoint.timestamp - 1000);

        // If there are any expired positions, save them.
        if (expiredPositions.length > 0) {
            expiredPositions = _(expiredPositions).map(function(position) {
                return {
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
                };
            });

            // Save the positions to the database.
            PositionModel.collection.insert(expiredPositions, function() {
                expiredPositions = [];
            });
        }
    }

    self.tickPreviousDataPoint = dataPoint;
};

Base.prototype.getConfiguration = function() {
    return this.configuration;
};

module.exports = Base;
