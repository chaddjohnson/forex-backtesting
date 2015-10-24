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

Base.prototype.tick = function(dataPoint, callback) {
    var self = this;
    var expiredPositions = [];
    var expiredPositionsBuffer = [];

    if (self.tickPreviousDataPoint) {
        // Simulate expiry of and profit/loss related to positions held.
        expiredPositions = self.closeExpiredPositions(self.tickPreviousDataPoint.close, dataPoint.timestamp - 1000);

        self.tickPreviousDataPoint = dataPoint;

        if (expiredPositions.length > 0) {
            expiredPositionsBuffer = _(expiredPositions).map(function(position) {
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

            expiredPositions = [];

            PositionModel.collection.insert(expiredPositionsBuffer, function() {
                expiredPositionsBuffer = [];
                callback();
            });
        }
        else {
            callback();
        }
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
