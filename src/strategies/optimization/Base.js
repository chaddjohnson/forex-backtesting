var _ = require('underscore');
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

Base.prototype.getUuid = function() {
    return this.uuid;
};

Base.prototype.tick = function(dataPoint, index, callback) {
    var self = this;
    var expiredPositions = [];
    var expiredPositionsBuffer = [];
    var expiredPositionsLength = 0;

    if (self.tickPreviousDataPoint) {
        // Simulate expiry of and profit/loss related to positions held.
        expiredPositions = self.closeExpiredPositions(self.tickPreviousDataPoint.close, dataPoint.timestamp - 1000);

        self.tickPreviousDataPoint = dataPoint;

        self.expiredPositions = self.expiredPositions || [];

        if (self.expiredPositions.length > 100 || index >= self.dataPointCount - 1) {
            expiredPositionsBuffer = _(self.expiredPositions).map(function(position) {
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

            self.expiredPositions = [];

            if (expiredPositionsBuffer.length === 0) {
                callback();
                return;
            }

            PositionModel.collection.insert(expiredPositionsBuffer, function() {
                expiredPositionsBuffer = [];
                callback();
            });
        }
        else {
            expiredPositionsLength = self.expiredPositions.length;

            expiredPositions.forEach(function(position) {
                self.expiredPositions[expiredPositionsLength++] = position;
            });

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
