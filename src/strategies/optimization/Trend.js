var _ = require('lodash');
var Base = require('./Base');
var Call = require('../../positions/Call');

function Trend(symbol, configuration, dataPointCount) {
    this.constructor = Trend;
    Base.call(this, symbol, configuration, dataPointCount);

    this.configuration = configuration;
    this.callNextTick = false;
}

// Create a copy of the Base "class" prototype for use in this "class."
Trend.prototype = Object.create(Base.prototype);

// Inherit "static" methods and data from the base constructor function.
_.extend(Trend, Base);

Trend.prototype.addPosition = function(position) {
    position.setStrategyUuid(this.getUuid());

    Base.prototype.addPosition.call(this, position);
};

Trend.prototype.backtest = function(dataPoint, index, investment, profitability, callback) {
    var self = this;
    var expirationMinutes = 5;
    var timestampHour = new Date(dataPoint.timestamp).getHours();
    var timestampMinute = new Date(dataPoint.timestamp).getMinutes();

    // Simulate the next tick.
    self.tick(dataPoint, index, function() {
        // Only trade when the profitability is highest (11:30pm - 4pm CST).
        // Note that MetaTrader automatically converts timestamps to the current timezone in exported CSV files.
        if (timestampHour >= 0 && (timestampHour < 7 || (timestampHour === 7 && timestampMinute < 30))) {
            // Track the current data point as the previous data point for the next tick.
            self.previousDataPoint = dataPoint;

            self.callNextTick = false;

            return callback();
        }

        if (self.previousDataPoint) {
            if (self.callNextTick) {
                // Create a new position.
                self.addPosition(new Call(self.getSymbol(), (dataPoint.timestamp - 1000), self.previousDataPoint.close, investment, profitability, expirationMinutes));
            }
        }

        self.callNextTick = true;

        if (self.configuration.rsi) {
            if (typeof dataPoint[self.configuration.rsi.rsi] === 'number') {
                // Require RSI to be above 50.
                if (self.callNextTick && dataPoint[self.configuration.rsi.rsi] < 50) {
                    self.callNextTick = false;
                }
            }
            else {
                self.callNextTick = false;
            }
        }
        if (!self.callNextTick) {
            self.previousDataPoint = dataPoint;
            return callback();
        }
        if (self.configuration.adx && self.previousDataPoint) {
            if (typeof dataPoint[self.configuration.adx.pDI] === 'number' && typeof dataPoint[self.configuration.adx.ADX] === 'number') {
                // Require +DI to be > 20.
                if (dataPoint[self.configuration.adx.pDI] < 20) {
                    self.callNextTick = false;
                }

                // Require +DI to cross -DI going upward.
                if (dataPoint[self.configuration.adx.pDI] < dataPoint[self.configuration.adx.mDI]) {
                    self.callNextTick = false;
                }

                if (self.previousDataPoint[self.configuration.adx.pDI] > self.previousDataPoint[self.configuration.adx.mDI]) {
                    self.callNextTick = false;
                }

                // Require ADX to be high enough.
                if (dataPoint[self.configuration.adx.ADX] < 20) {
                    self.callNextTick = false;
                }
            }
            else {
                self.callNextTick = false;
            }
        }

        // Track the current data point as the previous data point for the next tick.
        self.previousDataPoint = dataPoint;
        callback();
    });
};

module.exports = Trend;
