var _ = require('underscore');
var Base = require('./base');
var Call = require('../positions/Call');
var Put = require('../positions/Put');

function Reversals() {
    this.constructor = Reversals;
    Base.call(this);
}

// Create a copy of the Base "class" prototype for use in this "class."
Reversals.prototype = Object.create(Base.prototype);

Reversals.prototype.backtest = function(configuration, data, investment, profitability) {
    var self = this;
    var expirationMinutes = 5;
    var callConditions = [];
    var putConditions = [];
    var callNextTick = false;
    var putNextTick = false;
    var previousDataPoint;

    // For every data point...
    data.forEach(function(dataPoint) {
        // Simulate the next tick, and process update studies for the tick.
        self.tick(dataPoint);

        if (putNextTick) {
            // Create a new position.
            self.addPosition(new Put(dataPoint.symbol, dataPoint.timestamp, previousDataPoint.close, investment, profitability, expirationMinutes));
            putNextTick = false;
        }

        if (callNextTick) {
            // Create a new position.
            self.addPosition(new Call(dataPoint.symbol, dataPoint.timestamp, previousDataPoint.close, investment, profitability, expirationMinutes));
            callNextTick = false;
        }

        if (configuration.ema200 && configuration.ema100) {
            // Determine if a downtrend is occurring.
            putConditions.push(dataPoint.ema200 > dataPoint.ema100);

            // Determine if an uptrend is occurring.
            callConditions.push(dataPoint.ema200 < dataPoint.ema100);
        }
        if (configuration.ema100 && configuration.ema50) {
            // Determine if a downtrend is occurring.
            putConditions.push(dataPoint.ema100 > dataPoint.ema50);

            // Determine if an uptrend is occurring.
            callConditions.push(dataPoint.ema100 < dataPoint.ema50);
        }
        if (configuration.ema50 && configuration.sma13) {
            // Determine if a downtrend is occurring.
            putConditions.push(dataPoint.ema50 > dataPoint.sma13);

            // Determine if an uptrend is occurring.
            callConditions.push(dataPoint.ema50 < dataPoint.sma13);
        }
        if (configuration.ema50 && configuration.ema13) {
            // Determine if a downtrend is occurring.
            putConditions.push(dataPoint.ema50 > dataPoint.ema13);

            // Determine if an uptrend is occurring.
            callConditions.push(dataPoint.ema50 < dataPoint.ema13);
        }

        if (configuration.rsi) {
            // Determine if RSI is above the overbought line.
            putConditions.push(dataPoint[configuration.rsi.rsi] && dataPoint[configuration.rsi.rsi] >= configuration.rsi.overbought);

            // Determine if RSI is below the oversold line.
            callConditions.push(dataPoint[configuration.rsi.rsi] && dataPoint[configuration.rsi.rsi] <= configuration.rsi.oversold);
        }

        if (configuration.prChannel) {
            if (configuration.prChannel.close) {
                // Determine if the upper regression bound was breached by the close price.
                putConditions.push(dataPoint.close >= dataPoint[configuration.prChannel.upper]);

                // Determine if the lower regression bound was breached by the close price.
                callConditions.push(dataPoint.close <= dataPoint[configuration.prChannel.lower]);
            }
            else {
                // Determine if the upper regression bound was breached by the high price.
                putConditions.push(dataPoint.high >= dataPoint[configuration.prChannel.upper]);

                // Determine if the lower regression bound was breached by the low price.
                callConditions.push(dataPoint.low <= dataPoint[configuration.prChannel.lower]);
            }
        }

        if (configuration.trendPrChannel) {
            // Determine if a long-term downtrend is occurring.
            putConditions.push(previousDataPoint && dataPoint[configuration.trendPrChannel.regression] < previousDataPoint[configuration.trendPrChannel.regression]);

            // Determine if a long-term uptrand is occurring.
            callConditions.push(previousDataPoint && dataPoint[configuration.trendPrChannel.regression] > previousDataPoint[configuration.trendPrChannel.regression]);
        }

        // Determine if there is a significant gap (> 60 seconds) between the current timestamp and the previous timestamp.
        putConditions.push(previousDataPoint && (dataPoint.timestamp - previousDataPoint.timestamp) > 60 * 1000);
        callConditions.push(previousDataPoint && (dataPoint.timestamp - previousDataPoint.timestamp) > 60 * 1000);

        // Only do a PUT next tick if all necessary conditions for this strategy pass.
        putNextTick = putConditions.length > 0 && _(putConditions).filter(function(condition) {
            return condition === false;
        }).length === 0;

        // Only do a CALL next tick if all necessary conditions for this strategy pass.
        callNextTick = callConditions.length > 0 && _(callConditions).filter(function(condition) {
            return condition === false;
        }).length === 0;

        // Track the current data point as the previous data point for the next tick.
        previousDataPoint = dataPoint;
    });

    return this.getResults();
};

module.exports = Reversals;
