var _ = require('underscore');
var Base = require('./Base');
var Call = require('../../positions/Call');
var Put = require('../../positions/Put');

function Reversals() {
    this.constructor = Reversals;
    Base.call(this);
}

// Create a copy of the Base "class" prototype for use in this "class."
Reversals.prototype = Object.create(Base.prototype);

Reversals.prototype.backtest = function(configuration, data, investment, profitability) {
    var index = 0;
    var dataPointCount = data.length;
    var dataPoint;
    var expirationMinutes = 5;
    var putNextTick = false;
    var callNextTick = false;
    var previousDataPoint = null;

    // For every data point...
    //data.forEach(function(dataPoint) {
    for (index = 0; index < dataPointCount; index++) {
        dataPoint = data[index];

        // Simulate the next tick, and process update studies for the tick.
        this.tick(dataPoint);

        if (previousDataPoint && index < dataPointCount - 1) {
            if (putNextTick) {
                // Create a new position.
                this.addPosition(new Put(dataPoint.symbol, (dataPoint.timestamp - 1000), previousDataPoint.close, investment, profitability, expirationMinutes));
            }

            if (callNextTick) {
                // Create a new position.
                this.addPosition(new Call(dataPoint.symbol, (dataPoint.timestamp - 1000), previousDataPoint.close, investment, profitability, expirationMinutes));
            }
        }

        putNextTick = true;
        callNextTick = true;

        if (configuration.ema200 && configuration.ema100) {
            if (!dataPoint.ema200 || !dataPoint.ema100) {
                putNextTick = false;
                callNextTick = false;
            }

            // Determine if a downtrend is not occurring.
            if (putNextTick && dataPoint.ema200 < dataPoint.ema100) {
                putNextTick = false;
            }

            // Determine if an uptrend is not occurring.
            if (callNextTick && dataPoint.ema200 > dataPoint.ema100) {
                callNextTick = false;
            }
        }
        if (configuration.ema100 && configuration.ema50) {
            if (!dataPoint.ema100 || !dataPoint.ema50) {
                putNextTick = false;
                callNextTick = false;
            }

            // Determine if a downtrend is not occurring.
            if (putNextTick && dataPoint.ema100 < dataPoint.ema50) {
                putNextTick = false;
            }

            // Determine if an uptrend is not occurring.
            if (callNextTick && dataPoint.ema100 > dataPoint.ema50) {
                callNextTick = false;
            }
        }
        if (configuration.ema50 && configuration.sma13) {
            if (!dataPoint.ema50 || !dataPoint.sma13) {
                putNextTick = false;
                callNextTick = false;
            }

            // Determine if a downtrend is not occurring.
            if (putNextTick && dataPoint.ema50 < dataPoint.sma13) {
                putNextTick = false;
            }

            // Determine if an uptrend is not occurring.
            if (callNextTick && dataPoint.ema50 > dataPoint.sma13) {
                callNextTick = false;
            }
        }
        if (configuration.ema50 && configuration.ema13) {
            if (!dataPoint.ema50 || !dataPoint.ema13) {
                putNextTick = false;
                callNextTick = false;
            }

            // Determine if a downtrend is not occurring.
            if (putNextTick && dataPoint.ema50 < dataPoint.ema13) {
                putNextTick = false;
            }

            // Determine if an uptrend is not occurring.
            if (callNextTick && dataPoint.ema50 > dataPoint.ema13) {
                callNextTick = false;
            }
        }

        if (configuration.rsi) {
            if (typeof dataPoint[configuration.rsi.rsi] === 'number') {
                // Determine if RSI is not above the overbought line.
                if (putNextTick && dataPoint[configuration.rsi.rsi] <= configuration.rsi.overbought) {
                    putNextTick = false;
                }

                // Determine if RSI is not below the oversold line.
                if (callNextTick && dataPoint[configuration.rsi.rsi] >= configuration.rsi.oversold) {
                    callNextTick = false;
                }
            }
            else {
                putNextTick = false;
                callNextTick = false;
            }
        }

        if (configuration.prChannel) {
            if (dataPoint[configuration.prChannel.upper] && dataPoint[configuration.prChannel.lower]) {
                // Determine if the upper regression bound was not breached by the high price.
                if (putNextTick && (!dataPoint[configuration.prChannel.upper] || dataPoint.high <= dataPoint[configuration.prChannel.upper])) {
                    putNextTick = false;
                }

                // Determine if the lower regression bound was not breached by the low price.
                if (callNextTick && (!dataPoint[configuration.prChannel.lower] || dataPoint.low >= dataPoint[configuration.prChannel.lower])) {
                    callNextTick = false;
                }
            }
            else {
                putNextTick = false;
                callNextTick = false;
            }
        }

        if (configuration.trendPrChannel) {
            if (previousDataPoint && dataPoint[configuration.trendPrChannel.regression] && previousDataPoint[configuration.trendPrChannel.regression]) {
                // Determine if a long-term downtrend is not occurring.
                if (putNextTick && dataPoint[configuration.trendPrChannel.regression] > previousDataPoint[configuration.trendPrChannel.regression]) {
                    putNextTick = false;
                }

                // Determine if a long-term uptrend is not occurring.
                if (callNextTick && dataPoint[configuration.trendPrChannel.regression] < previousDataPoint[configuration.trendPrChannel.regression]) {
                    callNextTick = false;
                }
            }
            else {
                putNextTick = false;
                callNextTick = false;
            }
        }

        // Determine if there is a significant gap (> 60 seconds) between the current timestamp and the previous timestamp.
        if ((putNextTick || callNextTick) && (!previousDataPoint || (dataPoint.timestamp - previousDataPoint.timestamp) !== 60 * 1000)) {
            putNextTick = false;
            callNextTick = false;
        }

        // Track the current data point as the previous data point for the next tick.
        previousDataPoint = dataPoint;
    };

    return this.getResults();
};

module.exports = Reversals;
