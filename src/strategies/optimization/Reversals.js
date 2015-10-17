var Base = require('./Base');
var Call = require('../../positions/Call');
var Put = require('../../positions/Put');

function Reversals(configuration) {
    this.constructor = Reversals;
    Base.call(this, configuration);

    this.configuration = configuration;
    this.putNextTick = false;
    this.callNextTick = false;
}

// Create a copy of the Base "class" prototype for use in this "class."
Reversals.prototype = Object.create(Base.prototype);

Reversals.prototype.backtest = function(dataPoint, investment, profitability) {
    var expirationMinutes = 5;

    // Simulate the next tick.
    this.tick(dataPoint);

    if (this.previousDataPoint) {
        if (this.putNextTick) {
            // Create a new position.
            this.addPosition(new Put(dataPoint.symbol, (dataPoint.timestamp - 1000), this.previousDataPoint.close, investment, profitability, expirationMinutes));
        }

        if (this.callNextTick) {
            // Create a new position.
            this.addPosition(new Call(dataPoint.symbol, (dataPoint.timestamp - 1000), this.previousDataPoint.close, investment, profitability, expirationMinutes));
        }
    }

    this.putNextTick = true;
    this.callNextTick = true;

    if (this.configuration.ema200 && this.configuration.ema100) {
        if (!dataPoint.ema200 || !dataPoint.ema100) {
            this.putNextTick = false;
            this.callNextTick = false;
        }

        // Determine if a downtrend is not occurring.
        if (this.putNextTick && dataPoint.ema200 < dataPoint.ema100) {
            this.putNextTick = false;
        }

        // Determine if an uptrend is not occurring.
        if (this.callNextTick && dataPoint.ema200 > dataPoint.ema100) {
            this.callNextTick = false;
        }
    }
    if (this.configuration.ema100 && this.configuration.ema50) {
        if (!dataPoint.ema100 || !dataPoint.ema50) {
            this.putNextTick = false;
            this.callNextTick = false;
        }

        // Determine if a downtrend is not occurring.
        if (this.putNextTick && dataPoint.ema100 < dataPoint.ema50) {
            this.putNextTick = false;
        }

        // Determine if an uptrend is not occurring.
        if (this.callNextTick && dataPoint.ema100 > dataPoint.ema50) {
            this.callNextTick = false;
        }
    }
    if (this.configuration.ema50 && this.configuration.sma13) {
        if (!dataPoint.ema50 || !dataPoint.sma13) {
            this.putNextTick = false;
            this.callNextTick = false;
        }

        // Determine if a downtrend is not occurring.
        if (this.putNextTick && dataPoint.ema50 < dataPoint.sma13) {
            this.putNextTick = false;
        }

        // Determine if an uptrend is not occurring.
        if (this.callNextTick && dataPoint.ema50 > dataPoint.sma13) {
            this.callNextTick = false;
        }
    }
    if (this.configuration.rsi) {
        if (typeof dataPoint[this.configuration.rsi.rsi] === 'number') {
            // Determine if RSI is not above the overbought line.
            if (this.putNextTick && dataPoint[this.configuration.rsi.rsi] <= this.configuration.rsi.overbought) {
                this.putNextTick = false;
            }

            // Determine if RSI is not below the oversold line.
            if (this.callNextTick && dataPoint[this.configuration.rsi.rsi] >= this.configuration.rsi.oversold) {
                this.callNextTick = false;
            }
        }
        else {
            this.putNextTick = false;
            this.callNextTick = false;
        }
    }
    if (this.configuration.prChannel) {
        if (dataPoint[this.configuration.prChannel.upper] && dataPoint[this.configuration.prChannel.lower]) {
            // Determine if the upper regression bound was not breached by the high price.
            if (this.putNextTick && (!dataPoint[this.configuration.prChannel.upper] || dataPoint.high <= dataPoint[this.configuration.prChannel.upper])) {
                this.putNextTick = false;
            }

            // Determine if the lower regression bound was not breached by the low price.
            if (this.callNextTick && (!dataPoint[this.configuration.prChannel.lower] || dataPoint.low >= dataPoint[this.configuration.prChannel.lower])) {
                this.callNextTick = false;
            }
        }
        else {
            this.putNextTick = false;
            this.callNextTick = false;
        }
    }
    if (this.configuration.trendPrChannel) {
        if (this.previousDataPoint && dataPoint[this.configuration.trendPrChannel.regression] && this.previousDataPoint[this.configuration.trendPrChannel.regression]) {
            // Determine if a long-term downtrend is not occurring.
            if (this.putNextTick && dataPoint[this.configuration.trendPrChannel.regression] > this.previousDataPoint[this.configuration.trendPrChannel.regression]) {
                this.putNextTick = false;
            }

            // Determine if a long-term uptrend is not occurring.
            if (this.callNextTick && dataPoint[this.configuration.trendPrChannel.regression] < this.previousDataPoint[this.configuration.trendPrChannel.regression]) {
                this.callNextTick = false;
            }
        }
        else {
            this.putNextTick = false;
            this.callNextTick = false;
        }
    }

    // Determine if there is a significant gap (> 60 seconds) between the current timestamp and the previous timestamp.
    if ((this.putNextTick || this.callNextTick) && (!this.previousDataPoint || (dataPoint.timestamp - this.previousDataPoint.timestamp) !== 60 * 1000)) {
        this.putNextTick = false;
        this.callNextTick = false;
    }

    // Track the current data point as the previous data point for the next tick.
    this.previousDataPoint = dataPoint;
};

module.exports = Reversals;
