var Base = require('./Base');
var Call = require('../../positions/Call');
var Put = require('../../positions/Put');

function ReversalsCombined(symbol, configurations) {
    this.constructor = ReversalsCombined;
    Base.call(this, symbol, configurations);

    this.configurations = configurations;
}

ReversalsCombined.prototype = Object.create(Base.prototype);

ReversalsCombined.prototype.backtest = function(data, investment, profitability) {
    var self = this;
    var expirationMinutes = 5;
    var putNextTick = false;
    var callNextTick = false;
    var putThisConfiguration = false;
    var callThisConfiguration = false;
    var previousDataPoint;
    var dataPointCount = data.length;
    var previousDay = -1;
    var currentDay = -1;

    // For every data point...
    data.forEach(function(dataPoint, index) {
        var position = null;
        var timestampHour = new Date(dataPoint.timestamp).getHours();
        var timestampMinute = new Date(dataPoint.timestamp).getMinutes();

        currentDay = new Date(dataPoint.timestamp).getDay();

        if (currentDay !== previousDay) {
            investment = self.profitLoss * 0.02;
        }

        previousDay = currentDay;

        // Simulate the next tick.
        self.tick(dataPoint);

        // Only trade when the profitability is highest (11:30pm - 4pm CST).
        // Note that MetaTrader automatically converts timestamps to the current timezone in exported CSV files.
        if (timestampHour >= 16 && (timestampHour < 23 || (timestampHour === 23 && timestampMinute < 30))) {
            // Track the current data point as the previous data point for the next tick.
            previousDataPoint = dataPoint;

            self.putNextTick = false;
            self.callNextTick = false;

            return;
        }

        if (previousDataPoint && index < dataPointCount - 1) {
            if (putNextTick) {
                // Create a new position.
                position = new Put(self.getSymbol(), previousDataPoint.timestamp, previousDataPoint.close, investment, profitability, expirationMinutes);
                position.setShowTrades(self.getShowTrades());
                self.addPosition(position);
            }

            if (callNextTick) {
                // Create a new position.
                position = new Call(self.getSymbol(), previousDataPoint.timestamp, previousDataPoint.close, investment, profitability, expirationMinutes)
                position.setShowTrades(self.getShowTrades());
                self.addPosition(position);
            }
        }

        putNextTick = false;
        callNextTick = false;

        // For every configuration...
        self.configurations.forEach(function(configuration) {
            putThisConfiguration = true;
            callThisConfiguration = true;

            if (configuration.ema200 && configuration.ema100) {
                if (!dataPoint.ema200 || !dataPoint.ema100) {
                    putThisConfiguration = false;
                    callThisConfiguration = false;
                }

                // Determine if a downtrend is not occurring.
                if (putThisConfiguration && dataPoint.ema200 < dataPoint.ema100) {
                    putThisConfiguration = false;
                }

                // Determine if an uptrend is not occurring.
                if (callThisConfiguration && dataPoint.ema200 > dataPoint.ema100) {
                    callThisConfiguration = false;
                }
            }
            if (configuration.ema100 && configuration.ema50) {
                if (!dataPoint.ema100 || !dataPoint.ema50) {
                    putThisConfiguration = false;
                    callThisConfiguration = false;
                }

                // Determine if a downtrend is not occurring.
                if (putThisConfiguration && dataPoint.ema100 < dataPoint.ema50) {
                    putThisConfiguration = false;
                }

                // Determine if an uptrend is not occurring.
                if (callThisConfiguration && dataPoint.ema100 > dataPoint.ema50) {
                    callThisConfiguration = false;
                }
            }
            if (configuration.ema50 && configuration.sma13) {
                if (!dataPoint.ema50 || !dataPoint.sma13) {
                    putThisConfiguration = false;
                    callThisConfiguration = false;
                }

                // Determine if a downtrend is not occurring.
                if (putThisConfiguration && dataPoint.ema50 < dataPoint.sma13) {
                    putThisConfiguration = false;
                }

                // Determine if an uptrend is not occurring.
                if (callThisConfiguration && dataPoint.ema50 > dataPoint.sma13) {
                    callThisConfiguration = false;
                }
            }
            if (configuration.rsi) {
                if (typeof dataPoint[configuration.rsi.rsi] === 'number') {
                    // Determine if RSI is not above the overbought line.
                    if (putThisConfiguration && dataPoint[configuration.rsi.rsi] <= configuration.rsi.overbought) {
                        putThisConfiguration = false;
                    }

                    // Determine if RSI is not below the oversold line.
                    if (callThisConfiguration && dataPoint[configuration.rsi.rsi] >= configuration.rsi.oversold) {
                        callThisConfiguration = false;
                    }
                }
                else {
                    putThisConfiguration = false;
                    callThisConfiguration = false;
                }
            }
            if (configuration.stochastic) {
                if (typeof dataPoint[configuration.stochastic.K] === 'number' && typeof dataPoint[configuration.stochastic.D] === 'number') {
                    // Determine if stochastic is not above the overbought line.
                    if (putThisConfiguration && (dataPoint[configuration.stochastic.K] <= configuration.stochastic.overbought || dataPoint[configuration.stochastic.D] <= configuration.stochastic.overbought)) {
                        putThisConfiguration = false;
                    }

                    // Determine if stochastic is not below the oversold line.
                    if (callThisConfiguration && (dataPoint[configuration.stochastic.K] >= configuration.stochastic.oversold || dataPoint[configuration.stochastic.D] >= configuration.stochastic.oversold)) {
                        callThisConfiguration = false;
                    }
                }
                else {
                    putThisConfiguration = false;
                    callThisConfiguration = false;
                }
            }
            if (configuration.prChannel) {
                if (dataPoint[configuration.prChannel.upper] && dataPoint[configuration.prChannel.lower]) {
                    // Determine if the upper regression bound was not breached by the high price.
                    if (putThisConfiguration && (!dataPoint[configuration.prChannel.upper] || dataPoint.high <= dataPoint[configuration.prChannel.upper])) {
                        putThisConfiguration = false;
                    }

                    // Determine if the lower regression bound was not breached by the low price.
                    if (callThisConfiguration && (!dataPoint[configuration.prChannel.lower] || dataPoint.low >= dataPoint[configuration.prChannel.lower])) {
                        callThisConfiguration = false;
                    }
                }
                else {
                    putThisConfiguration = false;
                    callThisConfiguration = false;
                }
            }

            // Determine whether to trade next tick.
            putNextTick = putNextTick || putThisConfiguration;
            callNextTick = callNextTick || callThisConfiguration;
        });

        // Track the current data point as the previous data point for the next tick.
        previousDataPoint = dataPoint;
    });

    return self.getResults();
};

module.exports = ReversalsCombined;
