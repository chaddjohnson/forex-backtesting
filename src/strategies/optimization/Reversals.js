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
    var callNextTick = false;
    var putNextTick = false;
    var movingAveragesDowntrending = false;
    var movingAveragesUptrending = false;
    var rsiOverbought = false;
    var rsiOversold = false;
    var regressionUpperBoundBreached = false;
    var regressionLowerBoundBreached = false;
    var longRegressionDowntrending = false;
    var longRegressionUptrending = false;
    var timeGapPresent = false;
    var previousDataPoint;
    var previousBalance = 0;
    var consecutiveLosses = 0;
    var maxConsecutiveLosses = 0;
    var lowestProfitLoss = 99999.0;
    var results = {};

    // For every data point...
    data.forEach(function(dataPoint) {
        // Simulate the next tick, and process update studies for the tick.
        self.tick(dataPoint);

        if (callNextTick) {
            // Create a new position.
            self.addPosition(new Call(dataPoint.symbol, dataPoint.timestamp, previousDataPoint.close, investment, profitability, 5));
            callNextTick = false;
        }

        if (putNextTick) {
            // Create a new position.
            self.addPosition(new Put(dataPoint.symbol, dataPoint.timestamp, previousDataPoint.close, investment, profitability, 5));
            putNextTick = false;
        }

        // Determine if a downtrend is occurring.
        movingAveragesDowntrending = dataPoint.ema200 > dataPoint.ema100 && dataPoint.ema100 > dataPoint.ema50 && dataPoint.ema50 > dataPoint.sma13;

        // Determine if an uptrend is occurring.
        movingAveragesUptrending = dataPoint.ema200 < dataPoint.ema100 && dataPoint.ema100 < dataPoint.ema50 && dataPoint.ema50 < dataPoint.sma13;

        if (configuration.rsi) {
            // Determine if RSI is above the overbought line.
            rsiOverbought = dataPoint.rsi5 && dataPoint.rsi5 >= configuration.rsi.overbought;

            // Determine if RSI is below the oversold line.
            rsiOversold = dataPoint.rsi5 && dataPoint.rsi5 <= configuration.rsi.oversold;
        }

        // Determine if the upper regression bound was breached by the high.
        regressionUpperBoundBreached = dataPoint.high >= dataPoint.prChannelUpper250;

        // Determine if the lower regression bound was breached by the low.
        regressionLowerBoundBreached = dataPoint.low <= dataPoint.prChannelLower250;

        longRegressionUptrending = previousDataPoint && dataPoint.prChannel600 > previousDataPoint.prChannel600;
        longRegressionDowntrending = previousDataPoint && dataPoint.prChannel600 < previousDataPoint.prChannel600;

        // Determine if there is a significant gap (> 60 seconds) between the current timestamp and the previous timestamp.
        timeGapPresent = previousDataPoint && (dataPoint.timestamp - previousDataPoint.timestamp) > 60 * 1000;

        // Determine whether to buy (CALL).
        if (movingAveragesUptrending && rsiOversold && regressionLowerBoundBreached && !timeGapPresent) {
            callNextTick = true;
        }

        // Determine whether to buy (PUT).
        if (movingAveragesDowntrending && rsiOverbought && regressionUpperBoundBreached  && !timeGapPresent) {
            putNextTick = true;
        }

        // Determine and display the lowest profit/loss.
        if (self.getProfitLoss() < lowestProfitLoss) {
            lowestProfitLoss = self.getProfitLoss();
        }

        if (self.getProfitLoss() !== previousBalance) {
            // console.log('BALANCE: $' + self.getProfitLoss());
            // console.log();
        }
        previousBalance = self.getProfitLoss();

        // Track the current data point as the previous data point for the next tick.
        previousDataPoint = dataPoint;
    });

    // Determine the max consecutive losses.
    this.positions.forEach(function(position) {
        position.getProfitLoss() === 0 ? consecutiveLosses++ : consecutiveLosses = 0;

        if (consecutiveLosses > maxConsecutiveLosses) {
            maxConsecutiveLosses = consecutiveLosses;
        }
    });

    results = {
        profitLoss: self.getProfitLoss(),
        winRate: self.getWinRate(),
        wins: self.getWinCount(),
        losses: self.getLoseCount(),
        maxConsecutiveLosses: maxConsecutiveLosses,
        lowestProfitLoss: lowestProfitLoss
    };

    return results;
};

module.exports = Reversals;
