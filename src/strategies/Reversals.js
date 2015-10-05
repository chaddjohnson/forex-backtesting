var Base = require('./Base');
var studies = require('../studies');
var Call = require('../positions/Call');
var Put = require('../positions/Put');

// Define studies to use.
var studyDefinitions = [
    {
    //     study: studies.Ema,
    //     inputs: {
    //         length: 200
    //     },
    //     outputMap: {
    //         ema: 'ema200'
    //     }
    // },{
    //     study: studies.Ema,
    //     inputs: {
    //         length: 100
    //     },
    //     outputMap: {
    //         ema: 'ema100'
    //     }
    // },{
    //     study: studies.Ema,
    //     inputs: {
    //         length: 50
    //     },
    //     outputMap: {
    //         ema: 'ema50'
    //     }
    // },{
    //     study: studies.Sma,
    //     inputs: {
    //         length: 13
    //     },
    //     outputMap: {
    //         sma: 'sma13'
    //     }
    // },{
        study: studies.Rsi,
        inputs: {
            length: 5
        },
        outputMap: {
            rsi: 'rsi5'
        }
    },{
        study: studies.PolynomialRegressionChannel,
        inputs: {
            length: 100,
            degree: 4,
            deviations: 2.1
        },
        outputMap: {
            regression: 'prChannel',
            upper: 'prChannelUpper',
            lower: 'prChannelLower'
        }
    // },{
    //     study: studies.PolynomialRegressionChannel,
    //     inputs: {
    //         length: 600,
    //         degree: 2
    //     },
    //     outputMap: {
    //         regression: 'trendPrChannel'
    //     }
    }
];

function Reversals() {
    this.constructor = Reversals;
    Base.call(this);

    this.prepareStudies(studyDefinitions);
}

// Create a copy of the Base "class" prototype for use in this "class."
Reversals.prototype = Object.create(Base.prototype);

Reversals.prototype.backtest = function(data, investment, profitability) {
    var self = this;
    var callNextTick = false;
    var putNextTick = false;
    var movingAveragesDowntrending = false;
    var movingAveragesUptrending = false;
    var rsiOverbought = false;
    var rsiOversold = false;
    var regressionUpperBoundBreached = false;
    var regressionLowerBoundBreached = false;
    // var longRegressionDowntrending = false;
    // var longRegressionUptrending = false;
    var timeGapPresent = false;
    var previousDataPoint;
    var previousBalance = 0;
    var consecutiveLosses = 0;
    var maxConsecutiveLosses = 0;
    var lowestProfitLoss = 99999.0;

    // For every data point...
    data.forEach(function(dataPoint) {
        // Simulate the next tick, and process update studies for the tick.
        self.tick(dataPoint);

        if (previousDataPoint) {
            if (callNextTick) {
                // Create a new position.
                self.addPosition(new Call(dataPoint.symbol, dataPoint.timestamp, dataPoint.open, investment, profitability, 5));
                callNextTick = false;
            }

            if (putNextTick) {
                // Create a new position.
                self.addPosition(new Put(dataPoint.symbol, dataPoint.timestamp, dataPoint.open, investment, profitability, 5));
                putNextTick = false;
            }
        }

        // Determine if a downtrend is occurring.
        // movingAveragesDowntrending = dataPoint.ema200 && dataPoint.ema100 && dataPoint.ema50 && dataPoint.sma13 && dataPoint.ema200 > dataPoint.ema100 && dataPoint.ema100 > dataPoint.ema50 && dataPoint.ema50 > dataPoint.sma13;

        // Determine if an uptrend is occurring.
        // movingAveragesUptrending = dataPoint.ema200 && dataPoint.ema100 && dataPoint.ema50 && dataPoint.sma13 && dataPoint.ema200 < dataPoint.ema100 && dataPoint.ema100 < dataPoint.ema50 && dataPoint.ema50 < dataPoint.sma13;

        // Determine if RSI is above the overbought line.
        rsiOverbought = typeof dataPoint.rsi5 === 'number' && dataPoint.rsi5 >= 77;

        // Determine if RSI is below the oversold line.
        rsiOversold = typeof dataPoint.rsi5 === 'number' && dataPoint.rsi5 <= 23;

        // Determine if the upper regression bound was breached by the high.
        regressionUpperBoundBreached = dataPoint.prChannelUpper && dataPoint.high >= dataPoint.prChannelUpper;

        // Determine if the lower regression bound was breached by the low.
        regressionLowerBoundBreached = dataPoint.prChannelLower && dataPoint.low <= dataPoint.prChannelLower;

        // longRegressionUptrending = previousDataPoint && dataPoint.trendPrChannel && previousDataPoint.trendPrChannel && dataPoint.trendPrChannel > previousDataPoint.trendPrChannel;
        // longRegressionDowntrending = previousDataPoint && dataPoint.trendPrChannel && previousDataPoint.trendPrChannel && dataPoint.trendPrChannel < previousDataPoint.trendPrChannel;

        // Determine if there is a significant gap (> 60 seconds) between the current timestamp and the previous timestamp.
        timeGapPresent = previousDataPoint && (dataPoint.timestamp - previousDataPoint.timestamp) > 60 * 1000;

        // Determine whether to buy (CALL).
        if (rsiOversold && regressionLowerBoundBreached && !timeGapPresent) {
            callNextTick = true;
        }

        // Determine whether to buy (PUT).
        if (rsiOverbought && regressionUpperBoundBreached && !timeGapPresent) {
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

    // Show the results.
    console.log('SYMBOL:\t\t' + previousDataPoint.symbol);
    console.log('PROFIT/LOSS:\t$' + self.getProfitLoss());
    console.log('WIN RATE:\t' + self.getWinRate());
    console.log('WINS:\t\t' + self.getWinCount());
    console.log('LOSSES:\t\t' + self.getLoseCount());

    // Determine the max consecutive losses.
    this.positions.forEach(function(position) {
        position.getProfitLoss() === 0 ? consecutiveLosses++ : consecutiveLosses = 0;

        if (consecutiveLosses > maxConsecutiveLosses) {
            maxConsecutiveLosses = consecutiveLosses;
        }
    });

    console.log('MAX CONSECUTIVE LOSSES:\t' + maxConsecutiveLosses);
    console.log('LOWEST PROFIT/LOSS:\t$' + lowestProfitLoss);

    // Save the output to a file.
    this.saveOutput();
};

module.exports = Reversals;
