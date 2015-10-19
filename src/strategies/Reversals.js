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
            length: 7
        },
        outputMap: {
            rsi: 'rsi'
        }
    },{
        study: studies.PolynomialRegressionChannel,
        inputs: {
            length: 100,
            degree: 4,
            deviations: 2.0
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

function Reversals(symbol) {
    this.constructor = Reversals;
    Base.call(this, symbol);

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

    // For every data point...
    data.forEach(function(dataPoint) {
        // Simulate the next tick, and process studies for the tick.
        self.tick(dataPoint);

        if (previousDataPoint) {
            if (callNextTick) {
                // Create a new position.
                self.addPosition(new Call(self.getSymbol(), (dataPoint.timestamp - 1000), previousDataPoint.close, investment, profitability, 5));
                callNextTick = false;
            }

            if (putNextTick) {
                // Create a new position.
                self.addPosition(new Put(self.getSymbol(), (dataPoint.timestamp - 1000), previousDataPoint.close, investment, profitability, 5));
                putNextTick = false;
            }
        }

        // Determine if a downtrend is occurring.
        // movingAveragesDowntrending = dataPoint.ema200 && dataPoint.ema100 && dataPoint.ema50 && dataPoint.sma13 && dataPoint.ema200 > dataPoint.ema100 && dataPoint.ema100 > dataPoint.ema50 && dataPoint.ema50 > dataPoint.sma13;

        // Determine if an uptrend is occurring.
        // movingAveragesUptrending = dataPoint.ema200 && dataPoint.ema100 && dataPoint.ema50 && dataPoint.sma13 && dataPoint.ema200 < dataPoint.ema100 && dataPoint.ema100 < dataPoint.ema50 && dataPoint.ema50 < dataPoint.sma13;

        // Determine if RSI is above the overbought line.
        rsiOverbought = typeof dataPoint.rsi === 'number' && dataPoint.rsi >= 77;

        // Determine if RSI is below the oversold line.
        rsiOversold = typeof dataPoint.rsi === 'number' && dataPoint.rsi <= 23;

        // Determine if the upper regression bound was breached by the high.
        regressionUpperBoundBreached = dataPoint.prChannelUpper && dataPoint.high >= dataPoint.prChannelUpper;

        // Determine if the lower regression bound was breached by the low.
        regressionLowerBoundBreached = dataPoint.prChannelLower && dataPoint.low <= dataPoint.prChannelLower;

        // longRegressionUptrending = previousDataPoint && dataPoint.trendPrChannel && previousDataPoint.trendPrChannel && dataPoint.trendPrChannel > previousDataPoint.trendPrChannel;
        // longRegressionDowntrending = previousDataPoint && dataPoint.trendPrChannel && previousDataPoint.trendPrChannel && dataPoint.trendPrChannel < previousDataPoint.trendPrChannel;

        // Determine if there is a significant gap (> 60 seconds) between the current timestamp and the previous timestamp.
        timeGapPresent = previousDataPoint && (dataPoint.timestamp - previousDataPoint.timestamp) !== 60 * 1000;

        // Determine whether to buy (CALL).
        if (rsiOversold && regressionLowerBoundBreached && !timeGapPresent) {
            callNextTick = true;
        }

        // Determine whether to buy (PUT).
        if (rsiOverbought && regressionUpperBoundBreached && !timeGapPresent) {
            putNextTick = true;
        }

        if (self.getProfitLoss() !== previousBalance) {
            // console.log('BALANCE: $' + self.getProfitLoss());
            // console.log();
        }
        previousBalance = self.getProfitLoss();

        // Track the current data point as the previous data point for the next tick.
        previousDataPoint = dataPoint;
    });

    console.log(self.getResults());
};

module.exports = Reversals;
