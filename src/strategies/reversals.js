var studies = require('../studies');
var Base = require('./base');
var Call = require('../positions/call');
var Put = require('../positions/put');

// Define studies to use.
var studyDefinitions = [
    {
        study: studies.Ema,
        inputs: {
            length: 200
        },
        outputMap: {
            ema: 'ema200'
        }
    },{
        study: studies.Ema,
        inputs: {
            length: 100
        },
        outputMap: {
            ema: 'ema100'
        }
    },{
        study: studies.Ema,
        inputs: {
            length: 50
        },
        outputMap: {
            ema: 'ema50'
        }
    },{
        study: studies.Sma,
        inputs: {
            length: 13
        },
        outputMap: {
            sma: 'sma13'
        }
    },{
        study: studies.Rsi,
        inputs: {
            length: 7
        },
        outputMap: {
            rsi: 'rsi7'
        }
    },{
        study: studies.PolynomialRegressionCurve,
        inputs: {
            length: 200,
            deviations: 1.95
        },
        outputMap: {
            regression: 'pReg200',
            upper: 'pRegUpper200',
            lower: 'pRegLower200'
        }
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
    var downtrending = false;
    var uptrending = false;
    var rsiOverbought = false;
    var rsiOversold = false;
    var regressionUpperBoundBreached = false;
    var regressionLowerBoundBreached = false;
    var timeGapPresent = false;
    var previousDataPoint;

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
        downtrending = dataPoint.ema200 > dataPoint.ema100 && dataPoint.ema100 > dataPoint.ema50 && dataPoint.ema50 > dataPoint.sma13;

        // Determine if an uptrend is occurring.
        uptrending = dataPoint.ema200 < dataPoint.ema100 && dataPoint.ema100 < dataPoint.ema50 && dataPoint.ema50 < dataPoint.sma13;

        // Determine if RSI is above the overbought line.
        rsiOverbought = dataPoint.rsi7 && dataPoint.rsi7 >= 77;

        // Determine if RSI is below the oversold line.
        rsiOversold = dataPoint.rsi7 && dataPoint.rsi7 <= 23;

        // Determine if the upper regression bound was breached by the high.
        regressionUpperBoundBreached = dataPoint.high >= dataPoint.pRegUpper200;

        // Determine if the lower regression bound was breached by the low.
        regressionLowerBoundBreached = dataPoint.low <= dataPoint.pRegLower200;

        // Determine if there is a significant gap (> 60 seconds) between the current timestamp and the previous timestamp.
        timeGapPresent = previousDataPoint && (dataPoint.timestamp - previousDataPoint.timestamp) > 60 * 1000;

        // Determine whether to buy (CALL).
        if (uptrending && rsiOversold && regressionLowerBoundBreached && !timeGapPresent) {
            callNextTick = true;
        }

        // Determine whether to buy (PUT).
        if (downtrending && rsiOverbought && regressionUpperBoundBreached && !timeGapPresent) {
            putNextTick = true;
        }

        // Track the current data point as the previous data point for the next tick.
        previousDataPoint = dataPoint;
    });

    // Show the results.
    console.log('SYMBOL:\t\t' + previousDataPoint.symbol);
    console.log('PROFIT/LOSS:\t$' + self.getProfitLoss());
    console.log('WIN RATE:\t' + self.getWinRate());
    console.log('WINS:\t\t' + self.winCount);
    console.log('LOSSES:\t\t' + self.loseCount);

    // Save the output to a file.
    this.saveOutput();
};

module.exports = Reversals;
