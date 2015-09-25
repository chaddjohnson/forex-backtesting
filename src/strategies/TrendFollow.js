var studies = require('../studies');
var Base = require('./Base');
var Call = require('../positions/Call');
var Put = require('../positions/Put');

// Define studies to use.
var studyDefinitions = [
    {
        study: studies.Sma,
        inputs: {
            length: 50
        },
        outputMap: {
            sma: 'sma50'
        }
    },{
        study: studies.Sma,
        inputs: {
            length: 25
        },
        outputMap: {
            sma: 'sma25'
        }
    },{
        study: studies.Sma,
        inputs: {
            length: 14
        },
        outputMap: {
            sma: 'sma14'
        }
    },{
        study: studies.Sma,
        inputs: {
            length: 3
        },
        outputMap: {
            sma: 'sma3'
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
        study: studies.Ema,
        inputs: {
            length: 24
        },
        outputMap: {
            ema: 'ema24'
        }
    },{
        study: studies.PolynomialRegressionChannel,
        inputs: {
            length: 200,
            degree: 4,
            deviations: 1.618
        },
        outputMap: {
            regression: 'prChannel200',
            upper: 'prChannelUpper200',
            lower: 'prChannelLower200'
        }
    }
];

function TrendFollow() {
    this.constructor = TrendFollow;
    Base.call(this);

    this.prepareStudies(studyDefinitions);
}

// Create a copy of the Base "class" prototype for use in this "class."
TrendFollow.prototype = Object.create(Base.prototype);

TrendFollow.prototype.backtest = function(data, investment, profitability) {
    var self = this;
    var callNextTick = false;
    var putNextTick = false;
    var movingAveragesUptrending = false;
    var movingAveragesDowntrending = false;
    var regressionUpperBoundBreached = false;
    var regressionLowerBoundBreached = false;
    var timeGapPresent = false;
    var previousDataPoint;
    var previousBalance = 0;

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

        movingAveragesUptrending = dataPoint.sma14 > dataPoint.ema24 &&
                                   dataPoint.sma25 > dataPoint.ema50 &&
                                   dataPoint.sma50 > dataPoint.ema100 &&
                                   dataPoint.sma3  > dataPoint.ema50;

        movingAveragesDowntrending = dataPoint.sma14 < dataPoint.ema24 &&
                                     dataPoint.sma25 < dataPoint.ema50 &&
                                     dataPoint.sma50 < dataPoint.ema100 &&
                                     dataPoint.sma3  < dataPoint.ema50;

        // Determine if the upper regression bound was breached by the high.
        regressionUpperBoundBreached = dataPoint.high >= dataPoint.prChannelUpper200;

        // Determine if the lower regression bound was breached by the low.
        regressionLowerBoundBreached = dataPoint.low <= dataPoint.prChannelLower200;

        // Determine if there is a significant gap (> 60 seconds) between the current timestamp and the previous timestamp.
        timeGapPresent = previousDataPoint && (dataPoint.timestamp - previousDataPoint.timestamp) > 60 * 1000;

        // Determine whether to buy (CALL).
        if (movingAveragesUptrending && regressionLowerBoundBreached && dataPoint.close < dataPoint.ema50 && timeGapPresent) {
            callNextTick = true;
        }

        // Determine whether to buy (PUT).
        if (movingAveragesDowntrending && regressionUpperBoundBreached && dataPoint.close > dataPoint.ema50 && !timeGapPresent) {
            putNextTick = true;
        }

        if (self.getProfitLoss() !== previousBalance) {
            console.log('BALANCE: $' + self.getProfitLoss());
            console.log();
        }
        previousBalance = self.getProfitLoss();

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

module.exports = TrendFollow;
