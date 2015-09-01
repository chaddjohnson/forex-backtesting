var studies = require('../studies');
var Base = require('./base');
var Call = require('../positions/call');
var Put = require('../positions/put');

// Define studies to use.
var studyDefinitions = [
    {
        study: studies.Ema,
        inputs: {
            length: 350
        },
        outputMap: {
            ema: 'ema350'
        }
    },{
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
            ema: 'sma13'
        }
    },{
        study: studies.Sma,
        inputs: {
            length: 6
        },
        outputMap: {
            ema: 'sma6'
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
    var callCondition = false;
    var putCondition = false;
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

        callCondition = dataPoint.ema350 < dataPoint.ema200 &&
                        dataPoint.ema200 < dataPoint.ema100 &&
                        dataPoint.sma6 > dataPoint.ema350 &&
                        dataPoint.sma13 > dataPoint.ema350 &&
                        dataPoint.ema50 > dataPoint.ema100 &&
                        dataPoint.sma3 > dataPoint.ema50 &&
                        dataPoint.sma13 > dataPoint.ema50;

        putCondition = dataPoint.ema350 > dataPoint.ema200 &&
                       dataPoint.ema200 > dataPoint.ema100 &&
                       dataPoint.sma6 < dataPoint.ema350 &&
                       dataPoint.sma13 < dataPoint.ema350 &&
                       dataPoint.ema50 < dataPoint.ema100 &&
                       dataPoint.sma3 < dataPoint.ema50 &&
                       dataPoint.sma13 < dataPoint.ema50;

        crossedBelowEma50 = previousDataPoint && previousDataPoint.close >= previousDataPoint.ema50 && dataPoint.close < dataPoint.ema50;

        crossedAboveEma50 = previousDataPoint && previousDataPoint.close <= previousDataPoint.ema50 && dataPoint.close > dataPoint.ema50;

        // Determine if there is a significant gap (> 60 seconds) between the current timestamp and the previous timestamp.
        timeGapPresent = previousDataPoint && (dataPoint.timestamp - previousDataPoint.timestamp) > 60 * 1000;

        // Determine whether to buy (CALL).
        if (callCondition && crossedBelowEma50 && !timeGapPresent) {
            callNextTick = true;
        }

        // Determine whether to buy (PUT).
        if (putCondition && crossedAboveEma50 && !timeGapPresent) {
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

module.exports = TrendFollow;
