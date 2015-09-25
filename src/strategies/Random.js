var studies = require('../studies');
var Base = require('./Base');
var Call = require('../positions/Call');
var Put = require('../positions/Put');

// Define studies to use.
var studyDefinitions = [];

function Random() {
    this.constructor = Random;
    Base.call(this);

    this.prepareStudies(studyDefinitions);
}

// Create a copy of the Base "class" prototype for use in this "class."
Random.prototype = Object.create(Base.prototype);

Random.prototype.backtest = function(data, investment, profitability) {
    var self = this;
    var waitTicks = 0;
    var decisionPivot = 0;
    var previousDataPoint;

    // For every data point...
    data.forEach(function(dataPoint, index) {
        // Simulate the next tick, and process update studies for the tick.
        self.tick(dataPoint);

        if (waitTicks <= 0 && previousDataPoint) {
            decisionPivot = Math.floor(Math.random() * ((100 - 1) + 1)) + 1;
            waitTicks = Math.floor(Math.random() * ((250 - 1) + 1)) + 1;

            if (decisionPivot <= 50) {
                self.addPosition(new Call(dataPoint.symbol, dataPoint.timestamp, previousDataPoint.close, investment, profitability, 120));
            }
            else {
                self.addPosition(new Put(dataPoint.symbol, dataPoint.timestamp, previousDataPoint.close, investment, profitability, 120));
            }
        }

        waitTicks--;

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

module.exports = Random;
