var studies = require('../studies');
var Base = require('./base');

// Define studies to use.
var studyDefinitions = [
    {
        name: 'ema200',
        study: studies.Ema,
        inputs: {
            length: 200
        }
    },{
        name: 'ema100',
        study: studies.Ema,
        inputs: {
            length: 100
        }
    },{
        name: 'ema50',
        study: studies.Ema,
        inputs: {
            length: 50
        }
    },{
        name: 'sma13',
        study: studies.Sma,
        inputs: {
            length: 13
        }
    },{
        name: 'rsi7',
        study: studies.Rsi,
        inputs: {
            length: 7,
            overbought: 77,
            oversold: 23
        }
    }
];

function NateAug2015(data) {
    this.constructor = NateAug2015;
    Base.call(this, data);

    this.prepareStudies(studyDefinitions);
}

// Create a copy of the Base "class" prototype for use in this "class."
NateAug2015.prototype = Object.create(Base.prototype);

NateAug2015.prototype.backtest = function(data) {
    var self = this;
    var earnings = 0.0;
    var previousDataPoint;

    // For every data point...
    data.forEach(function(dataPoint) {
        var downtrending = false;
        var uptrending = false;
        var rsiCrossedBelowOverbought = false;
        var rsiCrossedAboveOversold = false;

        // Simulate the next tick, and process update studies for the tick.
        self.tick();

        // Determine if a downtrend is occurring. In this case, the following will hold true: ema200 > ema100 > ema50 > sma13
        if (data.ema200 > data.ema100 && data.ema100 > data.ema50 && data.ema50 > data.sma13) {
            downtrending = true;
        }

        // Determine if an uptrend is occurring. In this case, the following will hold true: ema200 < ema100 < ema50 < sma13
        if (data.ema200 < data.ema100 && data.ema100 < data.ema50 && data.ema50 < data.sma13) {
            uptrending = true;
        }

        // Determine if RSI just crossed below the overbought line.
        if (previousDataPoint && previousDataPoint.ask >= 77 && dataPoint.ask < 77) {
            rsiCrossedBelowOverbought = true;
        }

        // Determine if RSI just crossed above the oversold line.
        if (previousDataPoint && previousDataPoint.ask <= 23 && dataPoint.ask > 23) {
            rsiCrossedBelowOverbought = true;
        }

        // Determine whether to buy.
        if ((downtrending && rsiCrossedBelowOverbought) || (uptrending && rsiCrossedAboveOversold)) {
            // Create a new position.
            self.addPosition(new Position(dataPoint.symbol, dataPoint.timestamp, dataPoint.ask));
        }

        // Track the current data point as the previous data point for the next tick.
        previousDataPoint = dataPoint;
    });

    // Show the results.
    console.log('EARNINGS: $' + earnings);
};

module.exports = NateAug2015;
