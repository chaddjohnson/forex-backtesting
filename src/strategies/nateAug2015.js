var studies = require('../studies');
var Base = require('./base');
var Call = require('../positions/call');
var Put = require('../positions/put');

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

NateAug2015.prototype.backtest = function(investment, profitability) {
    var self = this;
    var data = self.getData();
    var earnings = 0.0;
    var callNextTick = false;
    var putNextTick = false;
    var downtrending = false;
    var uptrending = false;
    var rsiOverbought = false;
    var rsiOversold = false;
    var volumeIsHighEnough = false;
    var volumeChangedSignificantly = false;
    var timeGapPresent = false;
    var previousDataPoint;

    // For every data point...
    data.forEach(function(dataPoint) {
        // Simulate the next tick, and process update studies for the tick.
        self.tick();

        if (callNextTick) {
            // Create a new position.
            self.addPosition(new Call(dataPoint.symbol, dataPoint.timestamp, dataPoint.price, investment, profitability, 5));
            callNextTick = true;
        }

        if (putNextTick) {
            // Create a new position.
            self.addPosition(new Put(dataPoint.symbol, dataPoint.timestamp, dataPoint.price, investment, profitability, 5));
            putNextTick = false;
        }

        // Determine if a downtrend is occurring. In this case, the following will hold true: ema200 > ema100 > ema50 > sma13
        downtrending = data.ema200 > data.ema100 && data.ema100 > data.ema50 && data.ema50 > data.sma13;

        // Determine if an uptrend is occurring. In this case, the following will hold true: ema200 < ema100 < ema50 < sma13
        uptrending = data.ema200 < data.ema100 && data.ema100 < data.ema50 && data.ema50 < data.sma13;

        // Determine if RSI just crossed below the overbought line.
        rsiOverbought = dataPoint.price >= 77;

        // Determine if RSI just crossed above the oversold line.
        rsiOversold = dataPoint.price <= 23;

        // Determine if the volume is high enough.
        volumeIsHighEnough = dataPoint.volume > 50;

        // Determine if the volume changed significantly since the last minute.
        volumeChangedSignificantly = previousDataPoint && dataPoint.volume / previousDataPoint.volume >= 1.2;

        // Determine if there is a significant gap (> 5 seconds) between the current timestamp and the previous timestamp.
        timeGapPresent = previousDataPoint && (dataPoint.timestamp - previousDataPoint.timestamp) > 5000;

        // Determine whether to buy (CALL).
        if (uptrending && rsiOversold && volumeIsHighEnough && volumeChangedSignificantly && !timeGapPresent) {
            callNextTick = true;
        }

        // Determine whether to buy (PUT).
        if (downtrending && rsiOverbought && volumeIsHighEnough && volumeChangedSignificantly && !timeGapPresent) {
            putNextTick = true;
        }

        // Track the current data point as the previous data point for the next tick.
        previousDataPoint = dataPoint;
    });

    // Show the results.
    console.log('EARNINGS: $' + earnings);
};

module.exports = NateAug2015;
