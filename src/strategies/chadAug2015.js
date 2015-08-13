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

function ChadAug2015() {
    this.constructor = ChadAug2015;
    Base.call(this);

    this.prepareStudies(studyDefinitions);
}

// Create a copy of the Base "class" prototype for use in this "class."
ChadAug2015.prototype = Object.create(Base.prototype);

ChadAug2015.prototype.backtest = function(data, investment, profitability) {
    var self = this;
    var callNextTick = false;
    var putNextTick = false;
    var downtrending = false;
    var uptrending = false;
    var rsiOverbought = false;
    var rsiOversold = false;
    var volumeHighEnough = false;
    var volumeChangedSignificantly = false;
    var timeGapPresent = false;
    var sma13Ema50DistanceEnough = false;
    var previousDataPoint;

    // For every data point...
    data.forEach(function(dataPoint) {
        // Simulate the next tick, and process update studies for the tick.
        self.tick(dataPoint);

        if (callNextTick) {
            // Create a new position.
            self.addPosition(new Call(dataPoint.symbol, dataPoint.timestamp, previousDataPoint.price, investment, profitability, 5, dataPoint, previousDataPoint));
            callNextTick = false;
        }

        if (putNextTick) {
            // Create a new position.
            self.addPosition(new Put(dataPoint.symbol, dataPoint.timestamp, previousDataPoint.price, investment, profitability, 5, dataPoint, previousDataPoint));
            putNextTick = false;
        }

        // Determine if a downtrend is occurring. In this case, the following will hold true: ema200 > ema100 > ema50 > sma13
        downtrending = dataPoint.ema200 > dataPoint.ema100 && dataPoint.ema100 > dataPoint.ema50 && dataPoint.ema50 > dataPoint.sma13;

        // Determine if an uptrend is occurring. In this case, the following will hold true: ema200 < ema100 < ema50 < sma13
        uptrending = dataPoint.ema200 < dataPoint.ema100 && dataPoint.ema100 < dataPoint.ema50 && dataPoint.ema50 < dataPoint.sma13;

        // Determine if RSI is above the overbought line.
        rsiOverbought = dataPoint.rsi7 && dataPoint.rsi7 >= 77;

        // Determine if RSI is below the oversold line.
        rsiOversold = dataPoint.rsi7 && dataPoint.rsi7 <= 23;

        // Determine if the volume is high enough.
        volumeHighEnough = dataPoint.volume > 50;

        // Determine if the volume changed significantly since the last minute.
        volumeChangedSignificantly = previousDataPoint && dataPoint.volume / previousDataPoint.volume >= 1.2;

        // Determine if there is a significant gap (> 60 seconds) between the current timestamp and the previous timestamp.
        timeGapPresent = previousDataPoint && (dataPoint.timestamp - previousDataPoint.timestamp) > 60 * 1000;

        // sma13Ema50DistanceEnough = Math.abs(dataPoint.sma13 - dataPoint.ema50) >= 0.00009;  // USDCAD
        // sma13Ema50DistanceEnough = Math.abs(dataPoint.sma13 - dataPoint.ema50) >= 0.00045;  // EURUSD
        // sma13Ema50DistanceEnough = Math.abs(dataPoint.sma13 - dataPoint.ema50) >= 0.02;  // AUDJPY
        // sma13Ema50DistanceEnough = Math.abs(dataPoint.sma13 - dataPoint.ema50) / dataPoint.price < 0.0009;
        // sma13Ema50DistanceEnough = previousDataPoint && Math.abs(previousDataPoint.sma13 - previousDataPoint.ema50) >= 0.0005;
        // sma13Ema50DistanceEnough = Math.abs(dataPoint.sma13 - dataPoint.ema50) >= 0.0002;  // EURGBP and others
        sma13Ema50DistanceEnough = true;

        // Determine whether to buy (CALL).
        if (uptrending && rsiOversold && volumeHighEnough && volumeChangedSignificantly && !timeGapPresent && sma13Ema50DistanceEnough && self.getProfitLoss() < investment * 6) {
            callNextTick = true;
        }

        // Determine whether to buy (PUT).
        if (downtrending && rsiOverbought && volumeHighEnough && volumeChangedSignificantly && !timeGapPresent && sma13Ema50DistanceEnough && self.getProfitLoss() < investment * 6) {
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

module.exports = ChadAug2015;
