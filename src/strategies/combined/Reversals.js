var Base = require('./Base');
var Call = require('../../positions/Call');
var Put = require('../../positions/Put');
var studies = require('../../studies');

var studyDefinitions = [
    {study: studies.Ema, inputs: {length: 200}, outputMap: {ema: 'ema200'}},
    {study: studies.Ema, inputs: {length: 100}, outputMap: {ema: 'ema100'}},
    {study: studies.Ema, inputs: {length: 50}, outputMap: {ema: 'ema50'}},
    {study: studies.Sma, inputs: {length: 13}, outputMap: {sma: 'sma13'}},
    {study: studies.Rsi, inputs: {length: 7}, outputMap: {rsi: 'rsi7'}},
    {study: studies.Rsi, inputs: {length: 5}, outputMap: {rsi: 'rsi5'}},
    {study: studies.Rsi, inputs: {length: 2}, outputMap: {rsi: 'rsi2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 100, degree: 4, deviations: 1.9}, outputMap: {regression: 'prChannel100_4_19', upper: 'prChannelUpper100_4_19', lower: 'prChannelLower100_4_19', upper2: 'prChannelUpper100_4_19-2', lower2: 'prChannelLower100_4_19-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 200, degree: 2, deviations: 1.9}, outputMap: {regression: 'prChannel200_2_19', upper: 'prChannelUpper200_2_19', lower: 'prChannelLower200_2_19', upper2: 'prChannelUpper200_2_19-2', lower2: 'prChannelLower200_2_19-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 250, degree: 3, deviations: 1.9}, outputMap: {regression: 'prChannel250_3_19', upper: 'prChannelUpper250_3_19', lower: 'prChannelLower250_3_19', upper2: 'prChannelUpper250_3_19-2', lower2: 'prChannelLower250_3_19-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 250, degree: 3, deviations: 1.95}, outputMap: {regression: 'prChannel250_3_195', upper: 'prChannelUpper250_3_195', lower: 'prChannelLower250_3_195', upper2: 'prChannelUpper250_3_195-2', lower2: 'prChannelLower250_3_195-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 300, degree: 4, deviations: 1.9}, outputMap: {regression: 'prChannel300_4_19', upper: 'prChannelUpper300_4_19', lower: 'prChannelLower300_4_19', upper2: 'prChannelUpper300_4_19-2', lower2: 'prChannelLower300_4_19-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 500, degree: 2}, outputMap: {regression: 'trendPrChannel500_2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 550, degree: 2}, outputMap: {regression: 'trendPrChannel550_2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 600, degree: 2}, outputMap: {regression: 'trendPrChannel600_2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 650, degree: 2}, outputMap: {regression: 'trendPrChannel650_2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 700, degree: 2}, outputMap: {regression: 'trendPrChannel700_2'}}
];

function ReversalsCombined(symbol, configurations) {
    this.constructor = ReversalsCombined;
    Base.call(this, symbol, configurations);

    this.configurations = configurations;

    this.prepareStudies(studyDefinitions);
}

ReversalsCombined.prototype = Object.create(Base.prototype);

ReversalsCombined.prototype.backtest = function(data, investment, profitability) {
    var self = this;
    var expirationMinutes = 5;
    var putNextTick = false;
    var callNextTick = false;
    var putThisConfiguration = false;
    var callThisConfiguration = false;
    var previousDataPoint;
    var dataPointCount = data.length;
    var previousBalance = 0;
    var position = null;

    // For every data point...
    data.forEach(function(dataPoint, index) {
        var position;
        var timestampHour = new Date(dataPoint.timestamp).getHours();

        // Simulate the next tick.
        self.tick(dataPoint);

        // Only trade when the profitability is highest (11pm - 4pm CST).
        // Note that MetaTrader automatically converts timestamps to the current timezone in exported CSV files.
        if (timestampHour >= 16 && timestampHour < 23) {
            // Track the current data point as the previous data point for the next tick.
            previousDataPoint = dataPoint;
            return;
        }

        if (previousDataPoint && index < dataPointCount - 1) {
            if (putNextTick) {
                // Create a new position.
                position = new Put(self.getSymbol(), (dataPoint.timestamp - 1000), previousDataPoint.close, investment, profitability, expirationMinutes);
                position.setShowTrades(self.getShowTrades());
                self.addPosition(position);
            }

            if (callNextTick) {
                // Create a new position.
                position = new Call(self.getSymbol(), (dataPoint.timestamp - 1000), previousDataPoint.close, investment, profitability, expirationMinutes)
                position.setShowTrades(self.getShowTrades());
                self.addPosition(position);
            }
        }

        putNextTick = false;
        callNextTick = false;

        // For every configuration...
        self.configurations.forEach(function(configuration) {
            putThisConfiguration = true;
            callThisConfiguration = true;

            if (configuration.ema200 && configuration.ema100) {
                if (!dataPoint.ema200 || !dataPoint.ema100) {
                    putThisConfiguration = false;
                    callThisConfiguration = false;
                }

                // Determine if a downtrend is not occurring.
                if (putThisConfiguration && dataPoint.ema200 < dataPoint.ema100) {
                    putThisConfiguration = false;
                }

                // Determine if an uptrend is not occurring.
                if (callThisConfiguration && dataPoint.ema200 > dataPoint.ema100) {
                    callThisConfiguration = false;
                }
            }
            if (configuration.ema100 && configuration.ema50) {
                if (!dataPoint.ema100 || !dataPoint.ema50) {
                    putThisConfiguration = false;
                    callThisConfiguration = false;
                }

                // Determine if a downtrend is not occurring.
                if (putThisConfiguration && dataPoint.ema100 < dataPoint.ema50) {
                    putThisConfiguration = false;
                }

                // Determine if an uptrend is not occurring.
                if (callThisConfiguration && dataPoint.ema100 > dataPoint.ema50) {
                    callThisConfiguration = false;
                }
            }
            if (configuration.ema50 && configuration.sma13) {
                if (!dataPoint.ema50 || !dataPoint.sma13) {
                    putThisConfiguration = false;
                    callThisConfiguration = false;
                }

                // Determine if a downtrend is not occurring.
                if (putThisConfiguration && dataPoint.ema50 < dataPoint.sma13) {
                    putThisConfiguration = false;
                }

                // Determine if an uptrend is not occurring.
                if (callThisConfiguration && dataPoint.ema50 > dataPoint.sma13) {
                    callThisConfiguration = false;
                }
            }
            if (configuration.rsi) {
                if (typeof dataPoint[configuration.rsi.rsi] === 'number') {
                    // Determine if RSI is not above the overbought line.
                    if (putThisConfiguration && dataPoint[configuration.rsi.rsi] <= configuration.rsi.overbought) {
                        putThisConfiguration = false;
                    }

                    // Determine if RSI is not below the oversold line.
                    if (callThisConfiguration && dataPoint[configuration.rsi.rsi] >= configuration.rsi.oversold) {
                        callThisConfiguration = false;
                    }
                }
                else {
                    putThisConfiguration = false;
                    callThisConfiguration = false;
                }
            }
            if (configuration.prChannel) {
                if (dataPoint[configuration.prChannel.upper] && dataPoint[configuration.prChannel.lower]) {
                    // Determine if the upper regression bound was not breached by the high price.
                    if (putThisConfiguration && (!dataPoint[configuration.prChannel.upper] || dataPoint.high <= dataPoint[configuration.prChannel.upper] || dataPoint.close > dataPoint[configuration.prChannel.upper2])) {
                        putThisConfiguration = false;
                    }

                    // Determine if the lower regression bound was not breached by the low price.
                    if (callThisConfiguration && (!dataPoint[configuration.prChannel.lower] || dataPoint.low >= dataPoint[configuration.prChannel.lower] || dataPoint.close < dataPoint[configuration.prChannel.lower2])) {
                        callThisConfiguration = false;
                    }
                }
                else {
                    putThisConfiguration = false;
                    callThisConfiguration = false;
                }
            }
            if (configuration.trendPrChannel) {
                if (previousDataPoint && dataPoint[configuration.trendPrChannel.regression] && previousDataPoint[configuration.trendPrChannel.regression]) {
                    // Determine if a long-term downtrend is not occurring.
                    if (putThisConfiguration && dataPoint[configuration.trendPrChannel.regression] > previousDataPoint[configuration.trendPrChannel.regression]) {
                        putThisConfiguration = false;
                    }

                    // Determine if a long-term uptrend is not occurring.
                    if (callThisConfiguration && dataPoint[configuration.trendPrChannel.regression] < previousDataPoint[configuration.trendPrChannel.regression]) {
                        callThisConfiguration = false;
                    }
                }
                else {
                    putThisConfiguration = false;
                    callThisConfiguration = false;
                }
            }

            // Determine whether to trade next tick.
            putNextTick = putNextTick || putThisConfiguration;
            callNextTick = callNextTick || callThisConfiguration;
        });

        // Track the current data point as the previous data point for the next tick.
        previousDataPoint = dataPoint;

        if (putNextTick) {
            console.log('PUT at ' + new Date(dataPoint.timestamp + 1000));
        }

        if (callNextTick) {
            console.log('CALL at ' + new Date(dataPoint.timestamp + 1000));
        }
    });

    console.log(self.getResults());
};

module.exports = ReversalsCombined;
