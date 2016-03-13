var strategyFn = require('./src/strategies/combined/Reversals');
var combinations = [
    { "ema200" : false, "ema100" : true, "ema50" : true, "sma13" : false, "rsi" : { "rsi" : "rsi9", "overbought" : 70, "oversold" : 30 }, "stochastic" : { "K" : "stochastic10K", "D" : "stochastic10D", "overbought" : 80, "oversold" : 20 }, "prChannel" : { "upper" : "prChannelUpper200_4_21", "lower" : "prChannelLower200_4_21" } },
    { "ema200" : false, "ema100" : true, "ema50" : true, "sma13" : false, "rsi" : { "rsi" : "rsi5", "overbought" : 80, "oversold" : 20 }, "stochastic" : { "K" : "stochastic10K", "D" : "stochastic10D", "overbought" : 77, "oversold" : 23 }, "prChannel" : { "upper" : "prChannelUpper200_4_215", "lower" : "prChannelLower200_4_215" } },
    { "ema200" : false, "ema100" : true, "ema50" : true, "sma13" : false, "rsi" : { "rsi" : "rsi9", "overbought" : 70, "oversold" : 30 }, "stochastic" : { "K" : "stochastic10K", "D" : "stochastic10D", "overbought" : 80, "oversold" : 20 }, "prChannel" : { "upper" : "prChannelUpper200_4_205", "lower" : "prChannelLower200_4_205" } },
    { "ema200" : false, "ema100" : true, "ema50" : true, "sma13" : false, "rsi" : { "rsi" : "rsi9", "overbought" : 70, "oversold" : 30 }, "stochastic" : { "K" : "stochastic14K", "D" : "stochastic14D", "overbought" : 77, "oversold" : 23 }, "prChannel" : { "upper" : "prChannelUpper200_4_21", "lower" : "prChannelLower200_4_21" } },
    { "ema200" : false, "ema100" : true, "ema50" : true, "sma13" : false, "rsi" : { "rsi" : "rsi7", "overbought" : 77, "oversold" : 23 }, "stochastic" : { "K" : "stochastic10K", "D" : "stochastic10D", "overbought" : 80, "oversold" : 20 }, "prChannel" : { "upper" : "prChannelUpper200_4_20", "lower" : "prChannelLower200_4_20" } },
    { "ema200" : false, "ema100" : true, "ema50" : true, "sma13" : false, "rsi" : null, "stochastic" : { "K" : "stochastic14K", "D" : "stochastic14D", "overbought" : 77, "oversold" : 23 }, "prChannel" : { "upper" : "prChannelUpper250_5_205", "lower" : "prChannelLower250_5_205" } },
    { "ema200" : false, "ema100" : false, "ema50" : true, "sma13" : true, "rsi" : { "rsi" : "rsi9", "overbought" : 77, "oversold" : 23 }, "stochastic" : null, "prChannel" : { "upper" : "prChannelUpper200_3_185", "lower" : "prChannelLower200_3_185" } }
];
var dataParsers = require('./src/dataParsers');
var strategy = new strategyFn(process.argv[2], combinations);

// strategy.setShowTrades(true);

try {
    // Parse the raw data file.
    dataParsers.ctoption.parse('./data/ctoption/' + process.argv[2] + '.csv').then(function(parsedData) {
        strategy.setProfitLoss(5000);
        strategy.backtest(parsedData, 25, 0.76);
    });
}
catch (error) {
    console.error(error.message || error);
    process.exit(1);
}
