var strategyFn = require('./src/strategies/combined/Reversals');
var combinations = [ { "ema200" : false, "ema100" : true, "ema50" : true, "sma13" : false, "rsi" : { "rsi" : "rsi7", "overbought" : 80, "oversold" : 20 }, "stochastic" : null, "prChannel" : { "upper" : "prChannelUpper200_2_195", "lower" : "prChannelLower200_2_195" } }, { "ema200" : false, "ema100" : true, "ema50" : true, "sma13" : false, "rsi" : { "rsi" : "rsi7", "overbought" : 80, "oversold" : 20 }, "stochastic" : null, "prChannel" : { "upper" : "prChannelUpper200_2_205", "lower" : "prChannelLower200_2_205" } }, { "ema200" : false, "ema100" : true, "ema50" : true, "sma13" : false, "rsi" : { "rsi" : "rsi9", "overbought" : 77, "oversold" : 23 }, "stochastic" : null, "prChannel" : { "upper" : "prChannelUpper200_2_205", "lower" : "prChannelLower200_2_205" } }, { "ema200" : false, "ema100" : true, "ema50" : true, "sma13" : false, "rsi" : { "rsi" : "rsi9", "overbought" : 77, "oversold" : 23 }, "stochastic" : null, "prChannel" : { "upper" : "prChannelUpper200_2_185", "lower" : "prChannelLower200_2_185" } }, { "ema200" : false, "ema100" : true, "ema50" : true, "sma13" : false, "rsi" : { "rsi" : "rsi9", "overbought" : 77, "oversold" : 23 }, "stochastic" : null, "prChannel" : { "upper" : "prChannelUpper200_2_185", "lower" : "prChannelLower200_2_185" } }, { "ema200" : true, "ema100" : true, "ema50" : true, "sma13" : false, "rsi" : { "rsi" : "rsi7", "overbought" : 80, "oversold" : 20 }, "stochastic" : null, "prChannel" : { "upper" : "prChannelUpper200_2_205", "lower" : "prChannelLower200_2_205" } } ];
var dataParsers = require('./src/dataParsers');
var strategy = new strategyFn(process.argv[2], combinations);

// strategy.setShowTrades(true);

try {
    // Parse the raw data file.
    dataParsers.ctoption.parse('./data/ctoption/' + process.argv[2] + '.csv').then(function(parsedData) {
        strategy.backtest(parsedData, 120, 0.76);
    });
}
catch (error) {
    console.error(error.message || error);
    process.exit(1);
}
