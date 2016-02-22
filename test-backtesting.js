var strategyFn = require('./src/strategies/combined/Reversals');
var combinations = [ { "ema200" : true, "ema100" : true, "ema50" : true, "sma13" : false, "rsi" : { "rsi" : "rsi7", "overbought" : 77, "oversold" : 23 }, "stochastic" : { "K" : "stochastic10K", "D" : "stochastic10D", "overbought" : 80, "oversold" : 20 }, "prChannel" : { "upper" : "prChannelUpper350_4_195", "lower" : "prChannelLower350_4_195" } }, { "ema200" : true, "ema100" : true, "ema50" : true, "sma13" : false, "rsi" : { "rsi" : "rsi7", "overbought" : 77, "oversold" : 23 }, "stochastic" : { "K" : "stochastic10K", "D" : "stochastic10D", "overbought" : 77, "oversold" : 23 }, "prChannel" : { "upper" : "prChannelUpper350_4_195", "lower" : "prChannelLower350_4_195" } }, { "ema200" : true, "ema100" : true, "ema50" : true, "sma13" : false, "rsi" : { "rsi" : "rsi7", "overbought" : 77, "oversold" : 23 }, "stochastic" : { "K" : "stochastic21K", "D" : "stochastic21D", "overbought" : 77, "oversold" : 23 }, "prChannel" : { "upper" : "prChannelUpper350_4_195", "lower" : "prChannelLower350_4_195" } }, { "ema200" : true, "ema100" : true, "ema50" : true, "sma13" : false, "rsi" : { "rsi" : "rsi7", "overbought" : 77, "oversold" : 23 }, "stochastic" : { "K" : "stochastic14K", "D" : "stochastic14D", "overbought" : 77, "oversold" : 23 }, "prChannel" : { "upper" : "prChannelUpper350_4_195", "lower" : "prChannelLower350_4_195" } }, { "ema200" : true, "ema100" : true, "ema50" : true, "sma13" : false, "rsi" : { "rsi" : "rsi7", "overbought" : 77, "oversold" : 23 }, "stochastic" : { "K" : "stochastic14K", "D" : "stochastic14D", "overbought" : 77, "oversold" : 23 }, "prChannel" : { "upper" : "prChannelUpper350_4_19", "lower" : "prChannelLower350_4_19" } }, { "ema200" : true, "ema100" : true, "ema50" : true, "sma13" : false, "rsi" : { "rsi" : "rsi7", "overbought" : 77, "oversold" : 23 }, "stochastic" : { "K" : "stochastic21K", "D" : "stochastic21D", "overbought" : 77, "oversold" : 23 }, "prChannel" : { "upper" : "prChannelUpper350_4_185", "lower" : "prChannelLower350_4_185" } } ];
var dataParsers = require('./src/dataParsers');
var strategy = new strategyFn(process.argv[2], combinations);

// strategy.setShowTrades(true);

try {
    // Parse the raw data file.
    dataParsers.metatrader.parse('./data/metatrader/ten-year/' + process.argv[2] + '.csv').then(function(parsedData) {
        strategy.setProfitLoss(7000);
        strategy.backtest(parsedData, 120, 0.76);
    });
}
catch (error) {
    console.error(error.message || error);
    process.exit(1);
}
