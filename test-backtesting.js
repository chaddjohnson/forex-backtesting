var strategyFn = require('./src/strategies/combined/Reversals');
var combinations = [ { "ema200" : false, "ema100" : true, "ema50" : true, "sma13" : false, "rsi" : { "rsi" : "rsi5", "overbought" : 80, "oversold" : 20 }, "prChannel" : { "upper" : "prChannelUpper250_3_215", "lower" : "prChannelLower250_3_215" } }, { "ema200" : true, "ema100" : true, "ema50" : true, "sma13" : false, "rsi" : { "rsi" : "rsi7", "overbought" : 77, "oversold" : 23 }, "prChannel" : { "upper" : "prChannelUpper400_4_195", "lower" : "prChannelLower400_4_195" } }, { "ema200" : false, "ema100" : true, "ema50" : true, "sma13" : false, "rsi" : { "rsi" : "rsi9", "overbought" : 77, "oversold" : 23 }, "prChannel" : { "upper" : "prChannelUpper200_3_19", "lower" : "prChannelLower200_3_19" } }, { "ema200" : false, "ema100" : true, "ema50" : true, "sma13" : false, "rsi" : { "rsi" : "rsi7", "overbought" : 77, "oversold" : 23 }, "prChannel" : { "upper" : "prChannelUpper400_4_20", "lower" : "prChannelLower400_4_20" } }, { "ema200" : true, "ema100" : true, "ema50" : true, "sma13" : false, "rsi" : { "rsi" : "rsi5", "overbought" : 80, "oversold" : 20 }, "prChannel" : { "upper" : "prChannelUpper400_4_195", "lower" : "prChannelLower400_4_195" } }, { "ema200" : false, "ema100" : true, "ema50" : true, "sma13" : false, "rsi" : { "rsi" : "rsi7", "overbought" : 77, "oversold" : 23 }, "prChannel" : { "upper" : "prChannelUpper400_4_195", "lower" : "prChannelLower400_4_195" } }, { "ema200" : true, "ema100" : true, "ema50" : true, "sma13" : false, "rsi" : { "rsi" : "rsi7", "overbought" : 77, "oversold" : 23 }, "prChannel" : { "upper" : "prChannelUpper400_4_19", "lower" : "prChannelLower400_4_19" } }, { "ema200" : true, "ema100" : true, "ema50" : true, "sma13" : false, "rsi" : { "rsi" : "rsi7", "overbought" : 77, "oversold" : 23 }, "prChannel" : { "upper" : "prChannelUpper250_3_205", "lower" : "prChannelLower250_3_205" } }, { "ema200" : true, "ema100" : true, "ema50" : true, "sma13" : false, "rsi" : { "rsi" : "rsi7", "overbought" : 80, "oversold" : 20 }, "prChannel" : { "upper" : "prChannelUpper200_2_215", "lower" : "prChannelLower200_2_215" } }, { "ema200" : false, "ema100" : true, "ema50" : true, "sma13" : false, "rsi" : { "rsi" : "rsi7", "overbought" : 77, "oversold" : 23 }, "prChannel" : { "upper" : "prChannelUpper400_4_19", "lower" : "prChannelLower400_4_19" } } ];
var dataParsers = require('./src/dataParsers');
var strategy = new strategyFn(process.argv[2], combinations);

// strategy.setShowTrades(true);

try {
    // Parse the raw data file.
    dataParsers.ctoption.parse('./data/ctoption/' + process.argv[2] + '.csv').then(function(parsedData) {
        strategy.setProfitLoss(5000);
        strategy.backtest(parsedData, 120, 0.76);
    });
}
catch (error) {
    console.error(error.message || error);
    process.exit(1);
}
