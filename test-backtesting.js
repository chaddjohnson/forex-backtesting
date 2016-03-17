var strategyFn = require('./src/strategies/combined/Reversals');
var combinations = [
    { "ema200" : false, "ema100" : true, "ema50" : true, "sma13" : false, "rsi" : null, "stochastic" : { "K" : "stochastic10K", "D" : "stochastic10D", "overbought" : 77, "oversold" : 23 }, "prChannel" : { "upper" : "prChannelUpper200_4_215", "lower" : "prChannelLower200_4_215" } },
    { "ema200" : false, "ema100" : true, "ema50" : true, "sma13" : true, "rsi" : null, "stochastic" : { "K" : "stochastic10K", "D" : "stochastic10D", "overbought" : 80, "oversold" : 20 }, "prChannel" : { "upper" : "prChannelUpper200_4_205", "lower" : "prChannelLower200_4_205" } },
    { "ema200" : true, "ema100" : false, "ema50" : true, "sma13" : true, "rsi" : null, "stochastic" : { "K" : "stochastic10K", "D" : "stochastic10D", "overbought" : 77, "oversold" : 23 }, "prChannel" : { "upper" : "prChannelUpper200_4_21", "lower" : "prChannelLower200_4_21" } }
];
var dataParsers = require('./src/dataParsers');
var strategy = new strategyFn(process.argv[2], 25, combinations);

// strategy.setShowTrades(true);

try {
    // Parse the raw data file.
    dataParsers.metatrader.parse('./data/' + process.argv[2] + '.csv').then(function(parsedData) {
        strategy.setProfitLoss(5000);
        strategy.backtest(parsedData, 0.76);
    });
}
catch (error) {
    console.error(error.message || error);
    process.exit(1);
}
