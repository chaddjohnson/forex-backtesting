var _ = require('lodash');

var configurations = [
    { "ema200" : false, "ema100" : true, "ema50" : true, "sma13" : false, "rsi" : { "rsi" : "rsi9", "overbought" : 70, "oversold" : 30 }, "stochastic" : { "K" : "stochastic10K", "D" : "stochastic10D", "overbought" : 80, "oversold" : 20 }, "prChannel" : { "upper" : "prChannelUpper200_4_21", "lower" : "prChannelLower200_4_21" } },
    { "ema200" : false, "ema100" : true, "ema50" : true, "sma13" : false, "rsi" : { "rsi" : "rsi5", "overbought" : 80, "oversold" : 20 }, "stochastic" : { "K" : "stochastic10K", "D" : "stochastic10D", "overbought" : 77, "oversold" : 23 }, "prChannel" : { "upper" : "prChannelUpper200_4_215", "lower" : "prChannelLower200_4_215" } },
    { "ema200" : false, "ema100" : true, "ema50" : true, "sma13" : false, "rsi" : { "rsi" : "rsi9", "overbought" : 70, "oversold" : 30 }, "stochastic" : { "K" : "stochastic10K", "D" : "stochastic10D", "overbought" : 80, "oversold" : 20 }, "prChannel" : { "upper" : "prChannelUpper200_4_205", "lower" : "prChannelLower200_4_205" } },
    { "ema200" : false, "ema100" : true, "ema50" : true, "sma13" : false, "rsi" : { "rsi" : "rsi9", "overbought" : 70, "oversold" : 30 }, "stochastic" : { "K" : "stochastic14K", "D" : "stochastic14D", "overbought" : 77, "oversold" : 23 }, "prChannel" : { "upper" : "prChannelUpper200_4_21", "lower" : "prChannelLower200_4_21" } },
    { "ema200" : false, "ema100" : true, "ema50" : true, "sma13" : false, "rsi" : { "rsi" : "rsi7", "overbought" : 77, "oversold" : 23 }, "stochastic" : { "K" : "stochastic10K", "D" : "stochastic10D", "overbought" : 80, "oversold" : 20 }, "prChannel" : { "upper" : "prChannelUpper200_4_2", "lower" : "prChannelLower200_4_2" } },
    { "ema200" : false, "ema100" : true, "ema50" : true, "sma13" : false, "rsi" : null, "stochastic" : { "K" : "stochastic14K", "D" : "stochastic14D", "overbought" : 77, "oversold" : 23 }, "prChannel" : { "upper" : "prChannelUpper250_5_205", "lower" : "prChannelLower250_5_205" } },
    { "ema200" : false, "ema100" : false, "ema50" : true, "sma13" : true, "rsi" : { "rsi" : "rsi9", "overbought" : 77, "oversold" : 23 }, "stochastic" : null, "prChannel" : { "upper" : "prChannelUpper200_3_185", "lower" : "prChannelLower200_3_185" } },
    { "ema200" : false, "ema100" : true, "ema50" : true, "sma13" : false, "rsi" : null, "stochastic" : { "K" : "stochastic10K", "D" : "stochastic10D", "overbought" : 77, "oversold" : 23 }, "prChannel" : { "upper" : "prChannelUpper200_4_225", "lower" : "prChannelLower200_4_225" } },
    { "ema200" : false, "ema100" : true, "ema50" : true, "sma13" : false, "rsi" : null, "stochastic" : { "K" : "stochastic14K", "D" : "stochastic14D", "overbought" : 77, "oversold" : 23 }, "prChannel" : { "upper" : "prChannelUpper200_4_220", "lower" : "prChannelLower200_4_220" } },
    { "ema200" : false, "ema100" : true, "ema50" : true, "sma13" : false, "rsi" : null, "stochastic" : { "K" : "stochastic14K", "D" : "stochastic14D", "overbought" : 70, "oversold" : 30 }, "prChannel" : { "upper" : "prChannelUpper200_4_230", "lower" : "prChannelLower200_4_230" } }
];
var trendPrChannels = [];
var prChannels = [];

configurations.forEach(function(configuration) {
    if (configuration.trendPrChannel && configuration.trendPrChannel.regression) {
        trendPrChannels.push(configuration.trendPrChannel.regression);
    }

    if (configuration.prChannel && configuration.prChannel.upper) {
        prChannels.push(configuration.prChannel.upper);
    }
});

trendPrChannels = _.uniq(trendPrChannels);
trendPrChannels.sort();

prChannels = _.uniq(prChannels);
prChannels.sort();

console.log(trendPrChannels);
console.log();
console.log(prChannels);
