var _ = require('lodash');

var configurations = [ { "ema200" : true, "ema100" : true, "ema50" : true, "sma13" : false, "rsi" : { "rsi" : "rsi7", "overbought" : 77, "oversold" : 23 }, "stochastic" : { "K" : "stochastic10K", "D" : "stochastic10D", "overbought" : 80, "oversold" : 20 }, "prChannel" : { "upper" : "prChannelUpper350_4_195", "lower" : "prChannelLower350_4_195" } }, { "ema200" : true, "ema100" : true, "ema50" : true, "sma13" : false, "rsi" : { "rsi" : "rsi7", "overbought" : 77, "oversold" : 23 }, "stochastic" : { "K" : "stochastic10K", "D" : "stochastic10D", "overbought" : 77, "oversold" : 23 }, "prChannel" : { "upper" : "prChannelUpper350_4_195", "lower" : "prChannelLower350_4_195" } }, { "ema200" : true, "ema100" : true, "ema50" : true, "sma13" : false, "rsi" : { "rsi" : "rsi7", "overbought" : 77, "oversold" : 23 }, "stochastic" : { "K" : "stochastic21K", "D" : "stochastic21D", "overbought" : 77, "oversold" : 23 }, "prChannel" : { "upper" : "prChannelUpper350_4_195", "lower" : "prChannelLower350_4_195" } }, { "ema200" : true, "ema100" : true, "ema50" : true, "sma13" : false, "rsi" : { "rsi" : "rsi7", "overbought" : 77, "oversold" : 23 }, "stochastic" : { "K" : "stochastic5K", "D" : "stochastic5D", "overbought" : 77, "oversold" : 23 }, "prChannel" : { "upper" : "prChannelUpper350_4_195", "lower" : "prChannelLower350_4_195" } }, { "ema200" : true, "ema100" : true, "ema50" : true, "sma13" : false, "rsi" : { "rsi" : "rsi7", "overbought" : 77, "oversold" : 23 }, "stochastic" : { "K" : "stochastic14K", "D" : "stochastic14D", "overbought" : 77, "oversold" : 23 }, "prChannel" : { "upper" : "prChannelUpper350_4_195", "lower" : "prChannelLower350_4_195" } }, { "ema200" : true, "ema100" : true, "ema50" : true, "sma13" : false, "rsi" : { "rsi" : "rsi7", "overbought" : 77, "oversold" : 23 }, "stochastic" : { "K" : "stochastic14K", "D" : "stochastic14D", "overbought" : 77, "oversold" : 23 }, "prChannel" : { "upper" : "prChannelUpper350_4_19", "lower" : "prChannelLower350_4_19" } }, { "ema200" : true, "ema100" : true, "ema50" : true, "sma13" : false, "rsi" : { "rsi" : "rsi7", "overbought" : 77, "oversold" : 23 }, "stochastic" : { "K" : "stochastic21K", "D" : "stochastic21D", "overbought" : 77, "oversold" : 23 }, "prChannel" : { "upper" : "prChannelUpper350_4_185", "lower" : "prChannelLower350_4_185" } }, { "ema200" : true, "ema100" : true, "ema50" : true, "sma13" : false, "rsi" : { "rsi" : "rsi5", "overbought" : 80, "oversold" : 20 }, "stochastic" : { "K" : "stochastic5K", "D" : "stochastic5D", "overbought" : 80, "oversold" : 20 }, "prChannel" : { "upper" : "prChannelUpper400_4_20", "lower" : "prChannelLower400_4_20" } } ];

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
