var strategyFn = require('./src/strategies/combined/Reversals');
var combinations = [
    {
        "trendPrChannel": {
            "regression": "trendPrChannel700_2"
        },
        "prChannel": {
            "lower": "prChannelLower100_3_21",
            "upper": "prChannelUpper100_3_21"
        },
        "rsi": {
            "oversold": 20,
            "overbought": 80,
            "rsi": "rsi5"
        },
        "sma13": true,
        "ema50": false,
        "ema100": true,
        "ema200": true
    },
    {
        "trendPrChannel": {
            "regression": "trendPrChannel700_2"
        },
        "prChannel": {
            "lower": "prChannelLower100_3_21",
            "upper": "prChannelUpper100_3_21"
        },
        "rsi": {
            "oversold": 23,
            "overbought": 77,
            "rsi": "rsi7"
        },
        "sma13": true,
        "ema50": false,
        "ema100": true,
        "ema200": true
    },
    {
        "trendPrChannel": {
            "regression": "trendPrChannel750_2"
        },
        "prChannel": {
            "lower": "prChannelLower100_3_21",
            "upper": "prChannelUpper100_3_21"
        },
        "rsi": {
            "oversold": 20,
            "overbought": 80,
            "rsi": "rsi5"
        },
        "sma13": false,
        "ema50": false,
        "ema100": true,
        "ema200": true
    },
    {
        "trendPrChannel": {
            "regression": "trendPrChannel700_2"
        },
        "prChannel": {
            "lower": "prChannelLower100_3_20",
            "upper": "prChannelUpper100_3_20"
        },
        "rsi": {
            "oversold": 20,
            "overbought": 80,
            "rsi": "rsi7"
        },
        "sma13": true,
        "ema50": false,
        "ema100": true,
        "ema200": true
    },
    {
        "trendPrChannel": {
            "regression": "trendPrChannel700_2"
        },
        "prChannel": {
            "lower": "prChannelLower100_3_20",
            "upper": "prChannelUpper100_3_20"
        },
        "rsi": {
            "oversold": 20,
            "overbought": 80,
            "rsi": "rsi7"
        },
        "sma13": false,
        "ema50": true,
        "ema100": true,
        "ema200": false
    },
    {
        "trendPrChannel": {
            "regression": "trendPrChannel200_2"
        },
        "prChannel": {
            "lower": "prChannelLower300_3_21",
            "upper": "prChannelUpper300_3_21"
        },
        "rsi": {
            "oversold": 5,
            "overbought": 95,
            "rsi": "rsi2"
        },
        "sma13": false,
        "ema50": false,
        "ema100": false,
        "ema200": false
    },
    {
        "trendPrChannel": {
            "regression": "trendPrChannel700_2"
        },
        "prChannel": {
            "lower": "prChannelLower200_3_195",
            "upper": "prChannelUpper200_3_195"
        },
        "rsi": {
            "oversold": 20,
            "overbought": 80,
            "rsi": "rsi7"
        },
        "sma13": false,
        "ema50": false,
        "ema100": true,
        "ema200": true
    },
    {
        "trendPrChannel": {
            "regression": "trendPrChannel750_2"
        },
        "prChannel": {
            "lower": "prChannelLower300_2_195",
            "upper": "prChannelUpper300_2_195"
        },
        "rsi": {
            "oversold": 23,
            "overbought": 77,
            "rsi": "rsi7"
        },
        "sma13": true,
        "ema50": true,
        "ema100": true,
        "ema200": true
    }
];
var dataParsers = require('./src/dataParsers');
var strategy = new strategyFn(process.argv[2], combinations);

// strategy.setShowTrades(true);

try {
    // Parse the raw data file.
    dataParsers.ctoption.parse('./data/ctoption/' + process.argv[2] + '.csv').then(function(parsedData) {
        strategy.backtest(parsedData, 1000, 0.76);
    });
}
catch (error) {
    console.error(error.message || error);
    process.exit(1);
}
