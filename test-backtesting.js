var strategyFn = require('./src/strategies/combined/Reversals');
var combinations = [
    {
        "trendPrChannel": {
            "regression": "trendPrChannel650_2"
        },
        "prChannel": {
            "lower": "prChannelLower100_2_215",
            "upper": "prChannelUpper100_2_215"
        },
        "rsi": {
            "oversold": 20,
            "overbought": 80,
            "rsi": "rsi5"
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
            "lower": "prChannelLower100_2_195",
            "upper": "prChannelUpper100_2_195"
        },
        "rsi": {
            "oversold": 20,
            "overbought": 80,
            "rsi": "rsi7"
        },
        "sma13": false,
        "ema50": false,
        "ema100": false,
        "ema200": false
    },
    {
        "trendPrChannel": {
            "regression": "trendPrChannel650_2"
        },
        "prChannel": {
            "lower": "prChannelLower100_2_19",
            "upper": "prChannelUpper100_2_19"
        },
        "rsi": {
            "oversold": 20,
            "overbought": 80,
            "rsi": "rsi7"
        },
        "sma13": false,
        "ema50": false,
        "ema100": false,
        "ema200": false
    },
    {
        "trendPrChannel": {
            "regression": "trendPrChannel600_2"
        },
        "prChannel": {
            "lower": "prChannelLower100_2_19",
            "upper": "prChannelUpper100_2_19"
        },
        "rsi": {
            "oversold": 20,
            "overbought": 80,
            "rsi": "rsi7"
        },
        "sma13": false,
        "ema50": false,
        "ema100": false,
        "ema200": false
    },
    {
        "trendPrChannel": {
            "regression": "trendPrChannel850_2"
        },
        "prChannel": {
            "lower": "prChannelLower200_3_19",
            "upper": "prChannelUpper200_3_19"
        },
        "rsi": {
            "oversold": 20,
            "overbought": 80,
            "rsi": "rsi7"
        },
        "sma13": false,
        "ema50": false,
        "ema100": false,
        "ema200": false
    },
    {
        "trendPrChannel": {
            "regression": "trendPrChannel850_2"
        },
        "prChannel": {
            "lower": "prChannelLower100_2_195",
            "upper": "prChannelUpper100_2_195"
        },
        "rsi": {
            "oversold": 20,
            "overbought": 80,
            "rsi": "rsi7"
        },
        "sma13": true,
        "ema50": true,
        "ema100": false,
        "ema200": false
    },
    {
        "trendPrChannel": {
            "regression": "trendPrChannel650_2"
        },
        "prChannel": {
            "lower": "prChannelLower100_3_195",
            "upper": "prChannelUpper100_3_195"
        },
        "rsi": {
            "oversold": 20,
            "overbought": 80,
            "rsi": "rsi7"
        },
        "sma13": false,
        "ema50": false,
        "ema100": false,
        "ema200": false
    },
    {
        "trendPrChannel": {
            "regression": "trendPrChannel750_2"
        },
        "prChannel": {
            "lower": "prChannelLower300_2_19",
            "upper": "prChannelUpper300_2_19"
        },
        "rsi": {
            "oversold": 20,
            "overbought": 80,
            "rsi": "rsi5"
        },
        "sma13": true,
        "ema50": true,
        "ema100": true,
        "ema200": true
    },
    {
        "trendPrChannel": {
            "regression": "trendPrChannel800_2"
        },
        "prChannel": {
            "lower": "prChannelLower300_2_19",
            "upper": "prChannelUpper300_2_19"
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
    },
    {
        "trendPrChannel": {
            "regression": "trendPrChannel650_2"
        },
        "prChannel": {
            "lower": "prChannelLower200_4_21",
            "upper": "prChannelUpper200_4_21"
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
            "regression": "trendPrChannel650_2"
        },
        "prChannel": {
            "lower": "prChannelLower100_3_19",
            "upper": "prChannelUpper100_3_19"
        },
        "rsi": {
            "oversold": 20,
            "overbought": 80,
            "rsi": "rsi7"
        },
        "sma13": true,
        "ema50": true,
        "ema100": false,
        "ema200": false
    },
    {
        "trendPrChannel": {
            "regression": "trendPrChannel850_2"
        },
        "prChannel": {
            "lower": "prChannelLower100_3_21",
            "upper": "prChannelUpper100_3_21"
        },
        "rsi": {
            "oversold": 20,
            "overbought": 80,
            "rsi": "rsi7"
        },
        "sma13": true,
        "ema50": true,
        "ema100": false,
        "ema200": false
    },
    {
        "trendPrChannel": {
            "regression": "trendPrChannel600_2"
        },
        "prChannel": {
            "lower": "prChannelLower300_2_21",
            "upper": "prChannelUpper300_2_21"
        },
        "rsi": {
            "oversold": 20,
            "overbought": 80,
            "rsi": "rsi7"
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
            "lower": "prChannelLower200_4_195",
            "upper": "prChannelUpper200_4_195"
        },
        "rsi": {
            "oversold": 20,
            "overbought": 80,
            "rsi": "rsi7"
        },
        "sma13": false,
        "ema50": false,
        "ema100": false,
        "ema200": false
    },
    {
        "trendPrChannel": {
            "regression": "trendPrChannel850_2"
        },
        "prChannel": {
            "lower": "prChannelLower100_4_20",
            "upper": "prChannelUpper100_4_20"
        },
        "rsi": {
            "oversold": 20,
            "overbought": 80,
            "rsi": "rsi7"
        },
        "sma13": false,
        "ema50": false,
        "ema100": false,
        "ema200": false
    },
    {
        "trendPrChannel": {
            "regression": "trendPrChannel300_2"
        },
        "prChannel": {
            "lower": "prChannelLower100_4_195",
            "upper": "prChannelUpper100_4_195"
        },
        "rsi": {
            "oversold": 20,
            "overbought": 80,
            "rsi": "rsi5"
        },
        "sma13": false,
        "ema50": false,
        "ema100": false,
        "ema200": false
    },
    {
        "trendPrChannel": {
            "regression": "trendPrChannel600_2"
        },
        "prChannel": {
            "lower": "prChannelLower100_4_19",
            "upper": "prChannelUpper100_4_19"
        },
        "rsi": {
            "oversold": 20,
            "overbought": 80,
            "rsi": "rsi7"
        },
        "sma13": false,
        "ema50": false,
        "ema100": false,
        "ema200": false
    }
];
var dataParsers = require('./src/dataParsers');
var strategy = new strategyFn('EURGBP', combinations);

// strategy.setShowTrades(true);

try {
    // Parse the raw data file.
    dataParsers.ctoption.parse('./data/ctoption/EURGBP.csv').then(function(parsedData) {
        strategy.backtest(parsedData, 1000, 0.76);
    });
}
catch (error) {
    console.error(error.message || error);
    process.exit(1);
}
