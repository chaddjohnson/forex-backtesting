var Base = require('./Base');
var strategies = require('../src/strategies');

var configurationOptions = {
    ema200: [false, true],
    ema100: [false, true],
    ema50: [false, true],
    ema13: [false, true],
    sma13: [false, true],
    rsi: [
        null,
        {study: 'rsi14', overbought: 70, oversold: 30},
        {study: 'rsi7', overbought: 77, oversold: 23},
        {study: 'rsi7', overbought: 80, oversold: 20},
        {study: 'rsi5', overbought: 80, oversold: 20},
        {study: 'rsi2', overbought: 95, oversold: 5}
    ],
    prChannel: [
        null,
        {upper: 'prChannelUpper100_2_1618', lower: 'prChannelLower100_2_1618', close: false},
        {upper: 'prChannelUpper100_3_1618', lower: 'prChannelLower100_3_1618', close: false},
        {upper: 'prChannelUpper100_4_1618', lower: 'prChannelLower100_4_1618', close: false},
        {upper: 'prChannelUpper100_2_19', lower: 'prChannelLower100_2_17', close: false},
        {upper: 'prChannelUpper100_3_19', lower: 'prChannelLower100_3_17', close: false},
        {upper: 'prChannelUpper100_4_19', lower: 'prChannelLower100_4_17', close: false},
        {upper: 'prChannelUpper100_2_19', lower: 'prChannelLower100_2_19', close: false},
        {upper: 'prChannelUpper100_3_19', lower: 'prChannelLower100_3_19', close: false},
        {upper: 'prChannelUpper100_4_19', lower: 'prChannelLower100_4_19', close: false},
        {upper: 'prChannelUpper100_2_195', lower: 'prChannelLower100_2_195', close: false},
        {upper: 'prChannelUpper100_3_195', lower: 'prChannelLower100_3_195', close: false},
        {upper: 'prChannelUpper100_4_195', lower: 'prChannelLower100_4_195', close: false},
        {upper: 'prChannelUpper100_2_20', lower: 'prChannelLower100_2_20', close: false},
        {upper: 'prChannelUpper100_3_20', lower: 'prChannelLower100_3_20', close: false},
        {upper: 'prChannelUpper100_4_20', lower: 'prChannelLower100_4_20', close: false},
        {upper: 'prChannelUpper100_2_20', lower: 'prChannelLower100_2_21', close: false},
        {upper: 'prChannelUpper100_3_20', lower: 'prChannelLower100_3_21', close: false},
        {upper: 'prChannelUpper100_4_20', lower: 'prChannelLower100_4_21', close: false},
        {upper: 'prChannelUpper200_2_1618', lower: 'prChannelLower200_2_1618', close: false},
        {upper: 'prChannelUpper200_3_1618', lower: 'prChannelLower200_3_1618', close: false},
        {upper: 'prChannelUpper200_4_1618', lower: 'prChannelLower200_4_1618', close: false},
        {upper: 'prChannelUpper200_2_19', lower: 'prChannelLower200_2_17', close: false},
        {upper: 'prChannelUpper200_3_19', lower: 'prChannelLower200_3_17', close: false},
        {upper: 'prChannelUpper200_4_19', lower: 'prChannelLower200_4_17', close: false},
        {upper: 'prChannelUpper200_2_19', lower: 'prChannelLower200_2_19', close: false},
        {upper: 'prChannelUpper200_3_19', lower: 'prChannelLower200_3_19', close: false},
        {upper: 'prChannelUpper200_4_19', lower: 'prChannelLower200_4_19', close: false},
        {upper: 'prChannelUpper200_2_195', lower: 'prChannelLower200_2_195', close: false},
        {upper: 'prChannelUpper200_3_195', lower: 'prChannelLower200_3_195', close: false},
        {upper: 'prChannelUpper200_4_195', lower: 'prChannelLower200_4_195', close: false},
        {upper: 'prChannelUpper200_2_20', lower: 'prChannelLower200_2_20', close: false},
        {upper: 'prChannelUpper200_3_20', lower: 'prChannelLower200_3_20', close: false},
        {upper: 'prChannelUpper200_4_20', lower: 'prChannelLower200_4_20', close: false},
        {upper: 'prChannelUpper200_2_20', lower: 'prChannelLower200_2_21', close: false},
        {upper: 'prChannelUpper200_3_20', lower: 'prChannelLower200_3_21', close: false},
        {upper: 'prChannelUpper200_4_20', lower: 'prChannelLower200_4_21', close: false},
        {upper: 'prChannelUpper250_2_1618', lower: 'prChannelLower250_2_1618', close: false},
        {upper: 'prChannelUpper250_3_1618', lower: 'prChannelLower250_3_1618', close: false},
        {upper: 'prChannelUpper250_4_1618', lower: 'prChannelLower250_4_1618', close: false},
        {upper: 'prChannelUpper250_2_19', lower: 'prChannelLower250_2_17', close: false},
        {upper: 'prChannelUpper250_3_19', lower: 'prChannelLower250_3_17', close: false},
        {upper: 'prChannelUpper250_4_19', lower: 'prChannelLower250_4_17', close: false},
        {upper: 'prChannelUpper250_2_19', lower: 'prChannelLower250_2_19', close: false},
        {upper: 'prChannelUpper250_3_19', lower: 'prChannelLower250_3_19', close: false},
        {upper: 'prChannelUpper250_4_19', lower: 'prChannelLower250_4_19', close: false},
        {upper: 'prChannelUpper250_2_195', lower: 'prChannelLower250_2_195', close: false},
        {upper: 'prChannelUpper250_3_195', lower: 'prChannelLower250_3_195', close: false},
        {upper: 'prChannelUpper250_4_195', lower: 'prChannelLower250_4_195', close: false},
        {upper: 'prChannelUpper250_2_20', lower: 'prChannelLower250_2_20', close: false},
        {upper: 'prChannelUpper250_3_20', lower: 'prChannelLower250_3_20', close: false},
        {upper: 'prChannelUpper250_4_20', lower: 'prChannelLower250_4_20', close: false},
        {upper: 'prChannelUpper250_2_20', lower: 'prChannelLower250_2_21', close: false},
        {upper: 'prChannelUpper250_3_20', lower: 'prChannelLower250_3_21', close: false},
        {upper: 'prChannelUpper250_4_20', lower: 'prChannelLower250_4_21', close: false},
        {upper: 'prChannelUpper300_2_1618', lower: 'prChannelLower300_2_1618', close: false},
        {upper: 'prChannelUpper300_3_1618', lower: 'prChannelLower300_3_1618', close: false},
        {upper: 'prChannelUpper300_4_1618', lower: 'prChannelLower300_4_1618', close: false},
        {upper: 'prChannelUpper300_2_19', lower: 'prChannelLower300_2_17', close: false},
        {upper: 'prChannelUpper300_3_19', lower: 'prChannelLower300_3_17', close: false},
        {upper: 'prChannelUpper300_4_19', lower: 'prChannelLower300_4_17', close: false},
        {upper: 'prChannelUpper300_2_19', lower: 'prChannelLower300_2_19', close: false},
        {upper: 'prChannelUpper300_3_19', lower: 'prChannelLower300_3_19', close: false},
        {upper: 'prChannelUpper300_4_19', lower: 'prChannelLower300_4_19', close: false},
        {upper: 'prChannelUpper300_2_195', lower: 'prChannelLower300_2_195', close: false},
        {upper: 'prChannelUpper300_3_195', lower: 'prChannelLower300_3_195', close: false},
        {upper: 'prChannelUpper300_4_195', lower: 'prChannelLower300_4_195', close: false},
        {upper: 'prChannelUpper300_2_20', lower: 'prChannelLower300_2_20', close: false},
        {upper: 'prChannelUpper300_3_20', lower: 'prChannelLower300_3_20', close: false},
        {upper: 'prChannelUpper300_4_20', lower: 'prChannelLower300_4_20', close: false},
        {upper: 'prChannelUpper300_2_20', lower: 'prChannelLower300_2_21', close: false},
        {upper: 'prChannelUpper300_3_20', lower: 'prChannelLower300_3_21', close: false},
        {upper: 'prChannelUpper300_4_20', lower: 'prChannelLower300_4_21', close: false},
        {upper: 'prChannelUpper100_2_1618', lower: 'prChannelLower100_2_1618', close: true},
        {upper: 'prChannelUpper100_3_1618', lower: 'prChannelLower100_3_1618', close: true},
        {upper: 'prChannelUpper100_4_1618', lower: 'prChannelLower100_4_1618', close: true},
        {upper: 'prChannelUpper100_2_19', lower: 'prChannelLower100_2_17', close: true},
        {upper: 'prChannelUpper100_3_19', lower: 'prChannelLower100_3_17', close: true},
        {upper: 'prChannelUpper100_4_19', lower: 'prChannelLower100_4_17', close: true},
        {upper: 'prChannelUpper100_2_19', lower: 'prChannelLower100_2_19', close: true},
        {upper: 'prChannelUpper100_3_19', lower: 'prChannelLower100_3_19', close: true},
        {upper: 'prChannelUpper100_4_19', lower: 'prChannelLower100_4_19', close: true},
        {upper: 'prChannelUpper100_2_195', lower: 'prChannelLower100_2_195', close: true},
        {upper: 'prChannelUpper100_3_195', lower: 'prChannelLower100_3_195', close: true},
        {upper: 'prChannelUpper100_4_195', lower: 'prChannelLower100_4_195', close: true},
        {upper: 'prChannelUpper100_2_20', lower: 'prChannelLower100_2_20', close: true},
        {upper: 'prChannelUpper100_3_20', lower: 'prChannelLower100_3_20', close: true},
        {upper: 'prChannelUpper100_4_20', lower: 'prChannelLower100_4_20', close: true},
        {upper: 'prChannelUpper100_2_20', lower: 'prChannelLower100_2_21', close: true},
        {upper: 'prChannelUpper100_3_20', lower: 'prChannelLower100_3_21', close: true},
        {upper: 'prChannelUpper100_4_20', lower: 'prChannelLower100_4_21', close: true},
        {upper: 'prChannelUpper150_2_1618', lower: 'prChannelLower150_2_1618', close: true},
        {upper: 'prChannelUpper150_3_1618', lower: 'prChannelLower150_3_1618', close: true},
        {upper: 'prChannelUpper150_4_1618', lower: 'prChannelLower150_4_1618', close: true},
        {upper: 'prChannelUpper150_2_19', lower: 'prChannelLower150_2_17', close: true},
        {upper: 'prChannelUpper150_3_19', lower: 'prChannelLower150_3_17', close: true},
        {upper: 'prChannelUpper150_4_19', lower: 'prChannelLower150_4_17', close: true},
        {upper: 'prChannelUpper150_2_19', lower: 'prChannelLower150_2_19', close: true},
        {upper: 'prChannelUpper150_3_19', lower: 'prChannelLower150_3_19', close: true},
        {upper: 'prChannelUpper150_4_19', lower: 'prChannelLower150_4_19', close: true},
        {upper: 'prChannelUpper150_2_195', lower: 'prChannelLower150_2_195', close: true},
        {upper: 'prChannelUpper150_3_195', lower: 'prChannelLower150_3_195', close: true},
        {upper: 'prChannelUpper150_4_195', lower: 'prChannelLower150_4_195', close: true},
        {upper: 'prChannelUpper150_2_20', lower: 'prChannelLower150_2_20', close: true},
        {upper: 'prChannelUpper150_3_20', lower: 'prChannelLower150_3_20', close: true},
        {upper: 'prChannelUpper150_4_20', lower: 'prChannelLower150_4_20', close: true},
        {upper: 'prChannelUpper150_2_20', lower: 'prChannelLower150_2_21', close: true},
        {upper: 'prChannelUpper150_3_20', lower: 'prChannelLower150_3_21', close: true},
        {upper: 'prChannelUpper150_4_20', lower: 'prChannelLower150_4_21', close: true},
        {upper: 'prChannelUpper200_2_1618', lower: 'prChannelLower200_2_1618', close: true},
        {upper: 'prChannelUpper200_3_1618', lower: 'prChannelLower200_3_1618', close: true},
        {upper: 'prChannelUpper200_4_1618', lower: 'prChannelLower200_4_1618', close: true},
        {upper: 'prChannelUpper200_2_19', lower: 'prChannelLower200_2_17', close: true},
        {upper: 'prChannelUpper200_3_19', lower: 'prChannelLower200_3_17', close: true},
        {upper: 'prChannelUpper200_4_19', lower: 'prChannelLower200_4_17', close: true},
        {upper: 'prChannelUpper200_2_19', lower: 'prChannelLower200_2_19', close: true},
        {upper: 'prChannelUpper200_3_19', lower: 'prChannelLower200_3_19', close: true},
        {upper: 'prChannelUpper200_4_19', lower: 'prChannelLower200_4_19', close: true},
        {upper: 'prChannelUpper200_2_195', lower: 'prChannelLower200_2_195', close: true},
        {upper: 'prChannelUpper200_3_195', lower: 'prChannelLower200_3_195', close: true},
        {upper: 'prChannelUpper200_4_195', lower: 'prChannelLower200_4_195', close: true},
        {upper: 'prChannelUpper200_2_20', lower: 'prChannelLower200_2_20', close: true},
        {upper: 'prChannelUpper200_3_20', lower: 'prChannelLower200_3_20', close: true},
        {upper: 'prChannelUpper200_4_20', lower: 'prChannelLower200_4_20', close: true},
        {upper: 'prChannelUpper200_2_20', lower: 'prChannelLower200_2_21', close: true},
        {upper: 'prChannelUpper200_3_20', lower: 'prChannelLower200_3_21', close: true},
        {upper: 'prChannelUpper200_4_20', lower: 'prChannelLower200_4_21', close: true},
        {upper: 'prChannelUpper250_2_1618', lower: 'prChannelLower250_2_1618', close: true},
        {upper: 'prChannelUpper250_3_1618', lower: 'prChannelLower250_3_1618', close: true},
        {upper: 'prChannelUpper250_4_1618', lower: 'prChannelLower250_4_1618', close: true},
        {upper: 'prChannelUpper250_2_19', lower: 'prChannelLower250_2_17', close: true},
        {upper: 'prChannelUpper250_3_19', lower: 'prChannelLower250_3_17', close: true},
        {upper: 'prChannelUpper250_4_19', lower: 'prChannelLower250_4_17', close: true},
        {upper: 'prChannelUpper250_2_19', lower: 'prChannelLower250_2_19', close: true},
        {upper: 'prChannelUpper250_3_19', lower: 'prChannelLower250_3_19', close: true},
        {upper: 'prChannelUpper250_4_19', lower: 'prChannelLower250_4_19', close: true},
        {upper: 'prChannelUpper250_2_195', lower: 'prChannelLower250_2_195', close: true},
        {upper: 'prChannelUpper250_3_195', lower: 'prChannelLower250_3_195', close: true},
        {upper: 'prChannelUpper250_4_195', lower: 'prChannelLower250_4_195', close: true},
        {upper: 'prChannelUpper250_2_20', lower: 'prChannelLower250_2_20', close: true},
        {upper: 'prChannelUpper250_3_20', lower: 'prChannelLower250_3_20', close: true},
        {upper: 'prChannelUpper250_4_20', lower: 'prChannelLower250_4_20', close: true},
        {upper: 'prChannelUpper250_2_20', lower: 'prChannelLower250_2_21', close: true},
        {upper: 'prChannelUpper250_3_20', lower: 'prChannelLower250_3_21', close: true},
        {upper: 'prChannelUpper250_4_20', lower: 'prChannelLower250_4_21', close: true},
        {upper: 'prChannelUpper300_2_1618', lower: 'prChannelLower300_2_1618', close: true},
        {upper: 'prChannelUpper300_3_1618', lower: 'prChannelLower300_3_1618', close: true},
        {upper: 'prChannelUpper300_4_1618', lower: 'prChannelLower300_4_1618', close: true},
        {upper: 'prChannelUpper300_2_19', lower: 'prChannelLower300_2_17', close: true},
        {upper: 'prChannelUpper300_3_19', lower: 'prChannelLower300_3_17', close: true},
        {upper: 'prChannelUpper300_4_19', lower: 'prChannelLower300_4_17', close: true},
        {upper: 'prChannelUpper300_2_19', lower: 'prChannelLower300_2_19', close: true},
        {upper: 'prChannelUpper300_3_19', lower: 'prChannelLower300_3_19', close: true},
        {upper: 'prChannelUpper300_4_19', lower: 'prChannelLower300_4_19', close: true},
        {upper: 'prChannelUpper300_2_195', lower: 'prChannelLower300_2_195', close: true},
        {upper: 'prChannelUpper300_3_195', lower: 'prChannelLower300_3_195', close: true},
        {upper: 'prChannelUpper300_4_195', lower: 'prChannelLower300_4_195', close: true},
        {upper: 'prChannelUpper300_2_20', lower: 'prChannelLower300_2_20', close: true},
        {upper: 'prChannelUpper300_3_20', lower: 'prChannelLower300_3_20', close: true},
        {upper: 'prChannelUpper300_4_20', lower: 'prChannelLower300_4_20', close: true},
        {upper: 'prChannelUpper300_2_20', lower: 'prChannelLower300_2_21', close: true},
        {upper: 'prChannelUpper300_3_20', lower: 'prChannelLower300_3_21', close: true},
        {upper: 'prChannelUpper300_4_20', lower: 'prChannelLower300_4_21', close: true}
    ],
    trendPrChannel: [
        null,
        {regression: 'prChannel200_2'},
        {regression: 'prChannel200_3'},
        {regression: 'prChannel200_4'},
        {regression: 'prChannel300_2'},
        {regression: 'prChannel300_3'},
        {regression: 'prChannel300_4'},
        {regression: 'prChannel400_2'},
        {regression: 'prChannel400_3'},
        {regression: 'prChannel400_4'},
        {regression: 'prChannel450_2'},
        {regression: 'prChannel450_3'},
        {regression: 'prChannel450_4'},
        {regression: 'prChannel500_2'},
        {regression: 'prChannel500_3'},
        {regression: 'prChannel500_4'},
        {regression: 'prChannel550_2'},
        {regression: 'prChannel550_3'},
        {regression: 'prChannel550_4'},
        {regression: 'prChannel600_2'},
        {regression: 'prChannel600_3'},
        {regression: 'prChannel600_4'},
        {regression: 'prChannel650_2'},
        {regression: 'prChannel650_3'},
        {regression: 'prChannel650_4'}
    ]
};

// Define studies to use.
var studyDefinitions = [
    {
        study: studies.Ema,
        inputs: {
            length: 200
        },
        outputMap: {
            ema: 'ema200'
        }
    },{
        study: studies.Ema,
        inputs: {
            length: 100
        },
        outputMap: {
            ema: 'ema100'
        }
    },{
        study: studies.Ema,
        inputs: {
            length: 50
        },
        outputMap: {
            ema: 'ema50'
        }
    },{
        study: studies.Sma,
        inputs: {
            length: 13
        },
        outputMap: {
            sma: 'sma13'
        }
    },{
        study: studies.Ema,
        inputs: {
            length: 13
        },
        outputMap: {
            ema: 'ema13'
        }
    },{
        study: studies.Rsi,
        inputs: {
            length: 14
        },
        outputMap: {
            rsi: 'rsi14'
        }
    },{
        study: studies.Rsi,
        inputs: {
            length: 7
        },
        outputMap: {
            rsi: 'rsi7'
        }
    },{
        study: studies.Rsi,
        inputs: {
            length: 5
        },
        outputMap: {
            rsi: 'rsi5'
        }
    },{
        study: studies.Rsi,
        inputs: {
            length: 2
        },
        outputMap: {
            rsi: 'rsi2'
        }
    }
];

// Dynamically add prChannel studies.
var prChannelConfigurationOptions = {
    length: [100, 200, 250, 300],
    degree: [2, 3, 4],
    deviations: [1.618, 1.7, 1.9, 1.95, 2.0, 2.1]
};

// Dynamically add trendPrChannel studies.
var trendPrChannelConfigurationOptions = {
    length: [200, 300, 400, 450, 500, 550, 600, 650],
    degree: [2, 3, 4]
};

function Reversals(data, investment, profitability) {
    this.constructor = Reversals;
    Base.call(this, data, investment, profitability);

    // Prepare studies for use.
    this.prepareStudies(studyDefinitions);

    // Prepare all data in advance for use.
    this.prepareStudyData(data);
}

// Create a copy of the Base "class" prototype for use in this "class."
Reversals.prototype = Object.create(Base.prototype);

Reversals.prototype.prepareStudies = function(studyDefinitions) {
    // Generate additional study definitions.
    var prChannelStudyDefinitions = this.buildConfigurations(prChannelConfigurationOptions);
    var trendPrChannelStudyDefinitions = this.buildConfigurations(trendPrChannelConfigurationOptions);

    // Augment study definitions with additional, generated studied definitions.
    prChannelStudyDefinitions.forEach(function(studyDefinition) {
        studyDefinitions.push(studyDefinition);
    });
    trendPrChannelStudyDefinitions.forEach(function(studyDefinition) {
        studyDefinitions.push(studyDefinition);
    });

    // Now that all study definitions are prepared, prepare all studies.
    Base.prototype.prepareStudies.call(this, studyDefinitions);
};

Reversals.prototype.optimize = function() {
    var configurations = this.buildConfigurations();

    configurations.forEach(function(configuration) {
        // Instantiate a fresh strategy.
        var strategy = new strategies.configurable.Reversals();

        // Backtest the strategy using the current configuration and the pre-built data.
        var results = strategy.backtest(configuration, data, investment, profitability);

        // Record the results.
        Optimization.create(results);
    });
};

module.exports = Reversals;
