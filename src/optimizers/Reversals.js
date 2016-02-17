var Base = require('./Base');
var studies = require('../studies');

var studyDefinitions = [
    {study: studies.Sma, inputs: {length: 13}, outputMap: {sma: 'sma13'}},
    {study: studies.Ema, inputs: {length: 50}, outputMap: {ema: 'ema50'}},
    {study: studies.Ema, inputs: {length: 100}, outputMap: {ema: 'ema100'}},
    {study: studies.Ema, inputs: {length: 200}, outputMap: {ema: 'ema200'}},
    {study: studies.Rsi, inputs: {length: 14}, outputMap: {rsi: 'rsi14'}},
    {study: studies.Rsi, inputs: {length: 2}, outputMap: {rsi: 'rsi2'}},
    {study: studies.Rsi, inputs: {length: 5}, outputMap: {rsi: 'rsi5'}},
    {study: studies.Rsi, inputs: {length: 7}, outputMap: {rsi: 'rsi7'}},
    {study: studies.Rsi, inputs: {length: 9}, outputMap: {rsi: 'rsi9'}},
    {study: studies.StochasticOscillator, inputs: {length: 10, averageLength: 3}, outputMap: {K: 'stochastic10K', D: 'stochastic10D'}},
    {study: studies.StochasticOscillator, inputs: {length: 14, averageLength: 3}, outputMap: {K: 'stochastic14K', D: 'stochastic14D'}},
    {study: studies.StochasticOscillator, inputs: {length: 21, averageLength: 3}, outputMap: {K: 'stochastic21K', D: 'stochastic21D'}},
    {study: studies.StochasticOscillator, inputs: {length: 5, averageLength: 3}, outputMap: {K: 'stochastic5K', D: 'stochastic5D'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 150, degree: 2, deviations: 1.85}, outputMap: {regression: 'prChannel150_2_185', upper: 'prChannelUpper150_2_185', lower: 'prChannelLower150_2_185'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 150, degree: 2, deviations: 1.95}, outputMap: {regression: 'prChannel150_2_195', upper: 'prChannelUpper150_2_195', lower: 'prChannelLower150_2_195'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 150, degree: 2, deviations: 1.9}, outputMap: {regression: 'prChannel150_2_19', upper: 'prChannelUpper150_2_19', lower: 'prChannelLower150_2_19'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 150, degree: 2, deviations: 2.05}, outputMap: {regression: 'prChannel150_2_205', upper: 'prChannelUpper150_2_205', lower: 'prChannelLower150_2_205'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 150, degree: 2, deviations: 2.0}, outputMap: {regression: 'prChannel150_2_20', upper: 'prChannelUpper150_2_20', lower: 'prChannelLower150_2_20'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 150, degree: 2, deviations: 2.15}, outputMap: {regression: 'prChannel150_2_215', upper: 'prChannelUpper150_2_215', lower: 'prChannelLower150_2_215'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 150, degree: 2, deviations: 2.1}, outputMap: {regression: 'prChannel150_2_21', upper: 'prChannelUpper150_2_21', lower: 'prChannelLower150_2_21'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 150, degree: 3, deviations: 1.85}, outputMap: {regression: 'prChannel150_3_185', upper: 'prChannelUpper150_3_185', lower: 'prChannelLower150_3_185'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 150, degree: 3, deviations: 1.95}, outputMap: {regression: 'prChannel150_3_195', upper: 'prChannelUpper150_3_195', lower: 'prChannelLower150_3_195'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 150, degree: 3, deviations: 1.9}, outputMap: {regression: 'prChannel150_3_19', upper: 'prChannelUpper150_3_19', lower: 'prChannelLower150_3_19'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 150, degree: 3, deviations: 2.05}, outputMap: {regression: 'prChannel150_3_205', upper: 'prChannelUpper150_3_205', lower: 'prChannelLower150_3_205'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 150, degree: 3, deviations: 2.0}, outputMap: {regression: 'prChannel150_3_20', upper: 'prChannelUpper150_3_20', lower: 'prChannelLower150_3_20'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 150, degree: 3, deviations: 2.15}, outputMap: {regression: 'prChannel150_3_215', upper: 'prChannelUpper150_3_215', lower: 'prChannelLower150_3_215'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 150, degree: 3, deviations: 2.1}, outputMap: {regression: 'prChannel150_3_21', upper: 'prChannelUpper150_3_21', lower: 'prChannelLower150_3_21'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 150, degree: 4, deviations: 1.85}, outputMap: {regression: 'prChannel150_4_185', upper: 'prChannelUpper150_4_185', lower: 'prChannelLower150_4_185'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 150, degree: 4, deviations: 1.95}, outputMap: {regression: 'prChannel150_4_195', upper: 'prChannelUpper150_4_195', lower: 'prChannelLower150_4_195'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 150, degree: 4, deviations: 1.9}, outputMap: {regression: 'prChannel150_4_19', upper: 'prChannelUpper150_4_19', lower: 'prChannelLower150_4_19'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 150, degree: 4, deviations: 2.05}, outputMap: {regression: 'prChannel150_4_205', upper: 'prChannelUpper150_4_205', lower: 'prChannelLower150_4_205'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 150, degree: 4, deviations: 2.0}, outputMap: {regression: 'prChannel150_4_20', upper: 'prChannelUpper150_4_20', lower: 'prChannelLower150_4_20'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 150, degree: 4, deviations: 2.15}, outputMap: {regression: 'prChannel150_4_215', upper: 'prChannelUpper150_4_215', lower: 'prChannelLower150_4_215'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 150, degree: 4, deviations: 2.1}, outputMap: {regression: 'prChannel150_4_21', upper: 'prChannelUpper150_4_21', lower: 'prChannelLower150_4_21'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 150, degree: 5, deviations: 1.85}, outputMap: {regression: 'prChannel150_5_185', upper: 'prChannelUpper150_5_185', lower: 'prChannelLower150_5_185'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 150, degree: 5, deviations: 2.05}, outputMap: {regression: 'prChannel150_5_205', upper: 'prChannelUpper150_5_205', lower: 'prChannelLower150_5_205'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 200, degree: 2, deviations: 1.85}, outputMap: {regression: 'prChannel200_2_185', upper: 'prChannelUpper200_2_185', lower: 'prChannelLower200_2_185'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 200, degree: 2, deviations: 1.95}, outputMap: {regression: 'prChannel200_2_195', upper: 'prChannelUpper200_2_195', lower: 'prChannelLower200_2_195'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 200, degree: 2, deviations: 1.9}, outputMap: {regression: 'prChannel200_2_19', upper: 'prChannelUpper200_2_19', lower: 'prChannelLower200_2_19'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 200, degree: 2, deviations: 2.05}, outputMap: {regression: 'prChannel200_2_205', upper: 'prChannelUpper200_2_205', lower: 'prChannelLower200_2_205'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 200, degree: 2, deviations: 2.0}, outputMap: {regression: 'prChannel200_2_20', upper: 'prChannelUpper200_2_20', lower: 'prChannelLower200_2_20'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 200, degree: 2, deviations: 2.15}, outputMap: {regression: 'prChannel200_2_215', upper: 'prChannelUpper200_2_215', lower: 'prChannelLower200_2_215'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 200, degree: 2, deviations: 2.1}, outputMap: {regression: 'prChannel200_2_21', upper: 'prChannelUpper200_2_21', lower: 'prChannelLower200_2_21'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 200, degree: 3, deviations: 1.85}, outputMap: {regression: 'prChannel200_3_185', upper: 'prChannelUpper200_3_185', lower: 'prChannelLower200_3_185'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 200, degree: 3, deviations: 1.95}, outputMap: {regression: 'prChannel200_3_195', upper: 'prChannelUpper200_3_195', lower: 'prChannelLower200_3_195'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 200, degree: 3, deviations: 1.9}, outputMap: {regression: 'prChannel200_3_19', upper: 'prChannelUpper200_3_19', lower: 'prChannelLower200_3_19'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 200, degree: 3, deviations: 2.05}, outputMap: {regression: 'prChannel200_3_205', upper: 'prChannelUpper200_3_205', lower: 'prChannelLower200_3_205'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 200, degree: 3, deviations: 2.0}, outputMap: {regression: 'prChannel200_3_20', upper: 'prChannelUpper200_3_20', lower: 'prChannelLower200_3_20'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 200, degree: 3, deviations: 2.15}, outputMap: {regression: 'prChannel200_3_215', upper: 'prChannelUpper200_3_215', lower: 'prChannelLower200_3_215'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 200, degree: 3, deviations: 2.1}, outputMap: {regression: 'prChannel200_3_21', upper: 'prChannelUpper200_3_21', lower: 'prChannelLower200_3_21'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 200, degree: 4, deviations: 1.85}, outputMap: {regression: 'prChannel200_4_185', upper: 'prChannelUpper200_4_185', lower: 'prChannelLower200_4_185'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 200, degree: 4, deviations: 1.95}, outputMap: {regression: 'prChannel200_4_195', upper: 'prChannelUpper200_4_195', lower: 'prChannelLower200_4_195'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 200, degree: 4, deviations: 1.9}, outputMap: {regression: 'prChannel200_4_19', upper: 'prChannelUpper200_4_19', lower: 'prChannelLower200_4_19'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 200, degree: 4, deviations: 2.05}, outputMap: {regression: 'prChannel200_4_205', upper: 'prChannelUpper200_4_205', lower: 'prChannelLower200_4_205'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 200, degree: 4, deviations: 2.0}, outputMap: {regression: 'prChannel200_4_20', upper: 'prChannelUpper200_4_20', lower: 'prChannelLower200_4_20'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 200, degree: 4, deviations: 2.15}, outputMap: {regression: 'prChannel200_4_215', upper: 'prChannelUpper200_4_215', lower: 'prChannelLower200_4_215'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 200, degree: 4, deviations: 2.1}, outputMap: {regression: 'prChannel200_4_21', upper: 'prChannelUpper200_4_21', lower: 'prChannelLower200_4_21'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 200, degree: 5, deviations: 1.85}, outputMap: {regression: 'prChannel200_5_185', upper: 'prChannelUpper200_5_185', lower: 'prChannelLower200_5_185'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 200, degree: 5, deviations: 2.05}, outputMap: {regression: 'prChannel200_5_205', upper: 'prChannelUpper200_5_205', lower: 'prChannelLower200_5_205'}}
];

var configurationOptions = {
    ema200: [true, false],
    ema100: [true, false],
    ema50: [true, false],
    sma13: [true, false],
    rsi: [
        null,
        {rsi: 'rsi14', overbought: 70, oversold: 30},
        {rsi: 'rsi2', overbought: 95, oversold: 5},
        {rsi: 'rsi5', overbought: 80, oversold: 20},
        {rsi: 'rsi7', overbought: 77, oversold: 23},
        {rsi: 'rsi7', overbought: 80, oversold: 20},
        {rsi: 'rsi9', overbought: 70, oversold: 30},
        {rsi: 'rsi9', overbought: 77, oversold: 23}

    ],
    stochastic: [
        null,
        {K: 'stochastic10K', D: 'stochastic10D', overbought: 77, oversold: 23},
        {K: 'stochastic10K', D: 'stochastic10D', overbought: 80, oversold: 20},
        {K: 'stochastic14K', D: 'stochastic14D', overbought: 70, oversold: 30},
        {K: 'stochastic14K', D: 'stochastic14D', overbought: 77, oversold: 23},
        {K: 'stochastic21K', D: 'stochastic21D', overbought: 70, oversold: 30},
        {K: 'stochastic21K', D: 'stochastic21D', overbought: 77, oversold: 23},
        {K: 'stochastic5K', D: 'stochastic5D', overbought: 77, oversold: 23},
        {K: 'stochastic5K', D: 'stochastic5D', overbought: 80, oversold: 20},
        {K: 'stochastic5K', D: 'stochastic5D', overbought: 95, oversold: 5}
    ],
    prChannel: [
        null,
        {upper: 'prChannelUpper150_2_185', lower: 'prChannelLower150_2_185'},
        {upper: 'prChannelUpper150_2_19', lower: 'prChannelLower150_2_19'},
        {upper: 'prChannelUpper150_2_195', lower: 'prChannelLower150_2_195'},
        {upper: 'prChannelUpper150_2_20', lower: 'prChannelLower150_2_20'},
        {upper: 'prChannelUpper150_2_205', lower: 'prChannelLower150_2_205'},
        {upper: 'prChannelUpper150_2_21', lower: 'prChannelLower150_2_21'},
        {upper: 'prChannelUpper150_2_215', lower: 'prChannelLower150_2_215'},
        {upper: 'prChannelUpper150_3_185', lower: 'prChannelLower150_3_185'},
        {upper: 'prChannelUpper150_3_19', lower: 'prChannelLower150_3_19'},
        {upper: 'prChannelUpper150_3_195', lower: 'prChannelLower150_3_195'},
        {upper: 'prChannelUpper150_3_20', lower: 'prChannelLower150_3_20'},
        {upper: 'prChannelUpper150_3_205', lower: 'prChannelLower150_3_205'},
        {upper: 'prChannelUpper150_3_21', lower: 'prChannelLower150_3_21'},
        {upper: 'prChannelUpper150_3_215', lower: 'prChannelLower150_3_215'},
        {upper: 'prChannelUpper150_4_185', lower: 'prChannelLower150_4_185'},
        {upper: 'prChannelUpper150_4_19', lower: 'prChannelLower150_4_19'},
        {upper: 'prChannelUpper150_4_195', lower: 'prChannelLower150_4_195'},
        {upper: 'prChannelUpper150_4_20', lower: 'prChannelLower150_4_20'},
        {upper: 'prChannelUpper150_4_205', lower: 'prChannelLower150_4_205'},
        {upper: 'prChannelUpper150_4_21', lower: 'prChannelLower150_4_21'},
        {upper: 'prChannelUpper150_4_215', lower: 'prChannelLower150_4_215'},
        {upper: 'prChannelUpper150_5_185', lower: 'prChannelLower150_5_185'},
        {upper: 'prChannelUpper150_5_205', lower: 'prChannelLower150_5_205'},
        {upper: 'prChannelUpper200_2_185', lower: 'prChannelLower200_2_185'},
        {upper: 'prChannelUpper200_2_19', lower: 'prChannelLower200_2_19'},
        {upper: 'prChannelUpper200_2_195', lower: 'prChannelLower200_2_195'},
        {upper: 'prChannelUpper200_2_20', lower: 'prChannelLower200_2_20'},
        {upper: 'prChannelUpper200_2_205', lower: 'prChannelLower200_2_205'},
        {upper: 'prChannelUpper200_2_21', lower: 'prChannelLower200_2_21'},
        {upper: 'prChannelUpper200_2_215', lower: 'prChannelLower200_2_215'},
        {upper: 'prChannelUpper200_3_185', lower: 'prChannelLower200_3_185'},
        {upper: 'prChannelUpper200_3_19', lower: 'prChannelLower200_3_19'},
        {upper: 'prChannelUpper200_3_195', lower: 'prChannelLower200_3_195'},
        {upper: 'prChannelUpper200_3_20', lower: 'prChannelLower200_3_20'},
        {upper: 'prChannelUpper200_3_205', lower: 'prChannelLower200_3_205'},
        {upper: 'prChannelUpper200_3_21', lower: 'prChannelLower200_3_21'},
        {upper: 'prChannelUpper200_3_215', lower: 'prChannelLower200_3_215'},
        {upper: 'prChannelUpper200_4_185', lower: 'prChannelLower200_4_185'},
        {upper: 'prChannelUpper200_4_19', lower: 'prChannelLower200_4_19'},
        {upper: 'prChannelUpper200_4_195', lower: 'prChannelLower200_4_195'},
        {upper: 'prChannelUpper200_4_20', lower: 'prChannelLower200_4_20'},
        {upper: 'prChannelUpper200_4_205', lower: 'prChannelLower200_4_205'},
        {upper: 'prChannelUpper200_4_21', lower: 'prChannelLower200_4_21'},
        {upper: 'prChannelUpper200_4_215', lower: 'prChannelLower200_4_215'},
        {upper: 'prChannelUpper200_5_185', lower: 'prChannelLower200_5_185'},
        {upper: 'prChannelUpper200_5_205', lower: 'prChannelLower200_5_205'}
    ]
};

function Reversals(symbol) {
    this.constructor = Reversals;
    Base.call(this, 'Reversals', symbol);

    // Prepare studies for use.
    this.prepareStudies(studyDefinitions);

    // Prepare all optimization configurations.
    this.configurations = this.buildConfigurations(configurationOptions);
}

// Create a copy of the Base "class" prototype for use in this "class."
Reversals.prototype = Object.create(Base.prototype);

Reversals.prototype.optimize = function(data, investment, profitability, done) {
    var self = this;

    // Prepare all data in advance for use.
    self.prepareStudyData(data, function() {
        // Ensure memory is released.
        data = null;

        Base.prototype.optimize.call(self, self.configurations, investment, profitability, done);
    });
};

// Make study definitions publically and statically available.
Reversals.studyDefinitions = studyDefinitions;

module.exports = Reversals;
