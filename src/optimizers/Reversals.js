var Base = require('./Base');
var studies = require('../studies');
var strategies = require('../strategies');

var studyDefinitions = [
    {study: studies.Ema, inputs: {length: 200}, outputMap: {ema: 'ema200'}},
    {study: studies.Ema, inputs: {length: 100}, outputMap: {ema: 'ema100'}},
    {study: studies.Ema, inputs: {length: 50}, outputMap: {ema: 'ema50'}},
    {study: studies.Sma, inputs: {length: 13}, outputMap: {sma: 'sma13'}},
    {study: studies.Rsi, inputs: {length: 5}, outputMap: {rsi: 'rsi5'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 250, degree: 3, deviations: 1.9}, outputMap: {regression: 'prChannel250_3_19', upper: 'prChannelUpper250_3_19', lower: 'prChannelLower250_3_19'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 600, degree: 2}, outputMap: {regression: 'trendPrChannel600_2'}}
];

var configurationOptions = {
    ema200: [false],
    ema100: [true],
    ema50: [true],
    sma13: [true],
    ema13: [false],
    rsi: [
        // null,
        {rsi: 'rsi5', overbought: 80, oversold: 20}
    ],
    prChannel: [
        // null,
        {upper: 'prChannelUpper250_3_19', lower: 'prChannelLower250_3_19'}
    ],
    trendPrChannel: [
        // null,
        {regression: 'trendPrChannel600_2'}
    ]
};

function Reversals(symbol) {
    this.constructor = Reversals;
    Base.call(this, strategies.optimization.Reversals, symbol);

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
    self.prepareStudyData(data, function(preparedData) {
        // Ensure memory is released.
        data = null;

        Base.prototype.optimize.call(self, self.configurations, preparedData, investment, profitability, done);
    });
};

module.exports = Reversals;
