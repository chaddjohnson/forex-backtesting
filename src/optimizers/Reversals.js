var Base = require('./Base');
var studies = require('../studies');

var studyDefinitions = [
    {study: studies.Ema, inputs: {length: 200}, outputMap: {ema: 'ema200'}},
    {study: studies.Ema, inputs: {length: 100}, outputMap: {ema: 'ema100'}},
    {study: studies.Ema, inputs: {length: 50}, outputMap: {ema: 'ema50'}},
    {study: studies.Sma, inputs: {length: 13}, outputMap: {sma: 'sma13'}},
    {study: studies.Rsi, inputs: {length: 7}, outputMap: {rsi: 'rsi7'}},
    {study: studies.Rsi, inputs: {length: 5}, outputMap: {rsi: 'rsi5'}},
    {study: studies.Rsi, inputs: {length: 2}, outputMap: {rsi: 'rsi2'}},
    {study: studies.BollingerBands, inputs: {length: 20, deviations: 3.0}, outputMap: {middle: 'bollingerBandMiddle', upper: 'bollingerBandUpper', lower: 'bollingerBandLower'}}
];

var configurationOptions = {
    ema200: [true, false],
    ema100: [true, false],
    ema50: [true, false],
    sma13: [true, false],
    rsi: [
        null,
        {rsi: 'rsi7', overbought: 77, oversold: 23},
        {rsi: 'rsi7', overbought: 80, oversold: 20},
        {rsi: 'rsi5', overbought: 80, oversold: 20},
        {rsi: 'rsi2', overbought: 95, oversold: 5}
    ],
    bollingerBands: [
        {upper: 'bollingerBandUpper', lower: 'bollingerBandLower'}
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

module.exports = Reversals;
