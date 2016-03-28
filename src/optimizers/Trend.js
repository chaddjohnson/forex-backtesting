var Base = require('./Base');
var studies = require('../studies');

var studyDefinitions = [
    // {study: studies.Rsi, inputs: {length: 14}, outputMap: {rsi: 'rsi14'}},
    // {study: studies.Rsi, inputs: {length: 2}, outputMap: {rsi: 'rsi2'}},
    {study: studies.Rsi, inputs: {length: 5}, outputMap: {rsi: 'rsi5'}},
    {study: studies.Rsi, inputs: {length: 7}, outputMap: {rsi: 'rsi7'}},
    // {study: studies.Rsi, inputs: {length: 9}, outputMap: {rsi: 'rsi9'}},
    {study: studies.AverageDirectionalIndex, inputs: {length: 14}, outputMap: {pDI: 'pDI', mDI: 'mDI', ADX: 'ADX'}}
];

var configurationOptions = {
    rsi: [
        null,
        // {rsi: 'rsi14', overbought: 70, oversold: 30},
        // {rsi: 'rsi2', overbought: 95, oversold: 5},
        {rsi: 'rsi5', overbought: 80, oversold: 20},
        // {rsi: 'rsi7', overbought: 77, oversold: 23},
        {rsi: 'rsi7', overbought: 80, oversold: 20},
        // {rsi: 'rsi9', overbought: 70, oversold: 30},
        // {rsi: 'rsi9', overbought: 77, oversold: 23}
    ],
    adx: [
        {pDI: 'pDI', mDI: 'mDI', ADX: 'ADX'}
    ]
};

function Trend(symbol) {
    this.constructor = Trend;
    Base.call(this, 'Trend', symbol);

    // Prepare studies for use.
    this.prepareStudies(studyDefinitions);

    // Prepare all optimization configurations.
    this.configurations = this.buildConfigurations(configurationOptions);
}

// Create a copy of the Base "class" prototype for use in this "class."
Trend.prototype = Object.create(Base.prototype);

Trend.prototype.optimize = function(data, investment, profitability, done) {
    var self = this;

    // Prepare all data in advance for use.
    self.prepareStudyData(data, function() {
        // Ensure memory is released.
        data = null;

        Base.prototype.optimize.call(self, self.configurations, investment, profitability, done);
    });
};

// Make study definitions publically and statically available.
Trend.studyDefinitions = studyDefinitions;

module.exports = Trend;
