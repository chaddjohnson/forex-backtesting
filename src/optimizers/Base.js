var async = require('async');
var Optimization = require('./models/optimization');

function Base(strategyFn, symbol) {
    this.strategyFn = strategyFn;
    this.symbol = symbol;
    this.studies = [];
}

Base.prototype.prepareStudies = function(studyDefinitions) {
    var self = this;

    // Iterate over each study definition...
    studyDefinitions.forEach(function(studyDefinition) {
        // Instantiate the study, and add it to the list of studies for this strategy.
        self.studies.push(new studyDefinition.study(studyDefinition.inputs, studyDefinition.outputMap));
    });
};


Base.prototype.prepareStudyData = function(data) {
    // TODO
};

Base.prototype.buildConfigurations = function(combinations) {
    var configurations = [];
    var configuration = {};
    var optionKeys = Object.keys(options);
    var optionKeysCount = optionKeys.length;
    var optionValues;
    var optionValuesCount = optionValues.length;
    var innerOptions;
    var innerOptionsCount = innerOptions.length;
    var i = 0;
    var j = 0;
    var k = 0;
    var l = 0;

    // Iterate over the options.
    for (i = 0; i < optionKeysCount; i++) {
        // Get the values for the current option.
        optionValues = options[optionKeys[i]];

        // Iterate through values for the current configuration...
        for (j = 0; j < optionValuesCount; j++) {
            // Iterate through the options after the current one.
            for (k = i; k < ; k++) {
                innerOptions = ...;  // TODO

                for (l = 0; l < innerOptionsCount; l++) {
                    configuration = {};
                    configuration[optionKeys[i]] = optionValues[j];
                    configuration[...] = ...;  // TODO

                    configurations.push(configuration);
                }
            }
        }
    }

    return configurations;
};

Base.prototype.optimize = function(configurations, data, investment, profitability) {
    var self = this;

    async.each(configurations, function(configuration, callback) {
        // Instantiate a fresh strategy.
        var strategy = new self.strategyFn();

        // Backtest the strategy using the current configuration and the pre-built data.
        var results = strategy.backtest(configuration, data, investment, profitability);

        // Record the results.
        Optimization.create({
            symbol: self.symbol,
            strategyName: strategy.constructor.name,
            configuration: configuration,
            profitLoss: results.profitLoss,
            winCount: results.winCount,
            loseCount: results.loseCount,
            tradeCount: results.winCount + results.loseCount,
            winRate: results.winRate,
            maximumConsecutiveLosses: results.maximumConsecutiveLosses,
            minimumProfitLoss: results.minimumProfitLoss
        }, callback);
    });
};

module.exports = Base;
