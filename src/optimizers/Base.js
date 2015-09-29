var _ = require('underscore');
var async = require('async');
var Optimization = require('../models/optimization');

function Base(strategyFn, symbol) {
    this.strategyFn = strategyFn;
    this.symbol = symbol;
    this.studies = [];
    this.cumulativeData = [];
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
    var self = this;
    var previousDataPoint;

    // For every data point...
    data.forEach(function(dataPoint) {
        previousDataPoint = self.cumulativeData[self.cumulativeData.length - 1];

        // Add the data point to the cumulative data.
        self.cumulativeData.push(dataPoint);

        // Iterate over each study...
        self.studies.forEach(function(study) {
            var studyProperty = '';
            var studyTickValue = 0.0;
            var studyOutputs = study.getOutputMappings();

            // Update the data for the strategy.
            study.setData(self.cumulativeData);

            studyTickValues = study.tick();

            // Augment the last data point with the data the study generates.
            for (studyProperty in studyOutputs) {
                if (studyTickValues && typeof studyTickValues[studyOutputs[studyProperty]] === 'number') {
                    // Include output in main output, and limit decimal precision without rounding.
                    dataPoint[studyOutputs[studyProperty]] = studyTickValues[studyOutputs[studyProperty]];
                }
                else {
                    dataPoint[studyOutputs[studyProperty]] = '';
                }
            }
        });
    });
};

Base.prototype.buildConfigurations = function(options, optionIndex, results, current) {
    optionIndex = optionIndex || 0;
    results = results || [];
    current = current || {};

    var allKeys = Object.keys(options);
    var optionKey = allKeys[optionIndex];
    var vals = options[optionKey];
    var i = 0;

    for (i = 0; i < vals.length; i++) {
        current[optionKey] = vals[i];

        if (optionIndex + 1 < allKeys.length) {
            this.buildConfigurations(options, optionIndex + 1, results, current);
        }
        else {
            results.push(_(current).clone());
        }
    }

    return results;
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
