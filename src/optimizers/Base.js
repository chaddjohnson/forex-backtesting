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
    process.stdout.write('Preparing studies...\n');
    studyDefinitions.forEach(function(studyDefinition) {
        // Instantiate the study, and add it to the list of studies for this strategy.
        process.stdout.write('    ' + studyDefinition.study.name + '...');
        self.studies.push(new studyDefinition.study(studyDefinition.inputs, studyDefinition.outputMap));
        process.stdout.write('done\n');
    });
};


Base.prototype.prepareStudyData = function(data) {
    var self = this;
    var progress = 0.0;
    var dataPointCount = data.length;

    // For every data point...
    process.stdout.write('Preparing data for studies...');
    data.forEach(function(dataPoint, index) {
        percentage = ((index / dataPointCount) * 100).toFixed(5);
        process.stdout.cursorTo(29);
        process.stdout.write(percentage + '%');

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
    process.stdout.cursorTo(29);
    process.stdout.write((100).toFixed(5) + '%\n');

    return self.cumulativeData;
};

Base.prototype.buildConfigurations = function(options, optionIndex, results, current) {
    if (!optionIndex) {
        process.stdout.write('Building configurations...');
    }

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
            results.push(_.clone(current));
        }
    }

    if (!optionIndex) {
        process.stdout.write('done\n');
    }

    return results;
};

Base.prototype.optimize = function(configurations, data, investment, profitability, done) {
    var self = this;
    var progress = 0.0;
    var configurationsCount = configurations.length;

    process.stdout.write('Optimizing...');
    async.forEachOf(configurations, function(configuration, index, callback) {
        var percentage = ((index / configurationsCount) * 100).toFixed(5);
        process.stdout.cursorTo(13);
        process.stdout.write(percentage + '%');

        // Instantiate a fresh strategy.
        var strategy = new self.strategyFn();

        // Backtest the strategy using the current configuration and the pre-built data.
        var results = strategy.backtest(configuration, data, investment, profitability);

        // Record the results.
        var optimization = new Optimization({
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
        });
        optimization.save(function(error) {
            callback(error);
        });
    }, function(error) {
        if (error) {
            console.log(error.message || error);
        }
        done();
    });
    process.stdout.cursorTo(13);
    process.stdout.write((100).toFixed(5) + '%\n');
};

module.exports = Base;
