var _ = require('underscore');
var async = require('async');
var Backtest = require('../models/Backtest');
var DataPoint = require('../models/DataPoint');

function Base(strategyFn, symbol) {
    this.strategyFn = strategyFn;
    this.symbol = symbol;
    this.studies = [];
}

Base.prototype.prepareStudies = function(studyDefinitions) {
    var self = this;

    // Iterate over each study definition...
    process.stdout.write('Preparing studies...');
    studyDefinitions.forEach(function(studyDefinition) {
        // Instantiate the study, and add it to the list of studies for this strategy.
        self.studies.push(new studyDefinition.study(studyDefinition.inputs, studyDefinition.outputMap));
    });
    process.stdout.write('done\n');
};

Base.prototype.prepareStudyData = function(data, callback) {
    var self = this;
    var percentage = 0.0;
    var dataPointCount = data.length;

    process.stdout.write('Preparing data for studies...');

    // Find cached data points, if any.
    DataPoint.find({symbol: this.symbol}, function(error, dataPoints) {
        var cumulativeData = [];

        if (error) {
            console.error(error.message || error);
        }
        if (dataPoints.length) {
            process.stdout.write('using cached data\n');

            cumulativeData = _(dataPoints).map(function(dataPoint) {
                return dataPoint.data;
            });
            callback(cumulativeData);

            return;
        }

        // For every data point...
        data.forEach(function(dataPoint, index) {
            percentage = ((index / dataPointCount) * 100).toFixed(5);
            process.stdout.cursorTo(29);
            process.stdout.write(percentage + '%');

            // Add the data point to the cumulative data.
            cumulativeData.push(dataPoint);

            // Iterate over each study...
            self.studies.forEach(function(study) {
                var studyProperty = '';
                var studyTickValue = 0.0;
                var studyOutputs = study.getOutputMappings();

                // Update the data for the study.
                study.setData(cumulativeData);

                studyTickValues = study.tick();

                // Augment the last data point with the data the study generates.
                for (studyProperty in studyOutputs) {
                    if (studyTickValues && typeof studyTickValues[studyOutputs[studyProperty]] === 'number') {
                        dataPoint[studyOutputs[studyProperty]] = studyTickValues[studyOutputs[studyProperty]];
                    }
                    else {
                        dataPoint[studyOutputs[studyProperty]] = '';
                    }
                }

                // Ensure memory is freed.
                studyTickValues = null;
            });
        });

        process.stdout.cursorTo(29);
        process.stdout.write((100).toFixed(5) + '%\n');

        // Cache the data.
        process.stdout.write('Caching data...');
        self.cacheData(cumulativeData, function() {
            process.stdout.write('done\n');
            callback(cumulativeData);
        });
    });
};

Base.prototype.cacheData = function(data, callback) {
    var self = this;
    var dataPoints = _(data).map(function(dataPoint) {
        return {
            symbol: self.symbol,
            data: dataPoint
        }
    });
    DataPoint.collection.insert(dataPoints, function() {
        dataPoints = null;
        callback();
    });
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

Base.prototype.removeCompletedConfigurations = function(configurations, callback) {
    Backtest.find({symbol: this.symbol}, function(error, backtests) {
        if (error) {
            console.error(error.message || error);
        }

        // Get the configurations for completed backtests.
        var completedConfigurations = _(backtests).map(function(backtest) {
            return backtest.configuration;
        });

        // Iterate of configurations, and remove ones already used in completed backtests.
        completedConfigurations.forEach(function(completedConfiguration) {
            var foundIndex = -1;

            _(configurations).find(function(configuration, index) {
                var found = _(configuration).isEqual(completedConfiguration);
                if (found) {
                    foundIndex = index;
                }
                return found;
            });

            if (foundIndex > -1) {
                // The configuration was already backtested, so remove it from the list
                // of configurations to backtest.
                configurations.splice(foundIndex, 1);
            }
        });

        callback();
    });
};

Base.prototype.optimize = function(configurations, data, investment, profitability, callback) {
    var self = this;

    process.stdout.write('Optimizing...');

    // Exclude configurations that have already been backtested.
    self.removeCompletedConfigurations(configurations, function() {
        var configurationCompletionCount = -1;
        var configurationsCount = configurations.length;

        async.each(configurations, function(configuration, asyncCallback) {
            configurationCompletionCount++;
            process.stdout.cursorTo(13);
            process.stdout.write(configurationCompletionCount + ' of ' + configurationsCount + ' completed');

            // Instantiate a fresh strategy.
            var strategy = new self.strategyFn();

            // Backtest the strategy using the current configuration and the pre-built data.
            var results = strategy.backtest(configuration, data, investment, profitability);

            // Record the results.
            var backtest = {
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
            };
            Backtest.collection.insert(backtest, function(error) {
                // Ensure memory is freed.
                strategy = null;
                results = null;
                backtest = null;

                asyncCallback(error);
            });
        }, function(error) {
            if (error) {
                console.log(error.message || error);
            }
            process.stdout.cursorTo(13);
            process.stdout.write(configurationsCount + ' of ' + configurationsCount + ' completed\n');
            callback();
        });
    });
};

module.exports = Base;
