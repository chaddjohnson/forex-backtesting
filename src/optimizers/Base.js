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
    DataPoint.count({symbol: self.symbol}, function(error, count) {
        if (error) {
            console.error(error.message || error);
        }
        if (count > 0) {
            process.stdout.write('using cached data\n');
            callback();
            return;
        }

        var cumulativeData = [];

        var prepareDataPoint = function(dataPoint, index, taskCallback) {
            var completedDataPoints = [];

            percentage = ((index / dataPointCount) * 100).toFixed(5);
            process.stdout.cursorTo(29);
            process.stdout.write(percentage + '%');

            // Add the data point (cloned) to the cumulative data.
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

            // Periodically free up memory.
            if (cumulativeData.length >= 2000) {
                // Remove a chunk of completed items from the cumulative data.
                completedDataPoints = cumulativeData.splice(0, 1000);

                // Save the data just removed prior to derefrencing it.
                self.saveDataPoints(completedDataPoints, function() {
                    // Explicitly mark each item as ready for garbage collection, and then the whole array.
                    completedDataPoints.forEach(function(item, index) {
                        completedDataPoints[index] = null;
                    });
                    completedDataPoints = null;

                    taskCallback();
                });
            }
            else {
                taskCallback();
            }
        };

        var tasks = [];

        data.forEach(function(dataPoint, index) {
            tasks.push(function(taskCallback) {
                prepareDataPoint(_.clone(dataPoint), index, taskCallback);
            });
        });
        async.series(tasks, function(error) {
            if (error) {
                console.log(error.message || error);
            }

            // Save the data just removed prior to derefrencing it.
            self.saveDataPoints(cumulativeData, function() {
                data = null;
                cumulativeData = null;
                self.studies = null;

                process.stdout.cursorTo(29);
                process.stdout.write((100).toFixed(5) + '%\n');

                // Done preparing study data.
                callback();
            });
        });
    });
};

Base.prototype.saveDataPoints = function(data, callback) {
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
            results.push(JSON.parse(JSON.stringify(current)));
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

Base.prototype.optimize = function(configurations, investment, profitability, callback) {
    var self = this;
    var strategies = [];
    var dataPointCount = 0;
    var tasks = [];

    process.stdout.write('Optimizing...');

    // Exclude configurations that have already been backtested.
    tasks.push(function(taskCallback) {
        self.removeCompletedConfigurations(configurations, taskCallback);
    });

    // Get a count of all data points.
    tasks.push(function(taskCallback) {
        DataPoint.count({symbol: self.symbol}, function(error, count) {
            dataPointCount = count;
            taskCallback(error);
        });
    });

    // Instantiate one strategy per configuration.
    tasks.push(function(taskCallback) {
        strategies = _(configurations).map(function(configuration) {
            return new self.strategyFn(configuration);
        });

        taskCallback();
    });

    // Use a stream to interate over the data in batches so as to not consume too much memory.
    tasks.push(function(taskCallback) {
        var tasks = [];
        var index = 0;

        var stream = DataPoint.find({symbol: self.symbol}).stream();

        // Iterate through the data.
        stream.on('data', function(dataPoint) {
            // Backtest each strategy against the current data point..
            strategies.forEach(function(strategy) {
                strategy.backtest(dataPoint, investment, profitability);
            });

            index++;

            process.stdout.cursorTo(13);
            process.stdout.write(index + ' of ' + dataPointCount + ' completed');
        });

        stream.on('close', taskCallback);
    });

    // Record the results for each strategy.
    tasks.push(function(taskCallback) {
        process.stdout.cursorTo(13);
        process.stdout.write(dataPointCount + ' of ' + dataPointCount + ' completed\n');
        process.stdout.write('Saving results...');

        var backtests = [];

        strategies.forEach(function(strategy) {
            var results = strategy.getResults();

            backtests.push({
                symbol: self.symbol,
                strategyName: strategy.constructor.name,
                configuration: strategy.getConfiguration(),
                profitLoss: results.profitLoss,
                winCount: results.winCount,
                loseCount: results.loseCount,
                tradeCount: results.winCount + results.loseCount,
                winRate: results.winRate,
                maximumConsecutiveLosses: results.maximumConsecutiveLosses,
                minimumProfitLoss: results.minimumProfitLoss
            });
        });

        Backtest.collection.insert(backtests, function(error) {
            process.stdout.write('done\n');
            taskCallback(error);
        });
    });

    // Run tasks.
    async.series(tasks, callback);
};

module.exports = Base;
