var _ = require('lodash');
var async = require('async');
var forkFn = require('child_process').fork;
var Backtest = require('../models/Backtest');
var Forwardtest = require('../models/Forwardtest');
var DataPoint = require('../models/DataPoint');
var strategyFns = require('../strategies');

require('events').EventEmitter.defaultMaxListeners = Infinity;

function Base(strategyName, symbol, group) {
    this.strategyName = strategyName;
    this.symbol = symbol;
    this.group = group;
    this.studies = [];
    this.query = {symbol: this.symbol};
}

Base.prototype.setQuery = function(query) {
    this.query = query;
};

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
    DataPoint.count(self.query, function(error, count) {
        if (error) {
            console.error(error.message || error);
        }
        if (count > 0) {
            process.stdout.write('using cached data\n');
            callback();
            return;
        }

        var cumulativeData = [];
        var previousDataPoint = null;

        var prepareDataPoint = function(dataPoint, index, taskCallback) {
            var completedDataPoints = [];

            percentage = ((index / dataPointCount) * 100).toFixed(5);
            process.stdout.cursorTo(29);
            process.stdout.write(percentage + '%');

            // If there is a significant gap, save the current data points, and start over with recording.
            if (previousDataPoint && (dataPoint.timestamp - previousDataPoint.timestamp) > 60 * 1000) {
                self.saveDataPoints(cumulativeData.slice());
                cumulativeData = [];
            }

            // Add the data point (cloned) to the cumulative data.
            cumulativeData.push(dataPoint);

            // Iterate over each study...
            self.studies.forEach(function(study) {
                var studyProperty = '';
                var studyTickValues = {};
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

            previousDataPoint = dataPoint;

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
                    completedDataPoints = [];

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
                console.error(error.message || error);
            }

            // Save the data just removed prior to derefrencing it.
            self.saveDataPoints(cumulativeData, function() {
                data = [];
                cumulativeData = [];
                self.studies = [];

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
    callback = callback || function() {};

    if (!data || !data.length) {
        callback();
        return;
    }

    var dataPoints = _.map(data, function(dataPoint) {
        return {
            symbol: self.symbol,
            data: dataPoint
        }
    });
    DataPoint.collection.insert(dataPoints, function() {
        dataPoints = [];
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
        var completedConfigurations = _.map(backtests, function(backtest) {
            return backtest.configuration;
        });

        // Iterate of configurations, and remove ones already used in completed backtests.
        completedConfigurations.forEach(function(completedConfiguration) {
            var foundIndex = -1;

            _.find(configurations, function(configuration, index) {
                var found = _.isEqual(configuration, completedConfiguration);
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
    var forks = [];
    var cpuCoreCount = require('os').cpus().length;

    process.stdout.write('Optimizing...');

    // Create child processes for parallel processing.
    tasks.push(function(taskCallback) {
        var index = 0;

        for (index = 0; index < cpuCoreCount; index++) {
            forks.push(forkFn(__dirname + '/worker.js'));
        }

        taskCallback();
    });

    // Exclude configurations that have already been backtested.
    tasks.push(function(taskCallback) {
        self.removeCompletedConfigurations(configurations, taskCallback);
    });

    // Get a count of all data points.
    tasks.push(function(taskCallback) {
        DataPoint.count(self.query, function(error, count) {
            dataPointCount = count;
            taskCallback(error);
        });
    });

    // Assign configurations to forks.
    tasks.push(function(taskCallback) {
        configurations.forEach(function(configuration, index) {
            forks[index % cpuCoreCount].send({
                type: 'init',
                data: {
                    strategyName: self.strategyName,
                    symbol: self.symbol,
                    group: self.group,
                    configuration: configuration,
                    dataPointCount: dataPointCount
                }
            });
        });

        taskCallback();
    });

    // Use a stream to interate over the data in batches so as to not consume too much memory.
    tasks.push(function(taskCallback) {
        var index = 0;

        var streamer = function(dataPoint) {
            stream.pause();

            var completionCount = 0;

            forks.forEach(function(fork) {
                fork.send({
                    type: 'backtest',
                    data: {
                        dataPoint: dataPoint.data,
                        index: index,
                        investment: investment,
                        profitability: profitability
                    }
                });

                function handler(message) {
                    if (message.type !== 'done') {
                        return;
                    }

                    fork.removeListener('message', handler);

                    if (++completionCount === cpuCoreCount) {
                        index++;

                        process.stdout.cursorTo(13);
                        process.stdout.write(index + ' of ' + dataPointCount + ' completed');

                        if (index === dataPointCount) {
                            strategyFns.optimization[self.strategyName].saveExpiredPositionsPool(function() {
                                taskCallback();
                            });
                        }

                        stream.resume();
                    }
                }

                fork.on('message', handler);
            });
        };

        var stream = DataPoint.find(self.query, {}, {timeout: true}).sort({'data.timestamp': 1}).stream();

        stream.on('data', streamer);
    });

    // Record the results for each strategy.
    tasks.push(function(taskCallback) {
        process.stdout.cursorTo(13);
        process.stdout.write(dataPointCount + ' of ' + dataPointCount + ' completed\n');
        process.stdout.write('Saving results...');

        var backtests = [];
        var resultsCount = 0;

        forks.forEach(function(fork) {
            fork.send({type: 'results'});

            fork.on('message', function(message) {
                if (message.type !== 'results') {
                    return;
                }

                resultsCount++;
                backtests = backtests.concat(message.data);

                if (resultsCount === cpuCoreCount) {
                    Forwardtest.collection.insert(backtests, function(error) {
                        process.stdout.write('done\n');

                        taskCallback(error);
                    });
                }
            });
        });
    });

    // Run tasks.
    async.series(tasks, callback);
};

module.exports = Base;
