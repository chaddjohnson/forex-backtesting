var gulp = require('gulp');
var gutil = require('gulp-util');
var argv = require('yargs').argv;
var path = require('path');
var _ = require('lodash');
var async = require('async');
var slice = require('sliced');

var garbageCollectionTimeout = null;

function scheduleGarbageCollection() {
    if (!global.gc) {
        return;
    }
    garbageCollectionTimeout = setTimeout(function() {
        // Allow the timeout to be garbage collected.
        garbageCollectionTimeout = null;

        // Collect garbage.
        global.gc();

        // Re-schedule garbage collection.
        scheduleGarbageCollection();
    }, 1 * 60 * 1000);
}

// Replace slice() with a more efficient version.
Array.prototype.slice = function(begin, end) {
    return slice(this, begin, end);
};

scheduleGarbageCollection();

gulp.task('backtest', function(done) {
    function showUsageInfo() {
        console.log('Example usage:\n');
        console.log('gulp backtest --symbol AUDJPY --parser metatrader --data ./data/metatrader/three-year/AUDJPY.csv --optimizer Reversals --investment 1000 --profitability 0.7 --database forex-backtesting\n');
    }

    function handleInputError(message) {
        gutil.log(gutil.colors.red(message));
        showUsageInfo();
        process.exit(1);
    }

    var db = require('./db');
    var dataParsers = require('./src/dataParsers');
    var optimizers = require('./src/optimizers');

    var optimizerFn;
    var dataParser;
    var investment = 0.0;
    var profitability = 0.0;

    // Find the symbol based on the command line argument.
    if (!argv.symbol) {
        handleInputError('No symbol provided');
    }

    // Find the raw data parser based on command line argument.
    dataParser = dataParsers[argv.parser]
    if (!dataParser) {
        handleInputError('Invalid data parser');
    }

    // Find the strategy based on the command line argument.
    optimizerFn = optimizers[argv.optimizer]
    if (!optimizerFn) {
        handleInputError('Invalid strategy optimizer');
    }

    investment = parseFloat(argv.investment)
    if (!investment) {
        handleInputError('Invalid investment');
    }

    profitability = parseFloat(argv.profitability)
    if (!profitability) {
        handleInputError('No profitability provided');
    }

    if (!argv.database) {
        handleInputError('No database provided');
    }

    // Set up database connection.
    db.initialize(argv.database);

    try {
        // Parse the raw data file.
        dataParser.parse(argv.data).then(function(parsedData) {
            // Prepare the strategy.
            var optimizer = new optimizerFn(argv.symbol);

            // Backtest the strategy against the parsed data.
            optimizer.optimize(parsedData, investment, profitability, function() {
                db.disconnect();
                done();
            });
        });
    }
    catch (error) {
        console.error(error.message || error);
        process.exit(1);
    }
});

gulp.task('forwardtest', function(done) {
    function showUsageInfo() {
        console.log('Example usage:\n');
        console.log('gulp forwardtest --symbol AUDJPY --parser ctoption --data ./data/ctoption/AUDJPY.csv --investment 1000 --profitability 0.7 --database forex-backtesting\n');
    }

    function handleInputError(message) {
        gutil.log(gutil.colors.red(message));
        showUsageInfo();
        process.exit(1);
    }

    var db = require('./db');
    var dataParsers = require('./src/dataParsers');
    var Backtest = require('./src/models/Backtest');
    var Forwardtest = require('./src/models/Forwardtest');
    var optimizerFn = require('./src/optimizers/Reversals');
    var strategyFn = require('./src/strategies/combined/Reversals');
    var dataParser;
    var investment = 0.0;
    var profitability = 0.0;

    var backtestConstraints = {
        symbol: argv.symbol,
        //strategyName: argv.strategy,
        minimumProfitLoss: {'$gte': -20000},
        maximumConsecutiveLosses: {'$lte': 20},
        winRate: {'$gte': 0.62},
        tradeCount: {'$gte': 3000},
    };

    // Find the symbol based on the command line argument.
    if (!argv.symbol) {
        handleInputError('No symbol provided');
    }

    // Find the raw data parser based on command line argument.
    dataParser = dataParsers[argv.parser]
    if (!dataParser) {
        handleInputError('Invalid data parser');
    }

    investment = parseFloat(argv.investment)
    if (!investment) {
        handleInputError('Invalid investment');
    }

    profitability = parseFloat(argv.profitability)
    if (!profitability) {
        handleInputError('No profitability provided');
    }

    // Find the data file based on the command line argument.
    if (!argv.data) {
        handleInputError('No data file provided');
    }

    if (!argv.database) {
        handleInputError('No database provided');
    }

    // Set up database connection.
    db.initialize(argv.database);

    try {
        dataParser.parse(argv.data).then(function(parsedData) {
            var studyDefinitions = optimizerFn.studyDefinitions;
            var studies = [];
            var cumulativeData = [];
            var previousDataPoint = null;
            var dataCount = parsedData.length - 1;

            process.stdout.write('Preparing study data...');

            // Prepare studies.
            studyDefinitions.forEach(function(studyDefinition) {
                // Instantiate the study, and add it to the list of studies for this strategy.
                studies.push(new studyDefinition.study(studyDefinition.inputs, studyDefinition.outputMap));
            });

            // Parepare study data.
            parsedData.forEach(function(dataPoint, index) {
                // If there is a significant gap, start over.
                if (previousDataPoint && (dataPoint.timestamp - previousDataPoint.timestamp) > 600000) {
                    cumulativeData = [];
                }

                // Add the data point to the cumulative data.
                cumulativeData.push(dataPoint);

                // Iterate over each study...
                studies.forEach(function(study) {
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

                process.stdout.cursorTo(23);
                process.stdout.write(index + ' of ' + dataCount + ' completed');
            });

            process.stdout.cursorTo(23);
            process.stdout.write(dataCount + ' of ' + dataCount + ' completed\n');

            Backtest.find(backtestConstraints, function(error, backtests) {
                var backtestCount = backtests.length - 1;
                var backtestTasks = [];

                // Iterate through the remaining backtests.
                process.stdout.write('Forward testing...\n');

                backtests.forEach(function(backtest, index) {
                    backtestTasks.push(function(taskCallback) {
                        // Set up a new strategy instance.
                        var strategy = new strategyFn(argv.symbol, [backtest.configuration]);

                        // Backtest (forward test).
                        var results = strategy.backtest(parsedData, investment, profitability);

                        // Save results.
                        Forwardtest.create(_.extend(results, {
                            symbol: argv.symbol,
                            strategyUuid: backtest.strategyUuid,
                            configuration: backtest.configuration
                        }), function() {
                            process.stdout.cursorTo(18);
                            process.stdout.write(index + ' of ' + backtestCount + ' completed');

                            // Forward test the next backtest.
                            taskCallback();
                        });
                    });
                });

                async.series(backtestTasks, function(error) {
                    process.stdout.cursorTo(18);
                    process.stdout.write(backtestCount + ' of ' + backtestCount + ' completed\n');

                    db.disconnect();
                    done();
                });
            });
        });
    }
    catch (error) {
        cosole.error(error.message || error);
        process.exit(1);
    }
});

gulp.task('combine', function(done) {
    function showUsageInfo() {
        console.log('Example usage:\n');
        console.log('gulp combine --symbol AUDJPY --strategy Reversals --investment 1000 --profitability 0.7 --database forex-backtesting\n');
    }

    function handleInputError(message) {
        gutil.log(gutil.colors.red(message));
        showUsageInfo();
        process.exit(1);
    }

    var db = require('./db');
    var Backtest = require('./src/models/Backtest');
    var Position = require('./src/models/Position');
    var Combination = require('./src/models/Combination');
    var positionTester = require('./src/positionTester');

    var profitability = 0.0;

    var backtestConstraints = {
        symbol: argv.symbol,
        //strategyName: argv.strategy,
        minimumProfitLoss: {'$gte': -20000},
        maximumConsecutiveLosses: {'$lte': 20},
        winRate: {'$gte': 0.62},
        tradeCount: {'$gte': 3000},
    };

    // Find the symbol based on the command line argument.
    if (!argv.symbol) {
        handleInputError('No symbol provided');
    }

    // Find the strategy based on the command line argument.
    if (!argv.strategy) {
        handleInputError('Invalid strategy');
    }

    investment = parseFloat(argv.investment)
    if (!investment) {
        handleInputError('Invalid investment');
    }

    profitability = parseFloat(argv.profitability)
    if (!profitability) {
        handleInputError('No profitability provided');
    }

    if (!argv.database) {
        handleInputError('No database provided');
    }

    // Set up database connection.
    db.initialize(argv.database);

    // Find all backtests for the symbol.
    Backtest.find(backtestConstraints, function(error, backtests) {
        // Sort backtests descending by profitLoss.
        backtests = _.sortBy(backtests, 'profitLoss').reverse();

        // Use the highest profit/loss figure as the benchmark.
        var benchmarkProfitLoss = 0;
        var optimalConfigurations = [];
        var optimalPositions = [];
        var percentage = 0.0;
        var backtestCount = backtests.length - 1;
        var tasks = [];

        // Iterate through the remaining backtests.
        process.stdout.write('Combining configurations...');

        backtests.forEach(function(backtest, index) {
            tasks.push(function(taskCallback) {
                process.stdout.cursorTo(27);
                process.stdout.write(index + ' of ' + backtestCount + ' completed (' + optimalConfigurations.length + ' / $' + benchmarkProfitLoss + ')');

                // Find all positions for each backtest.
                Position.find({strategyUuid: backtest.strategyUuid}, function(error, positions) {
                    // Test with the optimal positions combined with the current positions.
                    var testPositions = optimalPositions.concat(positions);

                    // Get the unique set of trades.
                    var testPositions = _.uniq(testPositions, function(position) {
                        return position.timestamp;
                    });

                    // Sort positions by timestamp.
                    testPositions = _.sortBy(testPositions, 'timestamp');

                    // Determine if all the trades combined results in an improvement.
                    var testResults = positionTester.test(testPositions);

                    // See if the test resulted in an improvement.
                    if (testResults.profitLoss >= benchmarkProfitLoss + 100 && testResults.winRate >= 0.62 && testResults.tradeCount >= 3000 && testResults.maximumConsecutiveLosses <= 20 && testResults.minimumProfitLoss >= -20000) {
                        // Use the positions in future tests.
                        optimalPositions = testPositions;

                        // Include the backtest configuration in the list of optimal configurations.
                        optimalConfigurations.push(backtest.configuration);

                        // Update the benchmark.
                        benchmarkProfitLoss = testResults.profitLoss;
                    }

                    taskCallback(error);
                });
            });
        });

        // Execute the tasks, in order.
        async.series(tasks, function(error) {
            var optimalResults = positionTester.test(optimalPositions);

            // Save the results.
            Combination.create({
                symbol: argv.symbol,
                strategyName: argv.strategy,
                results: optimalResults,
                configurations: optimalConfigurations,
                positions: optimalPositions
            }, function() {
                process.stdout.cursorTo(27);
                process.stdout.write(backtestCount + ' of ' + backtestCount + ' completed\n');
                done();
                process.exit();
            });
        });
    });
});
