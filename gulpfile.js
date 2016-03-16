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

gulp.task('preparedata', function(done) {
    function showUsageInfo() {
        console.log('Example usage:\n');
        console.log('gulp data --symbol AUDJPY --parser metatrader --data ./data/metatrader/three-year/AUDJPY.csv --optimizer Reversals --database forex-backtesting\n');
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
            optimizer.prepareStudyData(parsedData, function() {
                db.disconnect();
                done();
                process.exit(0);
            });
        });
    }
    catch (error) {
        console.error(error.message || error);
        process.exit(1);
    }
});

gulp.task('test', function(done) {
    function showUsageInfo() {
        console.log('Example usage:\n');
        console.log('gulp test --symbol AUDJPY --group 4 --type testing --investment 1000 --profitability 0.7 --database forex-backtesting\n');
    }

    function handleInputError(message) {
        gutil.log(gutil.colors.red(message));
        showUsageInfo();
        process.exit(1);
    }

    var db = require('./db');
    var DataPoint = require('./src/models/DataPoint');
    var Forwardtest = require('./src/models/Forwardtest');
    var Validation = require('./src/models/Validation');
    var optimizerFn = require('./src/optimizers/Reversals');
    var strategyFn = require('./src/strategies/combined/Reversals');
    var optimizer;
    var group = 0;
    var investment = 0.0;
    var profitability = 0.0;
    var typeKey = '';
    var dataConstraints = {};
    var forwardtestConstraints;
    var tasks = [];

    // Find the symbol based on the command line argument.
    if (!argv.symbol) {
        handleInputError('No symbol provided');
    }

    if (!argv.type) {
        handleInputError('No type provided');
    }

    group = parseInt(argv.group);
    if (!group) {
        handleInputError('No group provided');
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

    forwardtestConstraints = {
        symbol: argv.symbol,
        group: group - 1,
        winRate: {'$gte': 0.58}
    };

    typeKey = 'data.groups.' + argv.type;
    dataConstraints.symbol = argv.symbol;
    dataConstraints[typeKey] = group;

    optimizer = new optimizerFn(argv.symbol, group);

    // Set up database connection.
    db.initialize(argv.database);

    // Get configurations.
    tasks.push(function(taskCallback) {
        if (group === 1) {
            // Just let optimizer use its own configurations.
            taskCallback();
        }
        else {
            Forwardtest.find(forwardtestConstraints, function(error, forwardtests) {
                // Override configurations used by optimizer.
                optimizer.configurations = _.pluck(forwardtests, 'configuration');
                taskCallback();
            });
        }
    });

    tasks.push(function(taskCallback) {
        try {
            // Override the data query for the optimizer.
            optimizer.setQuery(dataConstraints);
            optimizer.setType(argv.type);

            // Run optimization.
            optimizer.optimize([], investment, profitability, function() {
                taskCallback();
            });
        }
        catch (error) {
            console.error(error.message || error);
            process.exit(1);
        }
    });

    async.series(tasks, function() {
        db.disconnect();
        done();
        process.exit(0);
    });
});

gulp.task('average', function(done) {
    function showUsageInfo() {
        console.log('Example usage:\n');
        console.log('gulp average --symbol AUDJPY --database forex-backtesting\n');
    }

    function handleInputError(message) {
        gutil.log(gutil.colors.red(message));
        showUsageInfo();
        process.exit(1);
    }

    var db = require('./db');
    var Forwardtest = require('./src/models/Forwardtest');
    var Validation = require('./src/models/Validation');
    var ValidationAverage = require('./src/models/ValidationAverage');

    // Find the symbol based on the command line argument.
    if (!argv.symbol) {
        handleInputError('No symbol provided');
    }

    if (!argv.database) {
        handleInputError('No database provided');
    }

    // Set up database connection.
    db.initialize(argv.database);

    // Find the forward tests for group 10.
    Forwardtest.find({symbol: argv.symbol, group: 10}, function(error, forwardtests) {
        var tasks = [];
        var forwardtestCount = forwardtests.length;
        var percentage = 0.0;
        var validationAverages = [];

        process.stdout.write('Calculating validation averages...');

        forwardtests.forEach(function(forwardtest, index) {
            tasks.push(function(taskCallback) {
                // For each forward test, find the validation results for each round given the forward test's configuration.
                Validation.find({configuration: forwardtest.configuration}, function(error, validations) {
                    var validationCount = validations.length;

                    if (validationCount < 8) {
                        taskCallback();
                    }

                    // Calculate averages for all properties.
                    var averageProfitLoss = _.reduce(validations, function(memo, validation) {
                        return memo + validation.profitLoss;
                    }, 0) / validationCount;

                    var averageTradeCount = _.reduce(validations, function(memo, validation) {
                        return memo + validation.tradeCount;
                    }, 0) / validationCount;

                    var averageWinRate = _.reduce(validations, function(memo, validation) {
                        return memo + validation.winRate;
                    }, 0) / validationCount;

                    var averageMaximumConsecutiveLosses = _.reduce(validations, function(memo, validation) {
                        return memo + validation.maximumConsecutiveLosses;
                    }, 0) / validationCount;

                    var averageMinimumProfitLoss = _.reduce(validations, function(memo, validation) {
                        return memo + validation.minimumProfitLoss;
                    }, 0) / validationCount;

                    // Add to averages to be saved.
                    validationAverages.push({
                        symbol: argv.symbol,
                        configuration: forwardtest.configuration,
                        profitLoss: averageProfitLoss,
                        tradeCount: averageTradeCount,
                        winRate: averageWinRate,
                        maximumConsecutiveLosses: averageMaximumConsecutiveLosses,
                        minimumProfitLoss: averageMinimumProfitLoss
                    });

                    percentage = ((index / forwardtestCount) * 100).toFixed(5);
                    process.stdout.cursorTo(34);
                    process.stdout.write(percentage + '%');

                    taskCallback();
                });
            });
        });

        async.series(tasks, function() {
            process.stdout.cursorTo(34);
            process.stdout.write((100).toFixed(5) + '%\n');
            process.stdout.write('Saving validation averages...');

            ValidationAverage.collection.insert(validationAverages, function(error) {
                process.stdout.write('done\n');

                db.disconnect();
                done();
                process.exit(0);
            });
        });
    });
});
