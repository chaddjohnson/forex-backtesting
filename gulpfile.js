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

gulp.task('average', function(done) {
    function showUsageInfo() {
        console.log('Example usage:\n');
        console.log('gulp average --symbol AUDNZD --database forex-backtesting\n');
    }

    function handleInputError(message) {
        gutil.log(gutil.colors.red(message));
        showUsageInfo();
        process.exit(1);
    }

    var db = require('./db');
    var Test = require('./src/models/Test');
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
    Test.find({symbol: argv.symbol, group: 10}, function(error, tests) {
        var tasks = [];
        var testCount = tests.length;
        var percentage = 0.0;
        var validationAverages = [];

        process.stdout.write('Calculating validation averages...');

        tests.forEach(function(test, index) {
            tasks.push(function(taskCallback) {
                // For each test, find the validation results for each round given the test's configuration.
                Validation.find({symbol: argv.symbol, configuration: test.configuration}, function(error, validations) {
                    var validationCount = validations.length;

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
                        strategyName: 'reversals',
                        configuration: test.configuration,
                        profitLoss: averageProfitLoss,
                        tradeCount: averageTradeCount,
                        winRate: averageWinRate,
                        maximumConsecutiveLosses: averageMaximumConsecutiveLosses,
                        minimumProfitLoss: averageMinimumProfitLoss
                    });

                    percentage = ((index / testCount) * 100).toFixed(5);
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
