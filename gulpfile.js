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
    var Backtest = require('./src/models/Backtest');
    var Forwardtest = require('./src/models/Forwardtest');

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
                var backtestConstraints = {
                    symbol: argv.symbol,
                    winRate: {'$gte': 0.62}
                };

                // Find all backtests that meet a certain criteria.
                Backtest.find(backtestConstraints, function(error, backtests) {
                    backtests.forEach(function(backtest) {
                        Forwardtest.create({
                            symbol: argv.symbol,
                            strategyUuid: backtest.strategyUuid,
                            configuration: backtest.configuration,
                            profitLoss: backtest.profitLoss,
                            winCount: backtest.winCount,
                            loseCount: backtest.loseCount,
                            tradeCount: backtest.tradeCount,
                            winRate: backtest.winRate,
                            maximumConsecutiveLosses: backtest.maximumConsecutiveLosses,
                            minimumProfitLoss: backtest.minimumProfitLoss
                        }, function() {
                            db.disconnect();
                            done();
                        });
                    });
                });
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
        console.log('gulp forwardtest --symbol AUDJPY --group 4 --type testing --investment 1000 --profitability 0.7 --database forex-backtesting\n');
    }

    function handleInputError(message) {
        gutil.log(gutil.colors.red(message));
        showUsageInfo();
        process.exit(1);
    }

    var db = require('./db');
    var Forwardtest = require('./src/models/Forwardtest');
    var Validation = require('./src/models/Validation');
    var optimizerFn = require('./src/optimizers/Reversals');
    var strategyFn = require('./src/strategies/combined/Reversals');
    var group = 0;
    var typeKey = '';
    var investment = 0.0;
    var profitability = 0.0;
    var dataConstraints;
    var forwardtestConstraints;
    var ResultsModel;

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

    typeKey = 'data.groups.' + argv.type;

    dataConstraints = {
        symbol: argv.symbol,
        typeKey: group
    }

    forwardtestConstraints = {
        symbol: argv.symbol,
        group: group - 1,
        winRate: {'$gte': 0.62}
    };

    ResultsModel = argv.type === 'forwardtest' ? Forwardtest : Validation;

    // Set up database connection.
    db.initialize(argv.database);

    try {
        DataPoint.find(dataConstraints, function(error, data) {
            Forwardtest.find(forwardtestConstraints, function(error, forwardtests) {
                var forwardtestCount = forwardtests.length;
                var forwardtestTasks = [];

                // Iterate through the remaining forward tests.
                process.stdout.write('Forward testing...\n');

                forwardtests.forEach(function(forwardtest, index) {
                    forwardtestTasks.push(function(taskCallback) {
                        // Set up a new strategy instance.
                        var strategy = new strategyFn(argv.symbol, [forwardtest.configuration]);

                        strategy.setProfitLoss(10000);

                        // Forward test (backtest).
                        var results = strategy.backtest(data, investment, profitability);

                        // Save results.
                        ResultsModel.create(_.extend(results, {
                            symbol: argv.symbol,
                            group: group,
                            strategyUuid: forwardtest.strategyUuid,
                            configuration: forwardtest.configuration
                        }), function() {
                            process.stdout.cursorTo(18);
                            process.stdout.write(index + ' of ' + forwardtestCount + ' completed');

                            // Forward test the next forward test.
                            taskCallback();
                        });
                    });
                });

                async.series(forwardtestTasks, function(error) {
                    process.stdout.cursorTo(18);
                    process.stdout.write(forwardtestCount + ' of ' + forwardtestCount + ' completed\n');

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
