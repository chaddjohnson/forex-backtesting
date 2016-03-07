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
    var typeKey = '';
    var investment = 0.0;
    var profitability = 0.0;
    var dataConstraints;
    var forwardtestConstraints;
    var ResultsModel;
    var data = [];
    var configurations = [];
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

    ResultsModel = argv.type === 'testing' ? Forwardtest : Validation;
    optimizer = new optimizerFn(argv.symbol);

    // Set up database connection.
    db.initialize(argv.database);

    try {
        // Get data.
        tasks.push(function(taskCallback) {
            DataPoint.find(dataConstraints, function(error, documents) {
                data = documents;
                taskCallback()
            });
        });

        // Get configurations.
        tasks.push(function(taskCallback) {
            if (group === 1) {
                configurations = optimizer.configurations;
                taskCallback();
            }
            else {
                Forwardtest.find(forwardtestConstraints, function(error, forwardtests) {
                    configurations = _.pluck(forwardtests, 'configuration');
                    taskCallback();
                });
            }
        });

        tasks.push(function(taskCallback) {
            var configurationCount = configurations.length;
            var testTasks = [];

            // Iterate through the remaining forward tests.
            process.stdout.write('Forward testing...\n');

            configurations.forEach(function(configuration, index) {
                testTasks.push(function(testTaskCallback) {
                    // Set up a new strategy instance.
                    var strategy = new strategyFn(argv.symbol, [configuration]);

                    strategy.setProfitLoss(10000);

                    // Forward test (backtest).
                    var results = strategy.backtest(data, investment, profitability);

                    // Save results.
                    ResultsModel.create(_.extend(results, {
                        symbol: argv.symbol,
                        group: group,
                        configuration: configuration
                    }), function() {
                        process.stdout.cursorTo(18);
                        process.stdout.write(index + ' of ' + configurationCount + ' completed');

                        // Forward test the next forward test.
                        testTaskCallback();
                    });
                });
            });

            async.series(testTasks, function(error) {
                process.stdout.cursorTo(18);
                process.stdout.write(configurationCount + ' of ' + configurationCount + ' completed\n');

                taskCallback()
            });
        });

        async.series(tasks, function(error) {
            if (error) {
                console.error(error.message || error);
            }
            db.disconnect();
            done();
            process.exit(0);
        });
    }
    catch (error) {
        console.error(error.message || error);
        process.exit(1);
    }
});
