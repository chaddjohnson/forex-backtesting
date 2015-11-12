var gulp = require('gulp');
var gutil = require('gulp-util');
var argv = require('yargs').argv;
var path = require('path');
var _ = require('underscore');
var async = require('async');

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

scheduleGarbageCollection();

gulp.task('backtest', function(done) {
    function showUsageInfo() {
        console.log('Example usage:\n');
        console.log('gulp backtest --symbol EURCHF --parser metatrader --data ./data/metatrader/three-year/EURCHF.csv --strategy Reversals --investment 1000 --profitability 0.7\n');
    }

    function handleInputError(message) {
        gutil.log(gutil.colors.red(message));
        showUsageInfo();
        process.exit(1);
    }

    var dataParsers = require('./src/dataParsers');
    var strategies = require('./src/strategies');

    var strategyFn;
    var dataParser;
    var profitability = 0.0;

    // Find the data file based on the command line argument.
    if (!argv.data) {
        handleInputError('No data file provided');
    }

    // Find the symbol based on the command line argument.
    if (!argv.symbol) {
        handleInputError('No symbol provided');
    }

    // Find the strategy based on the command line argument.
    strategyFn = strategies[argv.strategy]
    if (!strategyFn) {
        handleInputError('Invalid strategy');
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

    try {
        // Parse the raw data file.
        dataParser.parse(argv.data).then(function(parsedData) {
            // Prepare the strategy.
            var strategy = new strategyFn(argv.symbol);

            // Backtest the strategy against the parsed data.
            strategy.backtest(parsedData, investment, profitability);

            done();
        });
    }
    catch (error) {
        console.error(error.message || error);
        process.exit(1);
    }
});

gulp.task('optimize', function(done) {
    function showUsageInfo() {
        console.log('Example usage:\n');
        console.log('gulp optimize --symbol EURCHF --parser metatrader --data ./data/metatrader/three-year/EURCHF.csv --optimizer Reversals --investment 1000 --profitability 0.7 --database forex-backtesting\n');
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

gulp.task('combine', function(done) {
    function showUsageInfo() {
        console.log('Example usage:\n');
        console.log('gulp combine --symbol EURCHF --strategy Reversals --investment 1000 --profitability 0.7 --database forex-backtesting\n');
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
        minimumProfitLoss: {'$gte': -10000},
        maximumConsecutiveLosses: {'$lte': 10},
        winRate: {'$gte': 0.65},
        profitLoss: {'$gte': 25000}
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
        backtests = _(backtests).sortBy('profitLoss').reverse();

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
                    var testPositions = _(testPositions).uniq(function(position) {
                        return position.timestamp;
                    });

                    // Sort positions by timestamp.
                    testPositions = _(testPositions).sortBy('timestamp');

                    // Determine if all the trades combined results in an improvement.
                    var testResults = positionTester.test(testPositions);

                    // See if the test resulted in an improvement.
                    if (testResults.profitLoss >= benchmarkProfitLoss + 1000 && testResults.winRate >= 0.65 && testResults.profitLoss >= 25000 && testResults.maximumConsecutiveLosses <= 10 && testResults.minimumProfitLoss >= -10000) {
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
                process.stdout.write(configurationCount + ' of ' + configurationCount + ' completed\n');;
                done();
                process.exit();
            });
        });
    });
});

gulp.task('generateMinuteData', function(done) {
    function showUsageInfo() {
        console.log('Example usage:\n');
        console.log('gulp generateMinuteData --symbol EURCHF --data /path/to/tickdata.csv\n');
    }

    function handleInputError(message) {
        gutil.log(gutil.colors.red(message));
        showUsageInfo();
        process.exit(1);
    }

    var minuteDataConverter = require('./src/converters/minuteData');

    // Find the data file based on the command line argument.
    if (!argv.data) {
        handleInputError('No data file provided');
    }

    // Find the symbol based on the command line argument.
    if (!argv.symbol) {
        handleInputError('No symbol provided');
    }

    try {
        minuteDataConverter.convert(argv.data).then(function(convertedData) {
            convertedData.forEach(function(dataPoint) {
                console.log(dataPoint.join(','));
            });
            done();
        });
    }
    catch (error) {
        console.error(error.message || error);
        process.exit(1);
    }
});
