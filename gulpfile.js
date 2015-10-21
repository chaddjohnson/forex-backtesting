var gulp = require('gulp');
var gutil = require('gulp-util');
var argv = require('yargs').argv;
var path = require('path');
var _ = require('underscore');

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
    var strategies = require('./src/strategies');

    var DataPoint = require('./src/models/DataPoint');
    var Backtest = require('./src/models/Backtest');
    var Combination = require('./src/models/Combination');

    var strategyFn;
    var profitability = 0.0;

    // Find the symbol based on the command line argument.
    if (!argv.symbol) {
        handleInputError('No symbol provided');
    }

    // Find the strategy based on the command line argument.
    strategyFn = strategies.combined[argv.strategy]
    if (!strategyFn) {
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

    DataPoint.find({symbol: argv.symbol}, function(error, dataPoints) {
        var backtestConstraints = {
            symbol: argv.symbol,
            //strategyName: strategyFn.name,
            minimumProfitLoss: {'$gte': -10000},
            maximumConsecutiveLosses: {'$lte': 12},
            winRate: {'$gte': 0.6}
        };
        var data = _(dataPoints).map(function(dataPoint) {
            return dataPoint.data;
        });

        // Free up memory.
        dataPoints = null;

        Backtest.find(backtestConstraints).exec(function(error, backtests) {
            // Sort backtests descending by profitLoss.
            backtests = _(backtests).sortBy('profitLoss').reverse();

            // Use the highest profit/loss figure as the benchmark.
            var benchmarkProfitLoss = backtests[0].profitLoss;
            var optimalConfigurations = [];
            var configurations = _(backtests).map(function(backtest) {
                return backtest.configuration;
            });
            var percentage = 0.0;
            var configurationCount = configurations.length - 1;
            var strategy;
            var results;

            // Use the backtest having the highest profit/loss as first configuration.
            optimalConfigurations.push(configurations.shift());

            // Iterate through the remaining strategy configurations.
            process.stdout.write('Combining configurations...');
            configurations.forEach(function(configuration, index) {
                process.stdout.cursorTo(27);
                process.stdout.write(index + ' of ' + configurationCount + ' completed (' + optimalConfigurations.length + ' / $' + benchmarkProfitLoss + ')');

                // Make a shallow copy of the optimal configurations.
                var strategyConfigurations = _.clone(optimalConfigurations);

                // Add the current configuration onto the list of optimal configurations.
                strategyConfigurations.push(configuration);

                // Backtest the strategy using the optimal configurations plus the current configuration.
                strategy = new strategyFn(argv.symbol, strategyConfigurations);
                strategy.backtest(data, investment, profitability);
                results = strategy.getResults();

                // If including the configuration improved things, then include it in the list of optimal configurations.
                if (results.profitLoss >= benchmarkProfitLoss + 1000 && results.winRate >= 0.6 && results.maximumConsecutiveLosses <= 12 && results.minimumProfitLoss >= -10000) {
                    optimalConfigurations.push(configuration);

                    // Update the benchmark.
                    benchmarkProfitLoss = results.profitLoss;
                }
            });

            // Do a final backtest using the optimal configuration combination.
            strategy = new strategyFn(argv.symbol, optimalConfigurations);
            strategy.setShowTrades(true);
            strategy.backtest(data, investment, profitability);

            // Save the results.
            Combination.create({
                symbol: argv.symbol,
                strategyName: strategyFn.name,
                results: strategy.getResults(),
                configurations: optimalConfigurations
            }, function() {
                process.stdout.cursorTo(27);
                process.stdout.write(configurationCount + ' of ' + configurationCount + ' completed\n');;
                done();
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
