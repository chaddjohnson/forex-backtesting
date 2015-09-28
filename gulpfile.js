var gulp = require('gulp');
var gutil = require('gulp-util');
var argv = require('yargs').argv;
var path = require('path');

gulp.task('backtest', function(done) {
    function showUsageInfo() {
        console.log('Example usage:\n');
        console.log('gulp backtest --symbol EURUSD --parser dukascopy --data ./data/EURUSD.csv --strategy Reversals --investment 1000 --profitability 0.7 --out ./data/processed/EURUSD.csv\n');
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
        dataParser.parse(argv.symbol, argv.data).then(function(parsedData) {
            // Prepare the strategy.
            var strategy = new strategyFn();

            if (argv.out) {
                strategy.setDataOutputFilePath(path.join(__dirname, argv.out));
            }

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
        console.log('gulp optimize --symbol EURUSD --parser metatrader --data-directory ./data/metatrader/three-year --optimizer Reversals --investment 1000 --profitability 0.7\n');
    }

    function handleInputError(message) {
        gutil.log(gutil.colors.red(message));
        showUsageInfo();
        process.exit(1);
    }

    var db = require('./db');
    var dataParsers = require('./src/dataParsers');

    var strategyOptimizerFn;
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

    // Find the data directory based on the command line argument.
    if (!argv.dataDirectory) {
        handleInputError('No data directory provided');
    }

    // Find the strategy based on the command line argument.
    strategyOptimizerFn = optimizers[argv.strategyOptimizer]
    if (!strategyOptimizerFn) {
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

    // Set up database connection.
    db.initialize();

    try {
        // Parse the raw data file.
        dataParser.parse(argv.symbol, argv.data).then(function(parsedData) {
            // Prepare the strategy.
            var strategyOptimizer = new strategyOptimizerFn(symbol);

            // Backtest the strategy against the parsed data.
            strategyOptimizer.optimize(parsedData, investment, profitability);

            done();
        });
    }
    catch (error) {
        console.error(error.message || error);
        process.exit(1);
    }
});
