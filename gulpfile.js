var gulp = require('gulp');
var gutil = require('gulp-util');
var argv = require('yargs').argv;
var path = require('path');

gulp.task('backtest', function(done) {
    function showUsageInfo() {
        console.log('Example usage:\n');
        console.log('    gulp backtest --symbol EURUSD --parser dukascopy --data ./data/EURUSD.csv --strategy Reversals --investment 1000 --profitability 0.7 --out ./data/processed/EURUSD.csv\n');
        console.log('Note that only minute-by-minute tick data may be used.\n');
    }

    var dataParsers = require('./src/dataParsers');
    var strategies = require('./src/strategies');

    var strategyFn;
    var strategy;
    var dataParser;
    var profitability = 0.0;

    // Find the data file based on the command line argument.
    if (!argv.data) {
        gutil.log(gutil.colors.red('No data file provided'));
        showUsageInfo();
        process.exit(1);
    }

    // Find the symbol based on the command line argument.
    if (!argv.symbol) {
        gutil.log(gutil.colors.red('No symbol provided'));
        showUsageInfo();
        process.exit(1);
    }

    // Find the strategy based on the command line argument.
    strategyFn = strategies[argv.strategy]
    if (!strategyFn) {
        gutil.log(gutil.colors.red('Invalid strategy'));
        showUsageInfo();
        process.exit(1);
    }

    // Find the raw data parser based on command line argument.
    dataParser = dataParsers[argv.parser]
    if (!dataParser) {
        gutil.log(gutil.colors.red('Invalid data parser'));
        showUsageInfo();
        process.exit(1);
    }

    investment = parseFloat(argv.investment)
    if (!investment) {
        gutil.log(gutil.colors.red('Invalid investment'));
        showUsageInfo();
        process.exit(1);
    }

    profitability = parseFloat(argv.profitability)
    if (!profitability) {
        gutil.log(gutil.colors.red('No profitability provided'));
        showUsageInfo();
        process.exit(1);
    }

    try {
        // Parse the raw data file.
        dataParser.parse(argv.symbol, argv.data).then(function(parsedData) {
            // Prepare the strategy.
            strategy = new strategyFn();

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
