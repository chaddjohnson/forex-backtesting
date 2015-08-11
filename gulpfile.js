var gulp = require('gulp');
var gutil = require('gulp-util');
var argv = require('yargs').argv;

gulp.task('backtest', function(done) {
    function showUsageInfo() {
        console.log('Example usage:\n');
        console.log('    gulp backtest --symbol EURUSD --parser dukascopy --data ./data/EURUSD_Candlestick_1_m_BID_04.08.2014-08.08.2015.csv --strategy NateAug2015 --investment 1000 --profitability 0.7 --out ./data/processed/EURUSD_Candlestick_1_m_BID_04.08.2014-08.08.2015.csv\n');
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
    strategy = strategies[argv.strategy]
    if (!strategy) {
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
            strategy = new strategyFn(parsedData);

            if (argv.out) {
                strategy.setDataOutputFilePath(argv.out);
            }

            // Backtest the strategy against the parsed data.
            strategy.backtest(investment, profitability);

            done();
        });
    }
    catch (error) {
        console.error(error.message || error);
        process.exit(1);
    }
});
