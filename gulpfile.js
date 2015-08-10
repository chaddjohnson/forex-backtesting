var gulp = require('gulp');
var gutil = require('gulp-util');
var argv = require('yargs').argv;

function showExampleUsage() {
    console.log('\n');
    console.log('Example usage:');
    console.log('\n');
    console.log('gulp backtest --parser trueFx --data ./data/EURUSD-2015-01.csv --strategy NateAug2015 --investment 1000 --profitability 0.85');
    console.log('\n');
}

gulp.task('backtest', function(done) {
    var dataParsers = require('./src/dataParsers');
    var strategies = require('./src/strategies');

    var strategyFn;
    var strategy;
    var parsedData = [];
    var dataParser;
    var profitability = 0.0;

    // Find the strategy based on the command line argument.
    if (!strategy = strategies[argv.strategy]) {
        gutil.log(gutil.colors.red('Invalid strategy'));
        showExampleUsage();
        process.exit(1);
    }

    // Find the raw data parser based on command line argument.
    if (!dataParser = dataParsers[argv.parser]) {
        gutil.log(gutil.colors.red('Invalid data parser'));
        showExampleUsage();
        process.exit(1);
    }

    if (!investment = parseFloat(argv.investment)) {
        gutil.log(gutil.colors.red('Invalid investment'));
        showExampleUsage();
        process.exit(1);
    }

    if (!profitability = parseFloat(argv.profitability)) {
        gutil.log(gutil.colors.red('Invalid profitability'));
        showExampleUsage();
        process.exit(1);
    }

    try {
        // Parse the raw data file.
        parsedData = dataParsers.parse(argv.data).then(function() {
            // Prepare the strategy.
            strategy = new strategyFn(parsedData);

            // Backtest the strategy against the parsed data.
            strategy.backtest(investment, profitability);
        }).done(done);
    }
    catch (error) {
        console.error(error.message || error);
        process.exit(1);
    }
});
