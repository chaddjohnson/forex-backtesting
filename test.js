var _ = require('underscore');
var db = require('./db');
var DataPoint = require('./src/models/DataPoint');
var Combination = require('./src/models/Combination');
var strategyFn = require('./src/strategies/combined/Reversals');

db.initialize('blah');

DataPoint.find({symbol: process.argv[2]}).sort({'data.timestamp': 1}).exec(function(error, dataPoints) {
    dataPoints = _(dataPoints).map(function(dataPoint) {
        return dataPoint.data;
    });

    Combination.findOne({symbol: process.argv[2]}, function(error, combination) {
        var strategy = new strategyFn(process.argv[2], combination.configurations);
        //strategy.setShowTrades(true);
        strategy.backtest(dataPoints, 1000, 0.76);

        console.log(strategy.getResults());
        process.exit();
    });
});
