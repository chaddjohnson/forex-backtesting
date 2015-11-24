var db = require('../../db');
var async = require('async');
var strategyFns = require('../strategies');
var strategies = [];

db.initialize('forex-backtesting');

function init(strategyName, symbol, configuration, dataPointCount) {
    strategies.push(new strategyFns.optimization[strategyName](symbol, configuration, dataPointCount));
};

function backtest(dataPoint, index, investment, profitability) {
    var tasks = [];

    strategies.forEach(function(strategy) {
        tasks.push(function(taskCallback) {
            strategy.backtest(dataPoint, index, investment, profitability, function() {
                taskCallback();
            });
        });
    });

    async.series(tasks, function() {
        process.send({type: 'done'});
    });
};

function getResults() {
    var allResults = [];

    strategies.forEach(function(strategy) {
        var results = strategy.getResults();

        allResults.push({
            symbol: strategy.getSymbol(),
            strategyUuid: strategy.getUuid(),
            strategyName: strategy.constructor.name,
            configuration: strategy.getConfiguration(),
            profitLoss: results.profitLoss,
            winCount: results.winCount,
            loseCount: results.loseCount,
            tradeCount: results.tradeCount,
            winRate: results.winRate,
            maximumConsecutiveLosses: results.maximumConsecutiveLosses,
            minimumProfitLoss: results.minimumProfitLoss
        });
    });

    process.send({type: 'results', data: allResults});
};


process.on('message', function(message) {
    switch (message.type) {
        case 'init':
            init(message.data.strategyName, message.data.symbol, message.data.configuration, message.data.dataPointCount);
            break;

        case 'backtest':
            backtest(message.data.dataPoint, message.data.index, message.data.investment, message.data.profitability);
            break;

        case 'results':
            getResults();
            break;
    }
});
