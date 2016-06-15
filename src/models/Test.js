var mongoose = require('mongoose');

var testSchema = mongoose.Schema({
    symbol: {type: String, required: true},
    strategyName: {type: String, required: false},
    group: {type: Number, required: true},
    profitLoss: {type: Number, required: true},
    winCount: {type: Number, required: true},
    loseCount: {type: Number, required: true},
    tradeCount: {type: Number, required: true},
    winRate: {type: Number, required: true},
    maximumConsecutiveLosses: {type: Number, required: true},
    minimumProfitLoss: {type: Number, required: true},
    configuration: {type: mongoose.Schema.Types.Mixed, required: true}
});

module.exports = mongoose.connection.model('Test', testSchema);
