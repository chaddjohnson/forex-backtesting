var mongoose = require('mongoose');

var validationSchema = mongoose.Schema({
    symbol: {type: String, required: true},
    group: {type: Number, required: true},
    strategyUuid: {type: String, required: true},
    configuration: {type: mongoose.Schema.Types.Mixed, required: true},
    profitLoss: {type: Number, required: true},
    winCount: {type: Number, required: true},
    loseCount: {type: Number, required: true},
    tradeCount: {type: Number, required: true},
    winRate: {type: Number, required: true},
    maximumConsecutiveLosses: {type: Number, required: true},
    minimumProfitLoss: {type: Number, required: true}
});

module.exports = mongoose.connection.model('Validation', validationSchema);
