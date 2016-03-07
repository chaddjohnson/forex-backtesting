var mongoose = require('mongoose');

var validationAverageSchema = mongoose.Schema({
    symbol: {type: String, required: true},
    configuration: {type: mongoose.Schema.Types.Mixed, required: true},
    profitLoss: {type: Number, required: true},
    tradeCount: {type: Number, required: true},
    winRate: {type: Number, required: true},
    maximumConsecutiveLosses: {type: Number, required: true},
    minimumProfitLoss: {type: Number, required: true}
});

module.exports = mongoose.connection.model('ValidationAverage', validationAverageSchema);
