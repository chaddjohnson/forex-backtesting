var mongoose = require('mongoose');

var positionSchema = mongoose.Schema({
    symbol: {type: String, required: true},
    strategyUuid: {type: String, required: true},
    transactionType: {type: String, required: true},
    timestamp: {type: Number, required: true},
    price: {type: Number, required: true},
    investment: {type: Number, required: true},
    profitability: {type: Number, required: true},
    closePrice: {type: Number, required: true},
    expirationTimestamp: {type: Number, required: true},
    closeTimestamp: {type: Number, required: true},
    profitLoss: {type: Number, required: true}
});

module.exports = mongoose.connection.model('Position', positionSchema);
