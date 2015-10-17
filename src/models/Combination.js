var mongoose = require('mongoose');

var combinationSchema = mongoose.Schema({
    symbol: {type: String, required: true},
    strategyName: {type: String, required: true},
    results: {type: mongoose.Schema.Types.Mixed, required: true},
    configurations: []
});

module.exports = mongoose.connection.model('Combination', combinationSchema);
