var mongoose = require('mongoose');

var dataPointSchema = mongoose.Schema({
    symbol: {type: String, required: true},
    data: {type: mongoose.Schema.Types.Mixed, required: true}
});

module.exports = mongoose.connection.model('DataPoint', dataPointSchema);
