var mongoose = require('mongoose');
var dbUri = 'mongodb://localhost/forex-trading';

module.exports.initialize = function() {
    mongoose.connect(dbUri);
    mongoose.connection.on('error', console.error.bind(console, 'Database connection error:'));
    mongoose.connection.once('connected', function() {
        console.log('Connected to DB: ' + dbUri);
    });
    return mongoose.connection;
};
