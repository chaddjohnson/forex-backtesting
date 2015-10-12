var mongoose = require('mongoose');

module.exports.initialize = function(dbName) {
    var dbUri = 'mongodb://localhost/' + dbName;

    mongoose.connect(dbUri);
    mongoose.connection.on('error', console.error.bind(console, 'Database connection error:'));
    mongoose.connection.once('connected', function() {
        console.log('Connected to DB: ' + dbUri);
    });
    return mongoose.connection;
};

module.exports.disconnect = function() {
    mongoose.disconnect();
};
