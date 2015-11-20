var fs = require('fs');
var es = require('event-stream');
var Q = require('q');

module.exports.parse = function(filePath) {
    var deferred = Q.defer();
    var stream;

    var transactionData = [];
    var formattedData = [];

    if (!filePath) {
        throw 'No filePath provided to dataParser.'
    }

    stream = fs.createReadStream(filePath)
        .pipe(es.split())
        .pipe(es.mapSync(function(line) {
            // Pause the read stream.
            stream.pause();

            (function() {
                // Ignore blank lines.
                if (!line) {
                    stream.resume();
                    return;
                }

                transactionData = line.split(',');

                formattedData.push({
                    timestamp: parseInt(transactionData[0]),
                    volume: 0,
                    open: parseFloat(transactionData[1]),
                    high: parseFloat(transactionData[2]),
                    low: parseFloat(transactionData[3]),
                    close: parseFloat(transactionData[4])
                });

                // Resume the read stream.
                stream.resume();
            })();
        }));

    stream.on('close', function() {
        deferred.resolve(formattedData);
    });

    return deferred.promise;
};
