var fs = require('fs');
var es = require('event-stream');
var Q = require('q');

module.exports.parse = function(filePath) {
    var deferred = Q.defer();
    var stream;

    var transactionData = [];
    var formattedData = [];
    var volume = 0.0;

    if (!filePath) {
        throw 'No filePath provided to dataParser.'
    }

    stream = fs.createReadStream(filePath)
        .pipe(es.split())
        .pipe(es.mapSync(function(line) {
            // Pause the readstream.
            stream.pause();

            (function() {
                // Ignore blank lines.
                if (!line) {
                    stream.resume();
                    return;
                }

                transactionData = line.split(',');
                volume = parseFloat(transactionData[6]);

                formattedData.push({
                    timestamp: new Date(transactionData[0] + ' ' + transactionData[1] + ':00').getTime(),
                    volume: volume,
                    open: parseFloat(transactionData[2]),
                    high: parseFloat(transactionData[3]),
                    low: parseFloat(transactionData[4]),
                    close: parseFloat(transactionData[5])
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
