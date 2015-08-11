var fs = require('fs');
var es = require('event-stream');
var Q = require('q');

module.exports.parse = function(symbol, filePath) {
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

            // pause the readstream
            stream.pause();

            (function() {
                transactionData = line.split(',');
                volume = parseFloat(transactionData[6]);

                formattedData.push({
                    symbol: symbol,
                    timestamp: new Date(transactionData[0] + ' ' + transactionData[1] + ':00').getTime(),
                    volume: volume,
                    price: parseFloat(transactionData[5])
                });

                // resume the readstream
                stream.resume();
            })();
        })
    );

    stream.on('close', function() {
        deferred.resolve(formattedData);
    });

    return deferred.promise;
};
