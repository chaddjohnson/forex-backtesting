var fs = require('fs');
var es = require('event-stream');
var Q = require('q');

module.exports.parse = function(symbol, filePath) {
    var deferred = Q.defer();
    var stream;

    var transactionData = [];
    var formattedData = [];
    var index = -1;
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
                index++;

                // Skip the first line (being header data).
                if (index === 0) {
                    stream.resume();
                    return;
                }

                transactionData = line.split(',');
                volume = parseFloat(transactionData[5]);

                if (volume > 0) {
                    formattedData.push({
                        symbol: symbol,
                        timestamp: new Date(transactionData[0].replace(/(\d{2})\.(\d{2})\.(\d{4}) (.*)/, '$2-$1-$3 $4')).getTime(),
                        volume: volume,
                        price: parseFloat(transactionData[4])
                    });
                }

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
