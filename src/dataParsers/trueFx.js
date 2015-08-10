// SYMBOL,TIMESTAMP,BID,ASK

var fs = require('fs');
var es = require('event-stream');
var Q = require('q');

module.exports.parse = function() {
    var deferred = Q.defer();

    // var transactionData = [];
    // var transaction = {};

    // // Output headers.
    // //console.log('security,direction,amount,value,createdAt,expiresAt');

    // var stream = fs.createReadStream(process.argv[2])
    //     .pipe(es.split())
    //     .pipe(es.mapSync(function(line) {

    //         // pause the readstream
    //         stream.pause();

    //         (function() {
    //             transactionData = line.split(',');

    //             if (transactionData[3] && transactionData[1]) {
    //                 // Reformat transaction data into an object.
    //                 transaction = {
    //                     security: transactionData[0].replace('/', ''),
    //                     direction: '',
    //                     volume: 0,
    //                     price: parseFloat(transactionData[3]),
    //                     createdAt: new Date(transactionData[1].replace(/(\d{4})(\d{2})(\d{2}) (.*)/, '$1-$2-$3 $4')).getTime(),
    //                     expiresAt: 0
    //                 };

    //                 console.log(transaction.security + ',' + transaction.direction + ',' + transaction.volume + ',' + transaction.price + ',' + transaction.createdAt + ',' + transaction.expiresAt);
    //             }

    //             // resume the readstream
    //             stream.resume();
    //         })();
    //     })
    // );

    // Resolve an array of objects.
    deferred.resolve([]);

    return deferred.promise;
};
