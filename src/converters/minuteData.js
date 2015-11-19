var fs = require('fs');
var es = require('event-stream');
var Q = require('q');
var _ = require('lodash');

function zeroPad(number, width) {
  number = number + '';
  return number.length >= width ? number : new Array(width - number.length + 1).join('0') + number;
}

function formatDate(date) {
    return date.getFullYear() + '-' + zeroPad(date.getMonth() + 1, 2) + '-' + zeroPad(date.getDate(), 2) + ' ' + zeroPad(date.getHours(), 2) + ':' + zeroPad(date.getMinutes(), 2) + ':00';
}

module.exports.convert = function(filePath) {
    var deferred = Q.defer();
    var stream;

    var index = 0;
    var transactionData = [];
    var convertedData = [];
    var previousDate = null;
    var date = null;
    var intraMinuteTicks = [];
    var previousConvertedData = null;

    if (!filePath) {
        throw 'No filePath provided to dataParser.'
    }

    stream = fs.createReadStream(filePath)
        .pipe(es.split())
        .pipe(es.mapSync(function(line) {
            // Pause the read stream.
            stream.pause();

            (function() {
                var convertedItem = {};
                var convertedItemCopy = {};

                // Ignore blank lines.
                if (!line) {
                    stream.resume();
                    return;
                }

                transactionData = line.split(',');
                date = new Date(parseInt(transactionData[1]));
                previousDate = previousDate || date;

                // Set the first second to the previous minute's close price if the current minute has no 0 second tick.
                if (intraMinuteTicks.length === 0 && date.getSeconds() !== 0 && convertedData.length > 0) {
                    previousConvertedData = convertedData[convertedData.length - 1];
                    intraMinuteTicks.push([previousConvertedData[4], new Date(formatDate(date)).getTime()]);
                }

                // Set the first transaction to that for the first second if the first transaction of the minute is at second 0.
                if (intraMinuteTicks.length === 0 && date.getSeconds() === 0) {
                    intraMinuteTicks.push(transactionData);
                }

                if (date.getMinutes() !== previousDate.getMinutes() && index > 0) {
                    convertedItem.date = formatDate(previousDate);
                    convertedItem.open = _.first(intraMinuteTicks)[0];
                    convertedItem.high = _.max(intraMinuteTicks, function(item) { return item[0]; })[0];
                    convertedItem.low = _.min(intraMinuteTicks, function(item) { return item[0]; })[0];
                    convertedItem.close = _.last(intraMinuteTicks)[0]

                    convertedData.push(_.values(convertedItem));

                    // Start over for next minute.
                    intraMinuteTicks = [];
                }
                else {
                    intraMinuteTicks.push(transactionData);
                }

                previousDate = date;
                index++;

                // Resume the read stream.
                stream.resume();
            })();
        }));

    stream.on('close', function() {
        deferred.resolve(convertedData);
    });

    return deferred.promise;
};
