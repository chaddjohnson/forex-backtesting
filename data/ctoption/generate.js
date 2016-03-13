var fs = require('fs');
var es = require('event-stream');
var _ = require('lodash');

var symbol = process.argv[2];
var filePath = process.argv[3];
var outputFilePath = process.argv[4];
var stream;
var minuteData = [];
var currentMinute = -1;
var previousMinute = -1;

try {
    fs.statSync(outputFilePath);
    fs.unlinkSync(outputFilePath);
}
catch (error) {}

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

            var data = JSON.parse(line);
            var closeDate;
            var closeTimestamp = 0;

            if (data.symbol !== symbol) {
                stream.resume();
                return;
            }
            if ('second' in data) {
                stream.resume();
                return;
            }

            currentMinute = new Date(data.timestamp).getMinutes();

            if (currentMinute !== previousMinute && previousMinute !== -1 && minuteData.length > 0) {
                closeTimestamp = _.last(minuteData).timestamp;
                closeDate = new Date(closeTimestamp);

                // Get the base second.
                closeTimestamp = closeTimestamp - (1000 * closeDate.getSeconds());

                // Remove microseconds.
                closeTimestamp = parseInt(closeTimestamp.toString().replace(/([0-9]{10})[0-9]{3}/, '$1000'))

                // Add 59 seconds.
                closeTimestamp = closeTimestamp + (1000 * 59);

                // Rectify time zone difference.
                closeTimestamp = closeTimestamp - (1000 * 60 * 60 * 2);

                fs.appendFileSync(outputFilePath, closeTimestamp + ',' + _.first(minuteData).price + ',' + _.max(minuteData, 'price').price + ',' + _.min(minuteData, 'price').price + ',' + _.last(minuteData).price + '\n');
                minuteData = [];
            }

            previousMinute = currentMinute;
            minuteData.push(data);

            // Resume the read stream.
            stream.resume();
        })();
    }));

stream.on('close', function() {
    process.exit(0);
});
