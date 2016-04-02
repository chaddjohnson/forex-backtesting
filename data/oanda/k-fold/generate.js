var fs = require('fs');
var es = require('event-stream');

var stream;
var transactionData = [];
var date;
var currentDay = -1;
var previousDay = -1;
var week = 1;
var buffer = [];

stream = fs.createReadStream('./original/' + process.argv[2] + '.csv')
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
            date = new Date(transactionData[0]);

            currentDay = date.getDay();

            if (previousDay != -1 && currentDay < previousDay) {
                fs.writeFileSync('./' + week + '.csv', buffer.join('\n'));
                buffer = [];
                week++;
            }

            previousDay = currentDay;
            buffer.push(line);

            // Resume the read stream.
            stream.resume();
        })();
    }));

stream.on('close', function() {
    process.exit();
});
