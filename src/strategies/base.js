var fs = require('fs');

function Base() {
    this.studies = [];
    this.positions = [];
    this.profitLoss = 0.0;
    this.cumulativeData = [];
    this.winCount = 0;
    this.loseCount = 0;
}

Base.prototype.prepareStudies = function(studyDefinitions) {
    var self = this;

    // Iterate over each study definition...
    studyDefinitions.forEach(function(studyDefinition) {
        // Instantiate the study, and add it to the list of studies for this strategy.
        self.studies.push(new studyDefinition.study(studyDefinition.inputs, studyDefinition.outputMap));
    });
};

Base.prototype.getStudies = function() {
    return this.studies;
};

Base.prototype.getProfitLoss = function() {
    return this.profitLoss;
};

Base.prototype.getWinRate = function() {
    if (this.winCount + this.loseCount === 0) {
        return 0;
    }
    return this.winCount / (this.winCount + this.loseCount);
};

Base.prototype.tick = function(dataPoint) {
    var self = this;
    var previousDataPoint = self.cumulativeData[self.cumulativeData.length - 1];

    // Add the data point to the cumulative data.
    self.cumulativeData.push(dataPoint);

    // Iterate over each study...
    self.getStudies().forEach(function(study) {
        var studyProperty = '';
        var studyTickValue = 0.0;
        var studyOutputs = study.getOutputMappings();

        // Update the data for the strategy.
        study.setData(self.cumulativeData);

        studyTickValues = study.tick();

        // Augment the last data point with the data the study generates.
        for (studyProperty in studyOutputs) {
            if (studyTickValues && studyTickValues[studyOutputs[studyProperty]]) {
                // Include output in main output, and limit decimal precision without rounding.
                dataPoint[studyOutputs[studyProperty]] = studyTickValues[studyOutputs[studyProperty]];
            }
            else {
                dataPoint[studyOutputs[studyProperty]] = '';
            }
        }
    });

    if (previousDataPoint) {
        // Simulate expiry of and profit/loss related to positions held.
        self.closeExpiredPositions(previousDataPoint.close, dataPoint.timestamp);
    }
};

Base.prototype.backtest = function(data, investment, profitability) {
    throw 'backtest() not implemented.';
};

Base.prototype.addPosition = function(position) {
    this.positions.push(position);

    // Deduct the investment amount from the profit/loss for this strategy.
    this.profitLoss -= position.getInvestment();
};

Base.prototype.closeExpiredPositions = function(price, timestamp) {
    var self = this;

    self.positions.forEach(function(position) {
        var profitLoss = 0.0;

        if (position.getIsOpen() && position.getHasExpired(timestamp)) {
            // Close the position since it is open and has expired.
            position.close(price, timestamp);

            // Add the profit/loss for this position to the profit/loss for this strategy.
            profitLoss = position.getProfitLoss();
            self.profitLoss += profitLoss;

            if (profitLoss > position.getInvestment()) {
                self.winCount++;
            }
            if (profitLoss === 0) {
                self.loseCount++;
            }
        }
    });
};

Base.prototype.setDataOutputFilePath = function(path) {
    this.dataOutputFilePath = path;
};

Base.prototype.saveOutput = function() {
    var self = this;

    if (!self.dataOutputFilePath) {
        return;
    }

    // Save the data to a file.
    stream = fs.createWriteStream(self.dataOutputFilePath, {flags: 'w'});

    // Write headers for base data.
    stream.write('symbol,timestamp,volume,open,high,low,close');

    // Add study output names to headers.
    self.getStudies().forEach(function(study) {
        var studyProperty = '';
        var studyOutputs = study.getOutputMappings();

        for (studyProperty in studyOutputs) {
            stream.write(',' + studyOutputs[studyProperty]);
        }
    });
    stream.write('\n');

    // Write data.
    self.cumulativeData.forEach(function(dataPoint) {
        // Write base data.
        stream.write(dataPoint.symbol + ',' + new Date(dataPoint.timestamp) + ',' + dataPoint.volume + ',' + dataPoint.open + ',' + dataPoint.high + ',' + dataPoint.low + ',' + dataPoint.close);

        // Write data for studies.
        self.getStudies().forEach(function(study) {
            var studyProperty = '';
            var studyOutputs = study.getOutputMappings();

            for (studyProperty in studyOutputs) {
                stream.write(',' + dataPoint[studyOutputs[studyProperty]]);
            }
        });
        stream.write('\n');
    });
};

module.exports = Base;
