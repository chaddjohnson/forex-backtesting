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
        self.studies.push(new studyDefinition.study(studyDefinition.name, studyDefinition.inputs));
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
        var studyTickValue = 0.0;

        // Update the data for the strategy.
        study.setData(self.cumulativeData);

        studyTickValue = study.tick();

        if (studyTickValue) {
            studyTickValue = studyTickValue.toFixed(4);
        }

        // Augment the last data point with the data the study generates.
        dataPoint[study.getName()] = studyTickValue;
    });

    if (previousDataPoint) {
        // Simulate expiry of and profit/loss related to positions held.
        self.closeExpiredPositions(previousDataPoint.price, dataPoint.timestamp);
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
    stream.write('symbol,timestamp,volume,price');

    // Add study names to headers.
    self.getStudies().forEach(function(study) {
        stream.write(',' + study.getName());
    });
    stream.write('\n');

    // Write data.
    self.cumulativeData.forEach(function(dataPoint) {
        // Write base data.
        stream.write(dataPoint.symbol + ',' + new Date(dataPoint.timestamp) + ',' + dataPoint.volume + ',' + dataPoint.price);

        // Write data for studies.
        self.getStudies().forEach(function(study) {
            stream.write(',' + dataPoint[study.getName()]);
        });
        stream.write('\n');
    });
};

module.exports = Base;
