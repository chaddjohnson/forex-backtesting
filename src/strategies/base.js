var fs = require('fs');

function Base() {
    this.studies = [];
    this.positions = [];
    this.profitLoss = 0.0;
    this.cumulativeData = [];
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

Base.prototype.tick = function(dataPoint) {
    var self = this;

    // Add the data point to the cumulative data.
    self.cumulativeData.push(dataPoint);

    // Iterate over each study...
    self.getStudies().forEach(function(study) {
        // Update the data for the strategy.
        study.setData(self.cumulativeData);

        // Augment the last data point with the data the study generates.
        dataPoint[study.getName()] = study.tick();
    });

    // Simulate expiry of and profit/loss related to positions held.
    self.closeExpiredPositions(dataPoint);
};

Base.prototype.backtest = function(data, investment, profitability) {
    throw 'backtest() not implemented.';
};

Base.prototype.addPosition = function(position) {
    this.positions.push(position);

    // Deduct the investment amount from the profit/loss for this strategy.
    this.profitLoss -= position.getInvestment();
};

Base.prototype.closeExpiredPositions = function(dataPoint) {
    var self = this;

    self.positions.forEach(function(position) {
        if (position.getIsOpen() && position.getHasExpired(dataPoint.timestamp)) {
            // Close the position since it is open and has expired.
            position.close(dataPoint);

            // Add the profit/loss for this position to the profit/loss for this strategy.
            self.profitLoss += position.getProfitLoss();
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
        stream.write(dataPoint.symbol + ',' + dataPoint.timestamp + ',' + dataPoint.volume + ',' + dataPoint.price);

        // Write data for studies.
        self.getStudies().forEach(function(study) {
            stream.write(',' + dataPoint[study.getName()]);
        });
        stream.write('\n');
    });
};

module.exports = Base;
