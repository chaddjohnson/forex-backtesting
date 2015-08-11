function Base(data) {
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
    var lastDataPoint = this.cumulativeData[cumulativeData.length - 1];

    // Iterate over each study...
    this.getStudies().forEach(function(study) {
        // Update the data for the strategy.
        study.setData(cumulativeData);

        // Augment the last data point with the data the study generates.
        lastDataPoint[study.getName()] = study.tick();
    });

    // Simulate expiry of and profit/loss related to positions held.
    this.closeExpiredPositions(lastDataPoint);
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
        if (position.isOpen() && position.hasExpired(dataPoint)) {
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
    if (!this.dataOutputFilePath) {
        return;
    }

    // Save the data to a file.
    stream = fs.createWriteStream(this.dataOutputFilePath, {flags: 'w'});

    // Write headers for base data.
    stream.write('symbol,timestamp,price,volume');

    // Add study names to headers.
    this.getStudies().forEach(function(study) {
        stream.write(',' + study.getName());
    });
    stream.write('\n');

    // Write data.
    this.cumulativeData.forEach(function(dataPoint) {
        // Write base data.
        stream.write(dataPoint.symbol + ',' + dataPoint.timestamp + ',' + dataPoint.price + ',' + dataPoint.volume);

        // Write data for studies.
        this.getStudies().forEach(function(study) {
            stream.write(',' + dataPoint[study.getName()]);
        });
        stream.write('\n');
    });
};

module.exports = Base;
