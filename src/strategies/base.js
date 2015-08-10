function Base(data) {
    if (!data.length) {
        throw 'Empty data set provided to strategy.';
    }

    this.data = data;
    this.studies = [];
    this.positions = [];
    this.profitLoss = 0.0;
}

Base.prototype.getData = function() {
    return this.data;
};

Base.prototype.prepareStudies = function(studyDefinitions) {
    var self = this;

    // Iterate over each study definition...
    studyDefinitions.forEach(function(studyDefinition) {
        // Instantiate the study, and add it to the list of studies for this strategy.
        self.studies.push(new studyDefinition.study(studyDefinition.namet, self.getData(), studyDefinition.inputs));
    });
};

Base.prototype.getStudies = function() {
    return this.studies;
};

Base.prototype.tick = function() {
    var data = this.getData();
    var lastDataPoint = data[data.length - 1];

    // Iterate over each study...
    this.getStudies().forEach(function(study) {
        // Augment the last data point with the data the study generates.
        lastDataPoint[study.getName()] = study.tick();
    });

    // Simulate expiry of and profit/loss related to positions held.
    this.closeExpiredPositions(lastDataPoint);
};

Base.prototype.backtest = function(investment, profitability) {
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
        if (position.isOpen() && position.isExpired(dataPoint)) {
            // Close the position since it is open and has expired.
            position.close(dataPoint);

            // Add the profit/loss for this position to the profit/loss for this strategy.
            self.profitLoss += position.getProfitLoss();
        }
    });
};

module.exports = Base;
