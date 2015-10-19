var fs = require('fs');

function Base(symbol) {
    this.symbol = symbol;
    this.studies = [];
    this.openPositions = [];
    this.profitLoss = 0.0;
    this.cumulativeData = [];
    this.cumulativeDataCount = 0;
    this.winCount = 0;
    this.loseCount = 0;
    this.showTrades = false;
    this.consecutiveLosses = 0;
    this.maximumConsecutiveLosses = 0;
    this.minimumProfitLoss = 99999;
}

Base.prototype.getSymbol = function() {
    return this.symbol;
};

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
    var i = 0;

    // Add the data point to the cumulative data.
    self.cumulativeData.push(dataPoint);
    self.cumulativeDataCount++;

    // Iterate over each study...
    self.getStudies().forEach(function(study) {
        var studyProperty = '';
        var studyTickValue = 0.0;
        var studyOutputs = study.getOutputMappings();

        // Update the data for the study.
        study.setData(self.cumulativeData);

        var studyTickValues = study.tick();

        // Augment the last data point with the data the study generates.
        for (studyProperty in studyOutputs) {
            if (studyTickValues && typeof studyTickValues[studyOutputs[studyProperty]] === 'number') {
                // Include output in main output, and limit decimal precision without rounding.
                dataPoint[studyOutputs[studyProperty]] = studyTickValues[studyOutputs[studyProperty]];
            }
            else {
                dataPoint[studyOutputs[studyProperty]] = '';
            }
        }
    });

    // Simulate expiry of and profit/loss related to positions held.
    self.closeExpiredPositions(dataPoint.open, dataPoint.timestamp);

    // Remove unused data every so often.
    if (self.cumulativeDataCount >= 2000) {
        // Manually free memory for old data points in the array.
        for (i = 0; i < 1000; i++) {
            self.cumulativeData[i] = null;
        }

        // Remove the excess data points from the array.
        self.cumulativeData.splice(0, 1000);
        self.cumulativeDataCount = 1000;
    }
};

Base.prototype.backtest = function(data, investment, profitability) {
    throw 'backtest() not implemented.';
};

Base.prototype.getResults = function() {
    return {
        profitLoss: this.profitLoss,
        winCount: this.winCount,
        loseCount: this.loseCount,
        winRate: this.getWinRate(),
        maximumConsecutiveLosses: this.maximumConsecutiveLosses,
        minimumProfitLoss: this.minimumProfitLoss
    };
};

Base.prototype.addPosition = function(position) {
    // Also track this position in the list of open positions.
    this.openPositions.push(position);

    // Deduct the investment amount from the profit/loss for this strategy.
    this.profitLoss -= position.getInvestment();
};

Base.prototype.closeExpiredPositions = function(price, timestamp) {
    var self = this;

    // Use a copy so that items can be removed from the original without messing up the loop.
    var openPositionsCopy = self.openPositions.slice();

    // Iterate over open positions.
    openPositionsCopy.forEach(function(position, index) {
        var profitLoss = 0.0;

        if (position.getHasExpired(timestamp)) {
            // Close the position since it is open and has expired.
            position.close(price, timestamp);

            // Add the profit/loss for this position to the profit/loss for this strategy.
            profitLoss = position.getProfitLoss();
            self.profitLoss += profitLoss;

            if (profitLoss > position.getInvestment()) {
                self.winCount++;
                self.consecutiveLosses = 0;
            }
            if (profitLoss === 0) {
                self.loseCount++;
                self.consecutiveLosses++;
            }

            // Track minimum profit/loss.
            if (profitLoss < self.minimumProfitLoss) {
                self.minimumProfitLoss = profitLoss;
            }

            // Track the maximum consecutive losses.
            if (self.consecutiveLosses > self.maximumConsecutiveLosses) {
                self.maximumConsecutiveLosses = self.consecutiveLosses;
            }

            // Remove the position from the list of open positions.
            self.openPositions.splice(index, 1);
        }
    });
};

Base.prototype.setShowTrades = function(showTrades) {
    this.showTrades = showTrades;
};

Base.prototype.getShowTrades = function() {
    return this.showTrades;
};

module.exports = Base;
