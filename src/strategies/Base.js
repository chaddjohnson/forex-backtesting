var fs = require('fs');

function Base() {
    this.studies = [];
    this.positions = [];
    this.openPositions = [];
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

Base.prototype.getWinCount = function() {
    return this.winCount;
};

Base.prototype.getLoseCount = function() {
    return this.loseCount;
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
            if (studyTickValues && typeof studyTickValues[studyOutputs[studyProperty]] === 'number') {
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

Base.prototype.getResults = function() {
    var consecutiveLosses = 0;
    var maximumConsecutiveLosses = 0;
    var minimumProfitLoss = 99999.0;
    var positionProfitLoss = 0;
    var balance = 0;

    // Determine the max consecutive losses.
    this.positions.forEach(function(position) {
        balance -= position.getInvestment();
        positionProfitLoss = position.getProfitLoss();

        if (positionProfitLoss === (position.getInvestment() + (position.getProfitability() * position.getInvestment()))) {
            // Won
            balance += (position.getInvestment() + (position.getProfitability() * position.getInvestment()));
            consecutiveLosses = 0;
        }
        else if (positionProfitLoss === position.getInvestment()) {
            // Broke even
            balance += position.getInvestment();
        }
        else {
            // Lost
            consecutiveLosses++;
        }

        // Track minimum profit/loss.
        if (balance < minimumProfitLoss) {
            minimumProfitLoss = balance;
        }

        // Track the maximum consecutive losses.
        if (consecutiveLosses > maximumConsecutiveLosses) {
            maximumConsecutiveLosses = consecutiveLosses;
        }
    });

    return {
        profitLoss: this.getProfitLoss(),
        winCount: this.getWinCount(),
        loseCount: this.getLoseCount(),
        winRate: this.getWinRate(),
        maximumConsecutiveLosses: maximumConsecutiveLosses,
        minimumProfitLoss: minimumProfitLoss
    };
};

Base.prototype.addPosition = function(position) {
    // Add this new position to the list of positions.
    this.positions.push(position);

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
            }
            if (profitLoss === 0) {
                self.loseCount++;
            }

            // Remove the position from the list of open positions.
            self.openPositions.splice(index, 1);
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
