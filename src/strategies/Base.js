var fs = require('fs');
var workerFarm = require('worker-farm');
var workers = workerFarm(require.resolve('../studyCalculator'));

function Base(studyDefinitions) {
    this.studyDefinitions = studyDefinitions;
    this.positions = [];
    this.openPositions = [];
    this.profitLoss = 0.0;
    this.cumulativeData = [];
    this.cumulativeDataCount = 0;
    this.winCount = 0;
    this.loseCount = 0;
}

Base.prototype.getStudyDefinitions = function() {
    return this.studyDefinitions;
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

Base.prototype.tick = function(dataPoint, callback) {
    var self = this;
    var studyCount = self.getStudyDefinitions().length;
    var completedWorkerCount = 0;

    // Add the data point to the cumulative data.
    self.cumulativeData.push(dataPoint);
    self.cumulativeDataCount++;

    // Iterate over each study...
    self.getStudyDefinitions().forEach(function(studyDefinition) {
        var workerInput = {
            data: self.cumulativeData,
            name: studyDefinition.name,
            inputs: studyDefinition.inputs,
            outputMap: studyDefinition.outputMap
        };
        workers(workerInput, function(error, studyTickValues) {
            console.log('callback called');
            var studyProperty = '';
            var studyTickValue = 0.0;

            // Augment the last data point with the data the study generates.
            for (studyProperty in studyDefinition.outputMap) {
                studyTickValue = studyTickValues[studyDefinition.outputMap[studyProperty]];

                if (studyTickValues && typeof studyTickValue === 'number') {
                    // Include output in main output, and limit decimal precision without rounding.
                    dataPoint[studyDefinition.outputMap[studyProperty]] = studyTickValue;
                }
                else {
                    dataPoint[studyDefinition.outputMap[studyProperty]] = '';
                }
            }

            completedWorkerCount++;

            if (completedWorkerCount >= studyCount) {
                // Simulate expiry of and profit/loss related to positions held.
                self.closeExpiredPositions(dataPoint.open, dataPoint.timestamp);

                // Remove unused data every so often.
                if (self.cumulativeDataCount >= 2000) {
                    self.cumulativeData.splice(0, 1000);
                    self.cumulativeDataCount = 1000;
                }

                // Done with the current tick for this strategy.
                callback();
            }
        });
    });
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

module.exports = Base;
