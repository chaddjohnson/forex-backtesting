var StrategyBase = require('../Base');

function Base(symbol, configurations) {
    this.constructor = Base;
    StrategyBase.call(this, symbol, configurations);

    this.configurations = configurations;
}

// Create a copy of the Base "class" prototype for use in this "class."
Base.prototype = Object.create(StrategyBase.prototype);

Base.prototype.tick = function(dataPoint) {
    var self = this;
    var i = 0;

    // Add the data point to the cumulative data.
    self.cumulativeData.push(dataPoint);
    self.cumulativeDataCount++;

    // If there is a gap in the data, reset the cumulative data.
    if (self.previousDataPoint && (dataPoint.timestamp - self.previousDataPoint.timestamp) > 600000) {
        self.cumulativeData = [];
        self.cumulativeDataCount = 0;
    }

    // Iterate over each study...
    self.getStudies().forEach(function(study) {
        var studyProperty = '';
        var studyTickValues = {};
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

    if (self.previousDataPoint) {
        // Simulate expiry of and profit/loss related to positions held.
        self.closeExpiredPositions(self.previousDataPoint.close, dataPoint.timestamp - 1000);
    }

    self.previousDataPoint = dataPoint;

    // Remove unused data every so often.
    if (self.cumulativeDataCount >= 1500) {
        // Manually free memory for old data points in the array.
        for (i = 0; i < 500; i++) {
            self.cumulativeData[i] = null;
        }

        // Remove the excess data points from the array.
        self.cumulativeData.splice(0, 500);
        self.cumulativeDataCount = 1000;
    }
};

Base.prototype.getConfigurations = function() {
    return this.configurations;
};

module.exports = Base;
