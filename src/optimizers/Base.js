var Optimization = require('./models/optimization');

function Base(callback, data, investment, profitability) {
    this.callback = callback;
    this.data = data;
    this.investment = investment;
    this.profitability = profitability;
}

Base.prototype.prepareStudies = function(studyDefinitions) {
    // ...
};

Base.prototype.prepareStudyData = function(data) {
    // ...
};

Base.prototype.buildConfigurations = function(combinations) {
    var configurations = [];
    var configuration = {};
    var optionKeys = Object.keys(options);
    var optionKeysCount = optionKeys.length;
    var optionValues;
    var optionValuesCount = optionValues.length;
    var innerOptions;
    var innerOptionsCount = innerOptions.length;
    var i = 0;
    var j = 0;
    var k = 0;
    var l = 0;

    // Iterate over the options.
    for (i = 0; i < optionKeysCount; i++) {
        // Get the values for the current option.
        optionValues = options[optionKeys[i]];

        // Iterate through values for the current configuration...
        for (j = 0; j < optionValuesCount; j++) {
            // Iterate through the options after the current one.
            for (k = i; k < ; k++) {
                innerOptions = ...

                for (l = 0; l < innerOptionsCountl l++) {
                    configuration = {};
                    configuration[optionKeys[i]] = optionValues[j];
                    configuration[...] = ...;

                    configurations.push(configuration);
                }
            }
        }
    }

    return configurations;
};

Base.prototype.optimize = function() {
    throw 'optimize() not implemented.';
};

module.exports = Base;
