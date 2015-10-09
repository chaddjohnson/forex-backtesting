var studies = require('./studies');

module.exports = function(input, callback) {
    var study = new studies[input.name](input.inputs, input.outputMap);
    study.setData(input.data);
console.log('calling callback');
    callback(null, study.tick());
};
