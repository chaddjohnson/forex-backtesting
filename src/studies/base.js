function Base(name, data, inputs) {
    if (!data.length) {
        throw 'Empty data set provided to study.';
    }

    this.name = name;
    this.data = data;
    this.inputs = inputs;
}

Base.prototype.getName = function() {
    return this.name;
};

Base.prototype.getData = function() {
    return this.data;
};

Base.prototype.getInput = function(key) {
    return this.input[key];
};

Base.prototype.getDataSegment = function(length) {
    var data = this.getData();

    // Get only last n data points, where n is either the length provided as input or the
    // length of the array (whichever is smallest so as to not surpass the data array length).
    var dataSegmentLength = Math.min(length, data.length);

    return data.slice(0, dataSegmentLength);
};

Base.prototype.getPrevious = function() {
    var data = this.getData();
    return data[data.length - 2];
};

Base.prototype.getLast = function() {
    var data = this.getData();
    return data[data.length - 1];
};

Base.prototype.tick = function() {
    throw 'tick() not implemented.';
};

module.exports = Base;
