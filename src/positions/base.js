function Base(symbol, timestamp, price, investment, profitability, expirationMinutes) {
    this.constructor = Base;

    this.symbol = symbol;
    this.timestamp = timestamp;
    this.price = price;
    this.investment = investment;
    this.profitability = profitability;
    this.closePrice = 0.0;
    this.isOpen = true;
    this.closeTimestamp = null;

    // Calculate the expiration time.
    this.expirationTimestamp = this.timestamp + (expirationMinutes * 1000 * 60);

    console.log('New position created:');
    console.log('    Timestamp:\t' + new Date(this.timestamp));
    console.log('    Symbol:\t' + symbol);
    console.log('    Time:\t' + timestamp);
    console.log('    Price:\t$' + price);
    console.log('    Expires: \t' + new Date(this.expirationTimestamp));
}

Base.prototype.getTimestamp = function() {
    return this.timestamp;
};

Base.prototype.getPrice = function() {
    return this.price;
};

Base.prototype.getClosePrice = function() {
    return this.closePrice;
};

Base.prototype.getInvestment = function() {
    return this.investment;
};

Base.prototype.getProfitability = function() {
    return this.profitability;
};

Base.prototype.getCloseTimestamp = function() {
    return this.closeTimestamp;
};

Base.prototype.isOpen = function() {
    return this.isOpen;
};

Base.prototype.hasExpired = function(dataPoint) {
    // The option has expired if the current data point's timestamp is after the expiration timestamp.
    if (dataPoint.timestamp >= this.expirationTimestamp) {
        console.log('Position ' + this.symbol + ' is expired');
        return true;
    }

    return false;
};

Base.prototype.close = function(dataPoint) {
    // Mark this position as closed, and record the closing price.
    this.isOpen = false;
    this.closePrice = dataPoint.price;
    this.closeTimestamp = dataPoint.timestamp;

    console.log('Position ' + this.symbol + ' closed for $' + this.getProfitLoss() + ' profit/loss at ' + new Date(dataPoint.timestamp));
};

Base.prototype.getProfitLoss = function() {
    throw 'getProfitLoss() not implemented.'
};

module.exports = Base;
