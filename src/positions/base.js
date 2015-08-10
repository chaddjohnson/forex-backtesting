function Base(symbol, timestamp, price, investment, profitability) {
    this.constructor = Base;

    this.symbol = symbol;
    this.timestamp = timestamp;
    this.price = price;
    this.investment = investment;
    this.profitability = profitability;
    this.closePrice = 0.0;
    this.profitLoss = 0.0;
    this.isOpen = true;

    console.log('New position created:');
    console.log('    Symbol:\t' + symbol);
    console.log('    Time:\t' + timestamp);
    console.log('    Price:\t$' + price);
}

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

Base.prototype.isOpen = function() {
    return this.isOpen;
};

Base.prototype.isExpired = function(dataPoint) {
    var isExpired = false;

    // Determine if this position has expired.
    // ...

    if (isExpired) {
        console.log('Position ' + this.symbol + ' is expired');
    }
};

Base.prototype.close = function(dataPoint) {
    // Mark this position as closed, and record the closing price.
    this.isOpen = false;
    this.closePrice = datapoint.bid;

    console.log('Position ' + this.symbol + ' closed for $' + this.profitLoss + ' profit/loss');
};

Base.prototype.getProfitLoss = function() {
    throw 'getProfitLoss() not implemented.'
};

module.exports = Base;
