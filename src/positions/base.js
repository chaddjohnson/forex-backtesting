function Base(symbol, timestamp, price, investment, profitability, expirationMinutes) {
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

    //console.log('New position created:');
    //console.log('    Investment:\t$' + this.investment);
    //console.log('    Timestamp:\t' + new Date(this.timestamp));
    //console.log('    Symbol:\t' + symbol);
    //console.log('    Time:\t' + new Date(timestamp));
    //console.log('    Price:\t$' + price);
    //console.log('    Expires:\t' + new Date(this.expirationTimestamp));
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

Base.prototype.getExpirationTimestamp = function() {
    return this.expirationTimestamp;
};

Base.prototype.getIsOpen = function() {
    return this.isOpen;
};

Base.prototype.getHasExpired = function(timestamp) {
    // The option has expired if the current data point's timestamp is after the expiration timestamp.
    return timestamp >= this.expirationTimestamp;
};

Base.prototype.close = function(price, timestamp) {
    // Mark this position as closed, and record the closing price.
    this.isOpen = false;
    this.closePrice = price;
    this.closeTimestamp = timestamp;

    //console.log('\n    Position ' + this.symbol + ' closed for $' + this.getProfitLoss() + ' profit/loss at ' + new Date(timestamp) + '\n');
};

Base.prototype.getProfitLoss = function() {
    throw 'getProfitLoss() not implemented.'
};

module.exports = Base;
