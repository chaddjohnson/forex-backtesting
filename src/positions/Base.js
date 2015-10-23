function Base(symbol, timestamp, price, investment, profitability, expirationMinutes) {
    this.symbol = symbol;
    this.timestamp = timestamp;
    this.price = price;
    this.investment = investment;
    this.profitability = profitability;
    this.closePrice = 0.0;
    this.isOpen = true;
    this.closeTimestamp = null;
    this.showTrades = false;

    // Calculate the expiration time.
    this.expirationTimestamp = this.timestamp + (expirationMinutes * 1000 * 60);
}

Base.prototype.getSymbol = function() {
    return this.symbol;
};

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

Base.prototype.getTransactionType = function() {
    throw 'getTransactionType() not implemented.';
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

    if (this.showTrades) {
        console.log(this.getTransactionType());
        console.log('    Investment:\t\t$' + this.investment);
        console.log('    Symbol:\t\t' + this.symbol);
        console.log('    Time:\t\t' + new Date(this.timestamp));
        console.log('    Price:\t\t$' + this.price);
        console.log('    Expire time:\t' + new Date(this.expirationTimestamp));
        console.log('    Close time:\t\t' + new Date(this.closeTimestamp));
        console.log('    Close price:\t$' + this.closePrice);
        console.log('    Profit/loss:\t$' + this.getProfitLoss());
        console.log('');
    }
};

Base.prototype.getProfitLoss = function() {
    throw 'getProfitLoss() not implemented.'
};

Base.prototype.setShowTrades = function(showTrades) {
    this.showTrades = showTrades;
};

module.exports = Base;
