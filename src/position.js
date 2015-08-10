function Position(symbol, timestamp, price) {
    this.constructor = Position;

    this.symbol = symbol;
    this.timestamp = timestamp;
    this.price = price;
    this.profitLoss = 0.0;
    this.isOpen = true;

    console.log('New position created:');
    console.log('    Symbol:\t' + symbol);
    console.log('    Time:\t' + timestamp);
    console.log('    Price:\t$' + price);
}

Position.prototype.isOpen = function() {
    return this.isOpen;
};

Position.prototype.isExpired = function(dataPoint) {
    var isExpired = false;

    // ...

    // Use bid price

    if (isExpired) {
        console.log('Position ' + this.symbol + ' is expired');
    }
};

Position.prototype.close = function(dataPoint) {
    // Disregard transaction (set it to 0 profit/loss) if we don't have a good data point close to the expiry time available.
    // ...

    // Use bid price

    // ...

    this.isOpen = false;

    console.log('Position ' + this.symbol + ' closed for $' + this.profitLoss + ' profit/loss');
};

Position.prototype.getProfitLoss = function() {
    // return ...
};

module.exports = Position;
