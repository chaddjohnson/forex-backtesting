var Base = require('./base');

function Call(symbol, timestamp, price, investment, profitability, expirationMinutes) {
    this.constructor = Call;
    Base.call(this, symbol, timestamp, price, investment, profitability, expirationMinutes);
}

// Create a copy of the Base "class" prototype for use in this "class."
Call.prototype = Object.create(Base.prototype);

Call.prototype.getProfitLoss = function() {
    if (this.getIsOpen()) {
        return 0.0;
    }

    // Disregard transaction (set it to 0 profit/loss) if we don't have a good data point close to the expiry time available.
    if (this.getCloseTimestamp() > this.getExpirationTimestamp()) {
        return this.getInvestment();
    }

    // A win occurs if the closing price is above the purchase price.
    if (this.getClosePrice() > this.getPrice()) {
        return this.getInvestment() + (this.getProfitability() * this.getInvestment());
    }
    // A draw occurs if the closing price is the same as the purchase price.
    else if (this.getClosePrice() === this.getPrice()) {
        return this.getInvestment();
    }
    else {
        return 0;
    }
};

module.exports = Call;
