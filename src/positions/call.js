var Base = require('./base');

function Call(symbol, timestamp, price, investment, profitability) {
    this.constructor = Call;
    Base.call(this, symbol, timestamp, price, investment, profitability);
}

// Create a copy of the Base "class" prototype for use in this "class."
Call.prototype = Object.create(Base.prototype);

Call.prototype.getProfitLoss = function() {
    if (this.isOpen()) {
        return 0.0;
    }

    // Disregard transaction (set it to 0 profit/loss) if we don't have a good data point close to the expiry time available.
    // ...

    // A win occurs if the closing price is above the purchase price.
    if (this.getClosePrice() > this.getPrice()) {
        return investment + (this.getProfitability() * this.investment);
    }
    // A draw occurs if the closing price is the same as the purchase price.
    else if (this.getClosePrice() === this.getPrice()) {
        return investment;
    }
    else {
        return 0;
    }
};

module.exports = Call;
