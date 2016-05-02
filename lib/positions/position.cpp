#include "positions/position.h"

Position::Position(std::string symbol, time_t timestamp, double price, double investment, double profitability, int expirationMinutes) {
    this->symbol = symbol;
    this->timestamp = timestamp;
    this->price = price;
    this->investment = investment;
    this->profitability = profitability;
    this->closePrice = 0.0;
    this->isOpen = true;

    // TODO
    this->expirationTimestamp = this->timestamp + (expirationMinutes * 1000 * 60);
}

std::string Position::getSymbol() {
    return this->symbol;
}

time_t Position::getTimestamp() {
    return this->timestamp;
}

double Position::getPrice() {
    return this->price;
}

double Position::getClosePrice() {
    return this->closePrice;
}

double Position::getInvestment() {
    return this->investment;
}

double Position::getProfitability() {
    return this->profitability;
}

time_t Position::getCloseTimestamp() {
    return this->closeTimestamp;
}

time_t Position::getExpirationTimestamp() {
    return this->expirationTimestamp;
}

bool Position::getIsOpen() {
    return this->isOpen;
}

bool Position::getHasExpired() {
    // TODO
    return this->timestamp >= this->expirationTimestamp;
}

void Position::close(double price, time_t timestamp) {
    this->isOpen = false;
    this->closePrice = price;
    this->closeTimestamp = timestamp;
}
