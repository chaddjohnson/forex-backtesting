#include "positions/position.cuh"

__device__ Position::Position(const char *symbol, time_t timestamp, double price, double investment, double profitability, int expirationMinutes) {
    this->symbol = symbol;
    this->timestamp = timestamp;
    this->price = price;
    this->investment = investment;
    this->profitability = profitability;
    this->closePrice = 0.0;
    this->isOpen = true;
    this->expirationTimestamp = this->timestamp + (expirationMinutes * 60);
}

__device__ const char *Position::getSymbol() {
    return this->symbol;
}

__device__ time_t Position::getTimestamp() {
    return this->timestamp;
}

__device__ double Position::getPrice() {
    return this->price;
}

__device__ double Position::getClosePrice() {
    return this->closePrice;
}

__device__ double Position::getInvestment() {
    return this->investment;
}

__device__ double Position::getProfitability() {
    return this->profitability;
}

__device__ time_t Position::getCloseTimestamp() {
    return this->closeTimestamp;
}

__device__ time_t Position::getExpirationTimestamp() {
    return this->expirationTimestamp;
}

__device__ bool Position::getIsOpen() {
    return this->isOpen;
}

__device__ bool Position::getHasExpired(time_t timestamp) {
    return timestamp >= this->expirationTimestamp;
}

__device__ void Position::close(double price, time_t timestamp) {
    this->isOpen = false;
    this->closePrice = price;
    this->closeTimestamp = timestamp;
}
