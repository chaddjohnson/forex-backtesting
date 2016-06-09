#include "positions/position.cuh"

__device__ __host__ Position::Position(const char *symbol, Real timestamp, Real price, Real investment, Real profitability, int expirationMinutes) {
    this->symbol = symbol;
    this->timestamp = timestamp;
    this->price = price;
    this->investment = investment;
    this->profitability = profitability;
    this->closePrice = 0.0;
    this->isOpen = true;
    this->expirationTimestamp = this->timestamp + (expirationMinutes * 60);
}

__device__ __host__ const char *Position::getSymbol() {
    return this->symbol;
}

__device__ __host__ Real Position::getTimestamp() {
    return this->timestamp;
}

__device__ __host__ Real Position::getPrice() {
    return this->price;
}

__device__ __host__ Real Position::getClosePrice() {
    return this->closePrice;
}

__device__ __host__ Real Position::getInvestment() {
    return this->investment;
}

__device__ __host__ Real Position::getProfitability() {
    return this->profitability;
}

__device__ __host__ Real Position::getCloseTimestamp() {
    return this->closeTimestamp;
}

__device__ __host__ Real Position::getExpirationTimestamp() {
    return this->expirationTimestamp;
}

__device__ __host__ bool Position::getIsOpen() {
    return this->isOpen;
}

__device__ __host__ bool Position::getHasExpired(Real timestamp) {
    return timestamp >= this->expirationTimestamp;
}

__device__ __host__ void Position::close(Real price, Real timestamp) {
    this->isOpen = false;
    this->closePrice = price;
    this->closeTimestamp = timestamp;
}
