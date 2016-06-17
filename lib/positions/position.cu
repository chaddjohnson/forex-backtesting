#include "positions/position.cuh"

__device__ __host__ Position::Position(const char *symbol, int timestamp, double price, double investment, double profitability, int expirationMinutes) {
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

__device__ __host__ int Position::getTimestamp() {
    return this->timestamp;
}

__device__ __host__ double Position::getPrice() {
    return this->price;
}

__device__ __host__ double Position::getClosePrice() {
    return this->closePrice;
}

__device__ __host__ double Position::getInvestment() {
    return this->investment;
}

__device__ __host__ double Position::getProfitability() {
    return this->profitability;
}

__device__ __host__ int Position::getCloseTimestamp() {
    return this->closeTimestamp;
}

__device__ __host__ int Position::getExpirationTimestamp() {
    return this->expirationTimestamp;
}

__device__ __host__ bool Position::getIsOpen() {
    return this->isOpen;
}

__device__ __host__ bool Position::getHasExpired(int timestamp) {
    return timestamp >= this->expirationTimestamp;
}

__device__ __host__ void Position::close(double price, int timestamp) {
    this->isOpen = false;
    this->closePrice = price;
    this->closeTimestamp = timestamp;
}
