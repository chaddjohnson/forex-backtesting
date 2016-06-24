#ifndef CALLPOSITION_H
#define CALLPOSITION_H

#include "position.cuh"

class CallPosition : public Position {
    public:
        __device__ __host__ CallPosition() : Position() {}
        __device__ __host__ CallPosition(const char *symbol, int timestamp, float price, float investment, float profitability, int expirationMinutes)
            : Position(symbol, timestamp, price, investment, profitability, expirationMinutes) {}
        __device__ __host__ ~CallPosition() {}
        __device__ __host__ float getProfitLoss();
};

#endif
