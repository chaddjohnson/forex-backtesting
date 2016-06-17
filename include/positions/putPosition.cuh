#ifndef PUTPOSITION_H
#define PUTPOSITION_H

#include "position.cuh"

class PutPosition : public Position {
    protected:
        __device__ __host__ const char *getTransactionType();

    public:
        __device__ __host__ PutPosition(const char *symbol, int timestamp, double price, double investment, double profitability, int expirationMinutes)
            : Position(symbol, timestamp, price, investment, profitability, expirationMinutes) {}
        __device__ __host__ ~PutPosition() {}
        __device__ __host__ double getProfitLoss();
};

#endif
