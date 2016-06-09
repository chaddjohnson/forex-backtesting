#ifndef PUTPOSITION_H
#define PUTPOSITION_H

#include "position.cuh"
#include "types/real.cuh"

class PutPosition : public Position {
    protected:
        __device__ __host__ const char *getTransactionType();

    public:
        __device__ __host__ PutPosition(const char *symbol, Real timestamp, Real price, Real investment, Real profitability, int expirationMinutes)
            : Position(symbol, timestamp, price, investment, profitability, expirationMinutes) {}
        __device__ __host__ ~PutPosition() {}
        __device__ __host__ Real getProfitLoss();
};

#endif
