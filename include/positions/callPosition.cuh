#ifndef CALLPOSITION_H
#define CALLPOSITION_H

#include "position.cuh"
#include "types/real.cuh"

class CallPosition : public Position {
    protected:
        __device__ __host__ const char *getTransactionType();

    public:
        __device__ __host__ CallPosition(const char *symbol, Real timestamp, Real price, Real investment, Real profitability, int expirationMinutes)
            : Position(symbol, timestamp, price, investment, profitability, expirationMinutes) {}
        __device__ __host__ ~CallPosition() {}
        __device__ __host__ Real getProfitLoss();
};

#endif
