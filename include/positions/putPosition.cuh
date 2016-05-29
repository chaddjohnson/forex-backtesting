#ifndef PUTPOSITION_H
#define PUTPOSITION_H

#include "position.cuh"

class PutPosition : public Position {
    protected:
        __device__ const char *getTransactionType();

    public:
        __device__ PutPosition(const char *symbol, time_t timestamp, double price, double investment, double profitability, int expirationMinutes)
            : Position(symbol, timestamp, price, investment, profitability, expirationMinutes) {}
        __device__ ~PutPosition() {}
        __device__ double getProfitLoss();
};

#endif
