#ifndef CALLPOSITION_H
#define CALLPOSITION_H

#include "position.cuh"

class CallPosition : public Position {
    protected:
        __device__ const char *getTransactionType();

    public:
        __device__ CallPosition(const char *symbol, time_t timestamp, double price, double investment, double profitability, int expirationMinutes)
            : Position(symbol, timestamp, price, investment, profitability, expirationMinutes) {}
        __device__ ~CallPosition() {}
        __device__ double getProfitLoss();
};

#endif
