#ifndef PUTPOSITION_H
#define PUTPOSITION_H

#include <string>
#include "position.cuh"

class PutPosition : public Position {
    protected:
        const char *getTransactionType();

    public:
        PutPosition(const char *symbol, time_t timestamp, double price, double investment, double profitability, int expirationMinutes)
            : Position(symbol, timestamp, price, investment, profitability, expirationMinutes) {}
        ~PutPosition() {}
        double getProfitLoss();
};

#endif
