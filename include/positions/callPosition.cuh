#ifndef CALLPOSITION_H
#define CALLPOSITION_H

#include <string>
#include "position.cuh"

class CallPosition : public Position {
    protected:
        char *getTransactionType();

    public:
        CallPosition(char *symbol, time_t timestamp, double price, double investment, double profitability, int expirationMinutes)
            : Position(symbol, timestamp, price, investment, profitability, expirationMinutes) {}
        ~CallPosition() {}
        double getProfitLoss();
};

#endif
