#ifndef CALLPOSITION_H
#define CALLPOSITION_H

#include <string>
#include "position.h"

class CallPosition : public Position {
    protected:
        std::string getTransactionType();

    public:
        CallPosition(std::string symbol, time_t timestamp, double price, double investment, double profitability, int expirationMinutes)
            : Position(symbol, timestamp, price, investment, profitability, expirationMinutes) {}
        double getProfitLoss();
};

#endif
