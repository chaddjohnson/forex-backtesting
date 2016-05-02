#ifndef CALLPOSITION_H
#define CALLPOSITION_H

#include <string>
#include "position.h"

class CallPosition : public Position {
    public:
        CallPosition::CallPosition(std::string symbol, time_t timestamp, double price, double investment, double profitability, int expirationMinutes)
            : Position(std::string symbol, time_t timestamp, double price, double investment, double profitability, int expirationMinutes) {}
};

#endif
