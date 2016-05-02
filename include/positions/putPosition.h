#ifndef PUTPOSITION_H
#define PUTPOSITION_H

#include <string>
#include "position.h"

class PutPosition : public Position {
    public:
        PutPosition::PutPosition(std::string symbol, time_t timestamp, double price, double investment, double profitability, int expirationMinutes)
            : Position(std::string symbol, time_t timestamp, double price, double investment, double profitability, int expirationMinutes) {}
};

#endif
