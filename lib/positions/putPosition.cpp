#include "positions/putPosition.h"

std::string PutPosition::getTransactionType() {
    return "PUT";
}

double PutPosition::getProfitLoss() {
    if (getIsOpen()) {
        return 0.0;
    }

    if (getCloseTimestamp() > getExpirationTimestamp()) {
        return getInvestment();
    }

    if (getClosePrice() > getPrice()) {
        return getInvestment() + (getProfitability() * getInvestment());
    }
    else if (getClosePrice() == getPrice()) {
        return getInvestment();
    }
    else {
        return 0.0;
    }
}
