#include "positions/putPosition.cuh"

__device__ __host__ const char *PutPosition::getTransactionType() {
    return "PUT";
}

__device__ __host__ double PutPosition::getProfitLoss() {
    if (getIsOpen()) {
        return 0.0;
    }

    if (getCloseTimestamp() > getExpirationTimestamp()) {
        return getInvestment();
    }

    if (getClosePrice() < getPrice()) {
        return getInvestment() + (getProfitability() * getInvestment());
    }
    else if (getClosePrice() == getPrice()) {
        return getInvestment();
    }

    return 0.0;
}
