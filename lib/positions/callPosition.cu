#include "positions/callPosition.cuh"

__device__ __host__ const char *CallPosition::getTransactionType() {
    return "CALL";
}

__device__ __host__ Real CallPosition::getProfitLoss() {
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
