#include "positions/putPosition.cuh"

__device__ __host__ float PutPosition::getProfitLoss() {
    if (getIsOpen()) {
        return 0.0;
    }

    if (getCloseTimestamp() != getExpirationTimestamp()) {
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
