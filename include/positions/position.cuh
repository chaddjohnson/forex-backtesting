#ifndef POSITION_H
#define POSITION_H

#include "types/real.cuh"

class Position {
    private:
        const char *symbol;
        Real timestamp;
        Real price;
        Real investment;
        Real profitability;
        Real closePrice;
        bool isOpen;
        Real closeTimestamp;
        Real expirationTimestamp;

    protected:
        __device__ __host__ const char *getTransactionType() {
            return "";
        }

    public:
        __device__ __host__ Position(const char *symbol, Real timestamp, Real price, Real investment, Real profitability, int expirationMinutes);
        __device__ __host__ virtual ~Position() {};
        __device__ __host__ const char *getSymbol();
        __device__ __host__ Real getTimestamp();
        __device__ __host__ Real getPrice();
        __device__ __host__ Real getClosePrice();
        __device__ __host__ Real getInvestment();
        __device__ __host__ Real getProfitability();
        __device__ __host__ Real getCloseTimestamp();
        __device__ __host__ Real getExpirationTimestamp();
        __device__ __host__ bool getIsOpen();
        __device__ __host__ bool getHasExpired(Real timestamp);
        __device__ __host__ void close(Real price, Real timestamp);
        __device__ __host__ virtual Real getProfitLoss() = 0;
};

#endif
