#ifndef POSITION_H
#define POSITION_H

class Position {
    private:
        const char *symbol;
        int timestamp;
        float price;
        float investment;
        float profitability;
        float closePrice;
        bool isOpen;
        int closeTimestamp;
        int expirationTimestamp;
        int expirationMinutes;

    public:
        __device__ __host__ Position();
        __device__ __host__ Position(const char *symbol, int timestamp, float price, float investment, float profitability, int expirationMinutes);
        __device__ __host__ ~Position() {};
        __device__ __host__ const char *getSymbol();
        __device__ __host__ int getTimestamp();
        __device__ __host__ float getPrice();
        __device__ __host__ float getClosePrice();
        __device__ __host__ float getInvestment();
        __device__ __host__ float getProfitability();
        __device__ __host__ int getCloseTimestamp();
        __device__ __host__ int getExpirationTimestamp();
        __device__ __host__ int getExpirationMinutes();
        __device__ __host__ bool getIsOpen();
        __device__ __host__ bool getHasExpired(int timestamp);
        __device__ __host__ void close(float price, int timestamp);
        __device__ __host__ float getProfitLoss();
};

#endif
