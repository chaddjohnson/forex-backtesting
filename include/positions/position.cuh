#ifndef POSITION_H
#define POSITION_H

class Position {
    private:
        const char *symbol;
        time_t timestamp;
        double price;
        double investment;
        double profitability;
        double closePrice;
        bool isOpen;
        time_t closeTimestamp;
        time_t expirationTimestamp;

    protected:
        __device__ __host__ const char *getTransactionType() {
            return "";
        }

    public:
        __device__ __host__ Position(const char *symbol, time_t timestamp, double price, double investment, double profitability, int expirationMinutes);
        __device__ __host__ ~Position() {};
        __device__ __host__ const char *getSymbol();
        __device__ __host__ time_t getTimestamp();
        __device__ __host__ double getPrice();
        __device__ __host__ double getClosePrice();
        __device__ __host__ double getInvestment();
        __device__ __host__ double getProfitability();
        __device__ __host__ time_t getCloseTimestamp();
        __device__ __host__ time_t getExpirationTimestamp();
        __device__ __host__ bool getIsOpen();
        __device__ __host__ bool getHasExpired(time_t timestamp);
        __device__ __host__ void close(double price, time_t timestamp);
        __device__ __host__ virtual double getProfitLoss() = 0;
};

#endif
