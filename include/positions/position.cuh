#ifndef POSITION_H
#define POSITION_H

class Position {
    private:
        const char *symbol;
        int timestamp;
        double price;
        double investment;
        double profitability;
        double closePrice;
        bool isOpen;
        int closeTimestamp;
        int expirationTimestamp;

    protected:
        __device__ __host__ const char *getTransactionType() {
            return "";
        }

    public:
        __device__ __host__ Position(const char *symbol, int timestamp, double price, double investment, double profitability, int expirationMinutes);
        __device__ __host__ virtual ~Position() {};
        __device__ __host__ const char *getSymbol();
        __device__ __host__ int getTimestamp();
        __device__ __host__ double getPrice();
        __device__ __host__ double getClosePrice();
        __device__ __host__ double getInvestment();
        __device__ __host__ double getProfitability();
        __device__ __host__ int getCloseTimestamp();
        __device__ __host__ int getExpirationTimestamp();
        __device__ __host__ bool getIsOpen();
        __device__ __host__ bool getHasExpired(int timestamp);
        __device__ __host__ void close(double price, int timestamp);
        __device__ __host__ virtual double getProfitLoss() = 0;
};

#endif
