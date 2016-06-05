#ifndef POSITION_H
#define POSITION_H

class Position {
    private:
        const char *symbol;
        double timestamp;
        double price;
        double investment;
        double profitability;
        double closePrice;
        bool isOpen;
        double closeTimestamp;
        double expirationTimestamp;

    protected:
        __device__ __host__ const char *getTransactionType() {
            return "";
        }

    public:
        __device__ __host__ Position(const char *symbol, double timestamp, double price, double investment, double profitability, int expirationMinutes);
        __device__ __host__ virtual ~Position() {};
        __device__ __host__ const char *getSymbol();
        __device__ __host__ double getTimestamp();
        __device__ __host__ double getPrice();
        __device__ __host__ double getClosePrice();
        __device__ __host__ double getInvestment();
        __device__ __host__ double getProfitability();
        __device__ __host__ double getCloseTimestamp();
        __device__ __host__ double getExpirationTimestamp();
        __device__ __host__ bool getIsOpen();
        __device__ __host__ bool getHasExpired(double timestamp);
        __device__ __host__ void close(double price, double timestamp);
        __device__ __host__ virtual double getProfitLoss() = 0;
};

#endif
