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
        __device__ const char *getTransactionType() {
            return "";
        }

    public:
        __device__ Position(const char *symbol, time_t timestamp, double price, double investment, double profitability, int expirationMinutes);
        __device__ ~Position() {}
        __device__ const char *getSymbol();
        __device__ time_t getTimestamp();
        __device__ double getPrice();
        __device__ double getClosePrice();
        __device__ double getInvestment();
        __device__ double getProfitability();
        __device__ time_t getCloseTimestamp();
        __device__ time_t getExpirationTimestamp();
        __device__ bool getIsOpen();
        __device__ bool getHasExpired(time_t timestamp);
        __device__ void close(double price, time_t timestamp);
        __device__ double getProfitLoss() {
            return 0.0;
        }
};

#endif
