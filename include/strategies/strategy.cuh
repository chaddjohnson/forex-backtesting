#ifndef STRATEGY_H
#define STRATEGY_H

#include "positions/position.cuh"
#include "positions/callPosition.cuh"
#include "positions/putPosition.cuh"
#include "types/configuration.cuh"
#include "types/strategyResult.cuh"

class Strategy {
    private:
        const char *symbol;
        PutPosition openPutPositions[5];
        CallPosition openCallPositions[5];
        double profitLoss;
        int winCount;
        int loseCount;
        int consecutiveLosses;
        int maximumConsecutiveLosses;
        double minimumProfitLoss;
        double previousClose;

    protected:
        __device__ __host__ void tick(double close, int timestamp);
        __device__ __host__ double getWinRate();
        __device__ __host__ double getProfitLoss();
        __device__ __host__ void closeExpiredPutPositions(double price, int timestamp);
        __device__ __host__ void closeExpiredCallPositions(double price, int timestamp);
        __device__ __host__ void addCallPosition(const char *symbol, int timestamp, double close, double investment, double profitability, double expirationMinutes);
        __device__ __host__ void addPutPosition(const char *symbol, int timestamp, double close, double investment, double profitability, double expirationMinutes);
        __device__ __host__ __forceinline__ double getPreviousClose() {
            return this->previousClose;
        }
        __device__ __host__ __forceinline__ void setPreviousClose(double close) {
            this->previousClose = close;
        }

    public:
        __device__ __host__ Strategy(const char *symbol);
        __device__ __host__ ~Strategy() {}
        __device__ __host__ void backtest(double *dataPoint, double investment, double profitability) {}
        __device__ __host__ const char *getSymbol();
        __device__ __host__ void setProfitLoss(double profitLoss);
        __device__ __host__ StrategyResult getResult();
};

#endif
