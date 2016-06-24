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
        float profitLoss;
        int winCount;
        int loseCount;
        int consecutiveLosses;
        int maximumConsecutiveLosses;
        float minimumProfitLoss;
        float previousClose;

    protected:
        __device__ __host__ void tick(float close, int timestamp);
        __device__ __host__ float getWinRate();
        __device__ __host__ float getProfitLoss();
        __device__ __host__ void closeExpiredPutPositions(float price, int timestamp);
        __device__ __host__ void closeExpiredCallPositions(float price, int timestamp);
        __device__ __host__ void addCallPosition(const char *symbol, int timestamp, float close, float investment, float profitability, float expirationMinutes);
        __device__ __host__ void addPutPosition(const char *symbol, int timestamp, float close, float investment, float profitability, float expirationMinutes);
        __device__ __host__ __forceinline__ float getPreviousClose() {
            return this->previousClose;
        }
        __device__ __host__ __forceinline__ void setPreviousClose(float close) {
            this->previousClose = close;
        }

    public:
        __device__ __host__ Strategy(const char *symbol);
        __device__ __host__ ~Strategy() {}
        __device__ __host__ void backtest(float *dataPoint, float investment, float profitability) {}
        __device__ __host__ const char *getSymbol();
        __device__ __host__ void setProfitLoss(float profitLoss);
        __device__ __host__ StrategyResult getResult();
};

#endif
