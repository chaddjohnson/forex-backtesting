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
        Position *openPositions[10];
        double profitLoss;
        int winCount;
        int loseCount;
        int consecutiveLosses;
        int maximumConsecutiveLosses;
        double minimumProfitLoss;
        double previousClose;

    protected:
        __device__ __host__ void tick(double *dataPoint, double close, int timestamp);
        __device__ __host__ double getWinRate();
        __device__ __host__ double getProfitLoss();
        __device__ __host__ void closeExpiredPositions(double price, int timestamp);
        __device__ __host__ void addPosition(Position *position);
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
