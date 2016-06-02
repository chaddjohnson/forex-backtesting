#ifndef STRATEGY_H
#define STRATEGY_H

#include "positions/position.cuh"
#include "types/configuration.cuh"
#include "types/strategyResults.cuh"
#include "types/basicDataIndexMap.cuh"

class Strategy {
    private:
        const char *symbol;
        BasicDataIndexMap dataIndexMap;
        Position *openPositions[10];
        double profitLoss;
        int winCount;
        int loseCount;
        int consecutiveLosses;
        int maximumConsecutiveLosses;
        double minimumProfitLoss;

    protected:
        __device__ __host__ BasicDataIndexMap getDataIndexMap();
        __device__ __host__ void tick(double *dataPoint) {}
        __device__ __host__ double getWinRate();
        __device__ __host__ double getProfitLoss();
        __device__ __host__ void closeExpiredPositions(double price, time_t timestamp);
        __device__ __host__ void addPosition(Position *position);

    public:
        __device__ __host__ Strategy(const char *symbol, BasicDataIndexMap dataIndexMap);
        __device__ __host__ ~Strategy() {}
        __device__ __host__ void backtest(double *dataPoint, double investment, double profitability) {}
        __device__ __host__ const char *getSymbol();
        __device__ __host__ void setProfitLoss(double profitLoss);
        __device__ __host__ StrategyResults getResults();
};

#endif
