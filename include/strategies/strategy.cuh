#ifndef STRATEGY_H
#define STRATEGY_H

#include "positions/position.cuh"
#include "positions/callPosition.cuh"
#include "positions/putPosition.cuh"
#include "types/configuration.cuh"
#include "types/strategyResult.cuh"
#include "types/real.cuh"

class Strategy {
    private:
        const char *symbol;
        Position *openPositions[10];
        Real profitLoss;
        int winCount;
        int loseCount;
        int consecutiveLosses;
        int maximumConsecutiveLosses;
        Real minimumProfitLoss;
        Real previousClose;
        Real previousTimestamp;

    protected:
        __device__ __host__ void tick(Real *dataPoint);
        __device__ __host__ Real getWinRate();
        __device__ __host__ Real getProfitLoss();
        __device__ __host__ void closeExpiredPositions(Real price, Real timestamp);
        __device__ __host__ void addPosition(Position *position);
        __device__ __host__ __forceinline__ Real getPreviousClose() {
            return this->previousClose;
        }
        __device__ __host__ __forceinline__ Real getPreviousTimestamp() {
            return this->previousTimestamp;
        }
        __device__ __host__ __forceinline__ void setPreviousClose(Real close) {
            this->previousClose = close;
        }
        __device__ __host__ __forceinline__ void setPreviousTimestamp(Real timestamp) {
            this->previousTimestamp = timestamp;
        }

    public:
        __device__ __host__ Strategy(const char *symbol);
        __device__ __host__ ~Strategy() {}
        __device__ __host__ void backtest(Real *dataPoint, Real investment, Real profitability) {}
        __device__ __host__ const char *getSymbol();
        __device__ __host__ void setProfitLoss(Real profitLoss);
        __device__ __host__ StrategyResult getResult();
};

#endif
