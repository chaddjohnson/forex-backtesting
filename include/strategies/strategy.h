#ifndef STRATEGY_H
#define STRATEGY_H

#include <vector>
#include <string>
#include <map>
#include "positions/position.h"
#include "types/configuration.h"

class Strategy {
    private:
        std::string symbol;
        std::vector<Position*> openPositions;
        double profitLoss;
        int winCount;
        int loseCount;
        int consecutiveLosses;
        int maximumConsecutiveLosses;
        double minimumProfitLoss;

    protected:
        virtual void tick(double *dataPoint) = 0;
        double getWinRate();
        double getProfitLoss();
        void closeExpiredPositions(double price, time_t timestamp);
        void addPosition(Position *position);

    public:
        Strategy(std::string symbol);
        virtual void backtest(double *dataPoint, double investment, double profitability) = 0;
        std::string getSymbol();
        void setProfitLoss(double profitLoss);
        std::map<std::string, double> *getResults();
};

#endif
