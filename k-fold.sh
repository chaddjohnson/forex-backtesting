#!/bin/bash

# Group 1
echo; echo "K-Fold Group 1..."
node --nouse-idle-notification --max-old-space-size=31000 --expose-gc `which gulp` backtest --symbol $1 --parser metatrader --data ./data/metatrader/k-fold/combined/$1.csv --optimizer Reversals --investment 1000 --profitability 0.76 --database forex-backtesting

# Group 2
echo; echo "K-Fold Group 2..."
node --nouse-idle-notification --max-old-space-size=31000 --expose-gc `which gulp` forwardtest --type testing --group 2 --symbol $1 --investment 1000 --profitability 0.76 --database forex-backtesting
node --nouse-idle-notification --max-old-space-size=31000 --expose-gc `which gulp` forwardtest --type validation --group 2 --symbol $1 --investment 1000 --profitability 0.76 --database forex-backtesting

# Group 3
echo; echo "K-Fold Group 3..."
node --nouse-idle-notification --max-old-space-size=31000 --expose-gc `which gulp` forwardtest --type testing--group 3 --symbol $1 --investment 1000 --profitability 0.76 --database forex-backtesting
node --nouse-idle-notification --max-old-space-size=31000 --expose-gc `which gulp` forwardtest --type validation --group 3 --symbol $1 --investment 1000 --profitability 0.76 --database forex-backtesting

# Group 4
echo; echo "K-Fold Group 4..."
node --nouse-idle-notification --max-old-space-size=31000 --expose-gc `which gulp` forwardtest --type testing--group 4 --symbol $1 --investment 1000 --profitability 0.76 --database forex-backtesting
node --nouse-idle-notification --max-old-space-size=31000 --expose-gc `which gulp` forwardtest --type validation --group 4 --symbol $1 --investment 1000 --profitability 0.76 --database forex-backtesting

# Group 5
echo; echo "K-Fold Group 5..."
node --nouse-idle-notification --max-old-space-size=31000 --expose-gc `which gulp` forwardtest --type testing--group 5 --symbol $1 --investment 1000 --profitability 0.76 --database forex-backtesting
node --nouse-idle-notification --max-old-space-size=31000 --expose-gc `which gulp` forwardtest --type validation --group 5 --symbol $1 --investment 1000 --profitability 0.76 --database forex-backtesting

# Group 6
echo; echo "K-Fold Group 6..."
node --nouse-idle-notification --max-old-space-size=31000 --expose-gc `which gulp` forwardtest --type testing--group 6 --symbol $1 --investment 1000 --profitability 0.76 --database forex-backtesting
node --nouse-idle-notification --max-old-space-size=31000 --expose-gc `which gulp` forwardtest --type validation --group 6 --symbol $1 --investment 1000 --profitability 0.76 --database forex-backtesting

# Group 7
echo; echo "K-Fold Group 7..."
node --nouse-idle-notification --max-old-space-size=31000 --expose-gc `which gulp` forwardtest --type testing--group 7 --symbol $1 --investment 1000 --profitability 0.76 --database forex-backtesting
node --nouse-idle-notification --max-old-space-size=31000 --expose-gc `which gulp` forwardtest --type validation --group 7 --symbol $1 --investment 1000 --profitability 0.76 --database forex-backtesting

# Group 8
echo; echo "K-Fold Group 8..."
node --nouse-idle-notification --max-old-space-size=31000 --expose-gc `which gulp` forwardtest --type testing--group 8 --symbol $1 --investment 1000 --profitability 0.76 --database forex-backtesting
node --nouse-idle-notification --max-old-space-size=31000 --expose-gc `which gulp` forwardtest --type validation --group 8 --symbol $1 --investment 1000 --profitability 0.76 --database forex-backtesting

# Group 9
echo; echo "K-Fold Group 9..."
node --nouse-idle-notification --max-old-space-size=31000 --expose-gc `which gulp` forwardtest --type testing--group 9 --symbol $1 --investment 1000 --profitability 0.76 --database forex-backtesting
node --nouse-idle-notification --max-old-space-size=31000 --expose-gc `which gulp` forwardtest --type validation --group 9 --symbol $1 --investment 1000 --profitability 0.76 --database forex-backtesting

# Group 10
echo; echo "K-Fold Group 10..."
node --nouse-idle-notification --max-old-space-size=31000 --expose-gc `which gulp` forwardtest --type testing--group 10 --symbol $1 --investment 1000 --profitability 0.76 --database forex-backtesting
node --nouse-idle-notification --max-old-space-size=31000 --expose-gc `which gulp` forwardtest --type validation --group 10 --symbol $1 --investment 1000 --profitability 0.76 --database forex-backtesting
