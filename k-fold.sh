#!/bin/bash

# Round 1
node --nouse-idle-notification --max-old-space-size=31000 --expose-gc `which gulp` backtest --symbol $1 --parser metatrader --data ./data/metatrader/k-fold/part1/testing/$1.csv --optimizer Reversals --investment 1000 --profitability 0.76 --database forex-backtesting
node --nouse-idle-notification --max-old-space-size=31000 --expose-gc `which gulp` forwardtest --type validation --round 1 --symbol $1 --parser metatrader --data ./data/metatrader/k-fold/part1/validation/$1.csv --investment 1000 --profitability 0.76 --database forex-backtesting

# Round 2
node --nouse-idle-notification --max-old-space-size=31000 --expose-gc `which gulp` forwardtest --type forwardtest --round 2 --symbol $1 --parser metatrader --data ./data/metatrader/k-fold/part2/testing/$1.csv --investment 1000 --profitability 0.76 --database forex-backtesting
node --nouse-idle-notification --max-old-space-size=31000 --expose-gc `which gulp` forwardtest --type validation --round 2 --symbol $1 --parser metatrader --data ./data/metatrader/k-fold/part2/validation/$1.csv --investment 1000 --profitability 0.76 --database forex-backtesting

# Round 3
node --nouse-idle-notification --max-old-space-size=31000 --expose-gc `which gulp` forwardtest --type forwardtest--round 3 --symbol $1 --parser metatrader --data ./data/metatrader/k-fold/part3/testing/$1.csv --investment 1000 --profitability 0.76 --database forex-backtesting
node --nouse-idle-notification --max-old-space-size=31000 --expose-gc `which gulp` forwardtest --type validation --round 3 --symbol $1 --parser metatrader --data ./data/metatrader/k-fold/part3/validation/$1.csv --investment 1000 --profitability 0.76 --database forex-backtesting

# Round 4
node --nouse-idle-notification --max-old-space-size=31000 --expose-gc `which gulp` forwardtest --type forwardtest--round 4 --symbol $1 --parser metatrader --data ./data/metatrader/k-fold/part4/testing/$1.csv --investment 1000 --profitability 0.76 --database forex-backtesting
node --nouse-idle-notification --max-old-space-size=31000 --expose-gc `which gulp` forwardtest --type validation --round 4 --symbol $1 --parser metatrader --data ./data/metatrader/k-fold/part4/validation/$1.csv --investment 1000 --profitability 0.76 --database forex-backtesting

# Round 5
node --nouse-idle-notification --max-old-space-size=31000 --expose-gc `which gulp` forwardtest --type forwardtest--round 5 --symbol $1 --parser metatrader --data ./data/metatrader/k-fold/part5/testing/$1.csv --investment 1000 --profitability 0.76 --database forex-backtesting
node --nouse-idle-notification --max-old-space-size=31000 --expose-gc `which gulp` forwardtest --type validation --round 5 --symbol $1 --parser metatrader --data ./data/metatrader/k-fold/part5/validation/$1.csv --investment 1000 --profitability 0.76 --database forex-backtesting

# Round 6
node --nouse-idle-notification --max-old-space-size=31000 --expose-gc `which gulp` forwardtest --type forwardtest--round 6 --symbol $1 --parser metatrader --data ./data/metatrader/k-fold/part6/testing/$1.csv --investment 1000 --profitability 0.76 --database forex-backtesting
node --nouse-idle-notification --max-old-space-size=31000 --expose-gc `which gulp` forwardtest --type validation --round 6 --symbol $1 --parser metatrader --data ./data/metatrader/k-fold/part6/validation/$1.csv --investment 1000 --profitability 0.76 --database forex-backtesting

# Round 7
node --nouse-idle-notification --max-old-space-size=31000 --expose-gc `which gulp` forwardtest --type forwardtest--round 7 --symbol $1 --parser metatrader --data ./data/metatrader/k-fold/part7/testing/$1.csv --investment 1000 --profitability 0.76 --database forex-backtesting
node --nouse-idle-notification --max-old-space-size=31000 --expose-gc `which gulp` forwardtest --type validation --round 7 --symbol $1 --parser metatrader --data ./data/metatrader/k-fold/part7/validation/$1.csv --investment 1000 --profitability 0.76 --database forex-backtesting

# Round 8
node --nouse-idle-notification --max-old-space-size=31000 --expose-gc `which gulp` forwardtest --type forwardtest--round 8 --symbol $1 --parser metatrader --data ./data/metatrader/k-fold/part8/testing/$1.csv --investment 1000 --profitability 0.76 --database forex-backtesting
node --nouse-idle-notification --max-old-space-size=31000 --expose-gc `which gulp` forwardtest --type validation --round 8 --symbol $1 --parser metatrader --data ./data/metatrader/k-fold/part8/validation/$1.csv --investment 1000 --profitability 0.76 --database forex-backtesting

# Round 9
node --nouse-idle-notification --max-old-space-size=31000 --expose-gc `which gulp` forwardtest --type forwardtest--round 9 --symbol $1 --parser metatrader --data ./data/metatrader/k-fold/part9/testing/$1.csv --investment 1000 --profitability 0.76 --database forex-backtesting
node --nouse-idle-notification --max-old-space-size=31000 --expose-gc `which gulp` forwardtest --type validation --round 9 --symbol $1 --parser metatrader --data ./data/metatrader/k-fold/part9/validation/$1.csv --investment 1000 --profitability 0.76 --database forex-backtesting

# Round 10
node --nouse-idle-notification --max-old-space-size=31000 --expose-gc `which gulp` forwardtest --type forwardtest--round 10 --symbol $1 --parser metatrader --data ./data/metatrader/k-fold/part10/testing/$1.csv --investment 1000 --profitability 0.76 --database forex-backtesting
node --nouse-idle-notification --max-old-space-size=31000 --expose-gc `which gulp` forwardtest --type validation --round 10 --symbol $1 --parser metatrader --data ./data/metatrader/k-fold/part10/validation/$1.csv --investment 1000 --profitability 0.76 --database forex-backtesting
