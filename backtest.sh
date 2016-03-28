#!/bin/bash

node --nouse-idle-notification --max-old-space-size=31000 --expose-gc `which gulp` backtest --symbol $1 --parser metatrader --data ./data/metatrader/$1.csv --optimizer Trend --investment 1000 --profitability 0.76 --database forex-backtesting
