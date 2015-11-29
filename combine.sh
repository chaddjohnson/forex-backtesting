#!/bin/bash

node --nouse-idle-notification --max-old-space-size=31000 --expose-gc `which gulp` combine --symbol $1 --strategy Reversals --investment 1000 --profitability 0.76 --database forex-backtesting-stochastic
