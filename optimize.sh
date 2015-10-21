#!/bin/bash

node --nouse-idle-notification --max-old-space-size=31000 --expose-gc `which gulp` optimize --symbol $1 --parser ctoption --data ./data/ctoption/temp.csv --optimizer Reversals --investment 1000 --profitability 0.76 --database forex-backtesting
