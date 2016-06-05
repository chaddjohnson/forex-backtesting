#!/bin/bash

# Prepare data.
./bin/prepareData --symbol AUDJPY --parser oanda --optimizer reversals --file /home/chad/development/desktop/forex-backtesting/data/oanda/k-fold/combined/AUDJPY.csv

# Perform k-fold testing.
time ./bin/optimize --symbol AUDJPY --type testing --group 1 --optimizer reversals --investment 1000 --profitability 0.76
time ./bin/optimize --symbol AUDJPY --type testing --group 2 --optimizer reversals --investment 1000 --profitability 0.76
time ./bin/optimize --symbol AUDJPY --type testing --group 3 --optimizer reversals --investment 1000 --profitability 0.76
time ./bin/optimize --symbol AUDJPY --type testing --group 4 --optimizer reversals --investment 1000 --profitability 0.76
time ./bin/optimize --symbol AUDJPY --type testing --group 5 --optimizer reversals --investment 1000 --profitability 0.76
time ./bin/optimize --symbol AUDJPY --type testing --group 6 --optimizer reversals --investment 1000 --profitability 0.76
time ./bin/optimize --symbol AUDJPY --type testing --group 7 --optimizer reversals --investment 1000 --profitability 0.76
time ./bin/optimize --symbol AUDJPY --type testing --group 8 --optimizer reversals --investment 1000 --profitability 0.76
time ./bin/optimize --symbol AUDJPY --type testing --group 9 --optimizer reversals --investment 1000 --profitability 0.76
time ./bin/optimize --symbol AUDJPY --type testing --group 10 --optimizer reversals --investment 1000 --profitability 0.76

# Perform k-fold validation.
time ./bin/optimize --symbol AUDJPY --type validation --group 1 --optimizer reversals --investment 1000 --profitability 0.76
time ./bin/optimize --symbol AUDJPY --type validation --group 2 --optimizer reversals --investment 1000 --profitability 0.76
time ./bin/optimize --symbol AUDJPY --type validation --group 3 --optimizer reversals --investment 1000 --profitability 0.76
time ./bin/optimize --symbol AUDJPY --type validation --group 4 --optimizer reversals --investment 1000 --profitability 0.76
time ./bin/optimize --symbol AUDJPY --type validation --group 5 --optimizer reversals --investment 1000 --profitability 0.76
time ./bin/optimize --symbol AUDJPY --type validation --group 6 --optimizer reversals --investment 1000 --profitability 0.76
time ./bin/optimize --symbol AUDJPY --type validation --group 7 --optimizer reversals --investment 1000 --profitability 0.76
time ./bin/optimize --symbol AUDJPY --type validation --group 8 --optimizer reversals --investment 1000 --profitability 0.76
time ./bin/optimize --symbol AUDJPY --type validation --group 9 --optimizer reversals --investment 1000 --profitability 0.76
time ./bin/optimize --symbol AUDJPY --type validation --group 10 --optimizer reversals --investment 1000 --profitability 0.76

# Perform k-fold averaging.
# ...
