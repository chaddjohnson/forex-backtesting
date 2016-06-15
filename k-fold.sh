#!/bin/bash

if [ "$#" -lt 1 ]; then
  echo "One or more symbols must be provided."
  exit 1
fi

for symbol in $@; do
    # Prepare data.
    # ./bin/prepareData --symbol $symbol --parser oanda --optimizer reversals --file /home/chad/development/desktop/forex-backtesting/data/oanda/k-fold/combined/$symbol.csv

    # Perform k-fold testing for group 1.
    echo; echo "$symbol k-fold group 1 testing..."; time ./bin/optimize --symbol $symbol --type test --group 1 --optimizer reversals --investment 1000 --profitability 0.76
    echo; echo "$symbol k-fold group 1 validation..."; time ./bin/optimize --symbol $symbol --type validation --group 1 --optimizer reversals --investment 1000 --profitability 0.76

    # Perform k-fold testing for group 2.
    echo; echo "$symbol k-fold group 2 testing..."; time ./bin/optimize --symbol $symbol --type test --group 2 --optimizer reversals --investment 1000 --profitability 0.76
    echo; echo "$symbol k-fold group 2 validation..."; time ./bin/optimize --symbol $symbol --type validation --group 2 --optimizer reversals --investment 1000 --profitability 0.76

    # Perform k-fold testing for group 3.
    echo; echo "$symbol k-fold group 3 testing..."; time ./bin/optimize --symbol $symbol --type test --group 3 --optimizer reversals --investment 1000 --profitability 0.76
    echo; echo "$symbol k-fold group 3 validation..."; time ./bin/optimize --symbol $symbol --type validation --group 3 --optimizer reversals --investment 1000 --profitability 0.76

    # Perform k-fold testing for group 4.
    echo; echo "$symbol k-fold group 4 testing..."; time ./bin/optimize --symbol $symbol --type test --group 4 --optimizer reversals --investment 1000 --profitability 0.76
    echo; echo "$symbol k-fold group 4 validation..."; time ./bin/optimize --symbol $symbol --type validation --group 4 --optimizer reversals --investment 1000 --profitability 0.76

    # Perform k-fold testing for group 5.
    echo; echo "$symbol k-fold group 5 testing..."; time ./bin/optimize --symbol $symbol --type test --group 5 --optimizer reversals --investment 1000 --profitability 0.76
    echo; echo "$symbol k-fold group 5 validation..."; time ./bin/optimize --symbol $symbol --type validation --group 5 --optimizer reversals --investment 1000 --profitability 0.76

    # Perform k-fold testing for group 6.
    echo; echo "$symbol k-fold group 6 testing..."; time ./bin/optimize --symbol $symbol --type test --group 6 --optimizer reversals --investment 1000 --profitability 0.76
    echo; echo "$symbol k-fold group 6 validation..."; time ./bin/optimize --symbol $symbol --type validation --group 6 --optimizer reversals --investment 1000 --profitability 0.76

    # Perform k-fold testing for group 7.
    echo; echo "$symbol k-fold group 7 testing..."; time ./bin/optimize --symbol $symbol --type test --group 7 --optimizer reversals --investment 1000 --profitability 0.76
    echo; echo "$symbol k-fold group 7 validation..."; time ./bin/optimize --symbol $symbol --type validation --group 7 --optimizer reversals --investment 1000 --profitability 0.76

    # Perform k-fold testing for group 8.
    echo; echo "$symbol k-fold group 8 testing..."; time ./bin/optimize --symbol $symbol --type test --group 8 --optimizer reversals --investment 1000 --profitability 0.76
    echo; echo "$symbol k-fold group 8 validation..."; time ./bin/optimize --symbol $symbol --type validation --group 8 --optimizer reversals --investment 1000 --profitability 0.76

    # Perform k-fold testing for group 9.
    echo; echo "$symbol k-fold group 9 testing..."; time ./bin/optimize --symbol $symbol --type test --group 9 --optimizer reversals --investment 1000 --profitability 0.76
    echo; echo "$symbol k-fold group 9 validation..."; time ./bin/optimize --symbol $symbol --type validation --group 9 --optimizer reversals --investment 1000 --profitability 0.76

    # Perform k-fold testing for group 10.
    echo; echo "$symbol k-fold group 10 testing..."; time ./bin/optimize --symbol $symbol --type test --group 10 --optimizer reversals --investment 1000 --profitability 0.76
    echo; echo "$symbol k-fold group 10 validation..."; time ./bin/optimize --symbol $symbol --type validation --group 10 --optimizer reversals --investment 1000 --profitability 0.76

    # Perform k-fold averaging.
    # TODO
done
