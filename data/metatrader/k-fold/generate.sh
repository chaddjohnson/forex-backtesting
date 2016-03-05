#!/bin/bash

# Remove existing CSV files.
rm -rf *.csv ./part*/**/$1.csv ./final/$1.csv

# Generate new CSV files.
node generate $1

# Put CSV files in a random order.
for f in *.csv
do
    mv $f $RANDOM.csv
done

# Display the number of CSV files.
ls *.csv | wc -l

# Wait for input (or rejection via ctrl+c).
read

# Group CSV files for final validation data.
for f in `ls *.csv | sort | tail -n 14`
do
    cat $f >> ./final/$1.csv
    rm $f
done

# Group 1 testing data.
for f in `ls *.csv | sort | tail -n 135`
do
    cat $f >> ./part1/testing/$1.csv
done

# Group 1 validation data.
for f in `ls *.csv | sort | head -n 15`
do
    cat $f >> ./part1/validation/$1.csv
done

# Remaining groups.
for i in {2..10}
do
    # Testing data.
    for f in `{ ls *.csv | sort | head -n $(((i-1)*15)); ls *.csv | sort | tail -n $((150-(i*15))); }`
    do
        cat $f >> ./part${i}/testing/$1.csv
    done

    # Validation data.
    for f in `{ ls *.csv | sort | sed -n $((((i-1)*15)+1)),$((i*15))p; }`
    do
        cat $f >> ./part${i}/validation/$1.csv
    done
done

rm *.csv
