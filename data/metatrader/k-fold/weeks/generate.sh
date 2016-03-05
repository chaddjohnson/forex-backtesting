#!/bin/bash

# Remove existing CSV files.
rm *.csv

# Generate new CSV files.
node generate

# Put CSV files in a random order.
for f in *.csv
do
    mv $f $RANDOM.csv
done

# Display the number of CSV files.
ls *.csv | wc -l

# Wait for input (or rejection via ctrl+c).
read

mkdir -p groups

# Group CSV files.
for i in {1..10}
do
    for f in `ls *.csv | head -15`
    do
        cat $f >> ./groups/$i.csv
        rm $f
    done
done

# Group remaining CSV files.
for f in *.csv
do
    cat $f >> ./groups/11.csv
    rm $f
done
