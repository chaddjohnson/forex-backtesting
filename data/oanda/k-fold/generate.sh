#!/bin/bash

# Remove existing CSV files.
rm -rf ./*.csv ./part*/**/$1.csv ./final/$1.csv ./combined/$1.csv

# Generate new CSV files.
node generate $1

# Put CSV files in a random order.
for f in ./*.csv
do
    mv $f $RANDOM$RANDOM.csv
done

mkdir -p final
mkdir -p combined

# Group CSV files for final validation data.
for f in `ls ./*.csv | sort | tail -n 59`
do
    cat $f >> ./final/$1.csv
    rm $f
done

# Remaining groups.
for i in {1..10}
do
    testing_groups=''
    validation_groups=''

    # Groups before validation (exclusion) group.
    if [ $i -ne 1 ]; then
        for j in $(seq 1 $((i-1)))
        do
            testing_groups="$testing_groups;$j"
        done
    fi

    # Groups after validation (exclusion) group except the last one.
    if [ $i -ne 10 ]; then
        for j in $(seq $((i+1)) 10)
        do
            testing_groups="$testing_groups;$j"
        done
    fi

    validation_groups="$validation_groups;$i"

    # Trim leading comma from strings.
    testing_groups=`echo $testing_groups | sed 's/^;//'`
    validation_groups=`echo $validation_groups | sed 's/^;//'`

    # Testing data.
    for f in `ls ./*.csv | awk -v start=$((((i-1)*57)+1)) -v end=$((i*57)) 'NR >= start && NR <= end'`
    do
        cat $f >> ./combined/$1-temp.csv

        # Add a newline.
        echo >> ./combined/$1-temp.csv
    done

    # Add testing and validation group lists to each line.
    ./add-prefix.sh ./combined/$1-temp.csv "$testing_groups,$validation_groups" >> ./combined/$1.csv

    rm ./combined/$1-temp.csv
done

rm ./*.csv
