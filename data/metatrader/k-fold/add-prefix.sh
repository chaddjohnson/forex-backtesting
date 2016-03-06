#!/bin/bash

while read lines
do
    echo "$2,$lines"
done < $1
