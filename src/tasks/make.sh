#!/bin/bash

g++ -std=c++11 -Wall -pedantic -L/usr/local/lib -I/usr/local/include -lgsl -lcblas -o parsedData $1 ../studies/study.cpp ../studies/emaStudy.cpp ../studies/rsiStudy.cpp ../studies/stochasticOscillator.cpp ../studies/polynomialRegressionChannelStudy.cpp
