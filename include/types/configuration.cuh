#ifndef CONFIGURATION_H
#define CONFIGURATION_H

#include "types/real.cuh"

typedef struct Configuration {
    // Index mappings
    int timestamp;
    int timestampHour;
    int timestampMinute;
    int open;
    int high;
    int low;
    int close;
    int sma13;
    int ema50;
    int ema100;
    int ema200;
    int ema250;
    int ema300;
    int ema350;
    int ema400;
    int ema450;
    int ema500;
    int rsi;
    int stochasticD;
    int stochasticK;
    int prChannelUpper;
    int prChannelLower;

    // Values
    Real rsiOverbought;
    Real rsiOversold;
    Real stochasticOverbought;
    Real stochasticOversold;
} Configuration;

#endif
