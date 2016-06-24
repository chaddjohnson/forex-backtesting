#ifndef CONFIGURATION_H
#define CONFIGURATION_H

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
    double rsiOverbought;
    double rsiOversold;
    double stochasticOverbought;
    double stochasticOversold;

    __device__ __host__ Configuration() {
        timestamp = 0;
        timestampHour = 0;
        timestampMinute = 0;
        open = 0;
        high = 0;
        low = 0;
        close = 0;
        sma13 = 0;
        ema50 = 0;
        ema100 = 0;
        ema200 = 0;
        ema250 = 0;
        ema300 = 0;
        ema350 = 0;
        ema400 = 0;
        ema450 = 0;
        ema500 = 0;
        rsi = 0;
        stochasticD = 0;
        stochasticK = 0;
        prChannelUpper = 0;
        prChannelLower = 0;
        rsiOverbought = 0.0;
        rsiOversold = 0.0;
        stochasticOverbought = 0.0;
        stochasticOversold = 0.0;
    }
} Configuration;

#endif
