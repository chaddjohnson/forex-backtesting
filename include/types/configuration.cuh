#ifndef CONFIGURATION_H
#define CONFIGURATION_H

typedef struct Configuration {
    // Index mappings
    int timestamp;
    int open;
    int high;
    int low;
    int close;
    int sma13;
    int ema50;
    int ema100;
    int ema200;
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
} Configuration;

#endif
