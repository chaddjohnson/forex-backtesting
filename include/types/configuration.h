#ifndef CONFIGURATION_H
#define CONFIGURATION_H

typedef struct Configuration {
    // Basic properties
    time_t timestamp;
    int open;
    int high;
    int low;
    int close;

    // Index mappings
    int ema200;
    int ema100;
    int ema50;
    int sma13;
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
