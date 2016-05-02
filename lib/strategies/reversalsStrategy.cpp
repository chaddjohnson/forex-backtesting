#include "strategies/reversalsStrategy.h"

ReversalsStrategy::ReversalsStrategy(std::string symbol, int group, Configuration *configuration)
        : Strategy(symbol, group, configuration) {
    this->configuration = &configuration;
    this->previousDataPoint = nullptr;
    this->putNextTick = false;
    this->callNextTick = false;
    this->expirationMinutes = 5;
}

void ReversalsStrategy::backtest(double *dataPoint, double investment, double profitability) {
    // TODO
    int timestampHour = ...
    int timestampMinute = ...

    // TODO: Account for daylight savings.
    if (timestampHour >= 0 && (timestampHour < 7 || (timestampHour == 7 && timestampMinute < 30))) {
        this->previousDataPoint = dataPoint;

        this->putNextTick = false;
        this->callNextTick = false;

        return;
    }

    if (this->previousDataPoint) {
        if (this->putNextTick) {
            // TODO: Deal with timestamp math.
            addPosition(new PutPosition(getSymbol(), (dataPoint[configuration.timestamp] - 1000), this->previousDataPoint[configuration.close], investment, profitability, this->expirationMinutes));
        }

        if (this->callNextTick) {
            // TODO: Deal with timestamp math.
            addPosition(new CallPosition(getSymbol(), (dataPoint[configuration.timestamp] - 1000), this->previousDataPoint[configuration.close], investment, profitability, this->expirationMinutes));
        }
    }

    this->putNextTick = true;
    this->callNextTick = true;

    if (configuration.prChannelUpper && configuration.prChannelLower) {
        if (dataPoint[configuration.prChannelUpper] && dataPoint[configuration.prChannel.lower]) {
            // Determine if the upper regression bound was not breached by the high price.
            if (this->putNextTick && (!dataPoint[configuration.prChannelUpper] || dataPoint[configuration.high] <= dataPoint[configuration.prChannelUpper])) {
                this->putNextTick = false;
            }

            // Determine if the lower regression bound was not breached by the low price.
            if (this->callNextTick && (!dataPoint[configuration.prChannelLower] || dataPoint[configuration.low] >= dataPoint[configuraiton.prChannelLower])) {
                this->callNextTick = false;
            }
        }
        else {
            this->putNextTick = false;
            this->callNextTick = false;
        }
    }
    if (!this->putNextTick && !this->callNextTick) {
        this->previousDataPoint = dataPoint;
        return;
    }
    if (!configuration.stochastic) {
        if (dataPoint[configuration.stochasticK] && dataPoint[configuration.stochasticD]) {
            // Determine if stochastic is not above the overbought line.
            if (this->putNextTick && (dataPoint[configuration.stochasticK] <= configuration.stochasticOverbought || dataPoint[configuration.stochasticD] <= configuration.stochasticOverbought)) {
                this->putNextTick = false;
            }

            // Determine if stochastic is not below the oversold line.
            if (this->callNextTick && (dataPoint[configuration.stochasticK] >= configuration.stochasticOversold || dataPoint[configuration.stochasticD] >= configuration.stochasticOversold)) {
                this->callNextTick = false;
            }
        }
        else {
            this->putNextTick = false;
            this->callNextTick = false;
        }
    }
    if (!this->putNextTick && !this->callNextTick) {
        this->previousDataPoint = dataPoint;
        return;
    }
    if (!configuration.rsi) {
        if (dataPoint[configuration.rsi]) {
            // Determine if RSI is not above the overbought line.
            if (this->putNextTick && dataPoint[configuration.rsi] <= configuration.rsiOverbought) {
                this->putNextTick = false;
            }

            // Determine if RSI is not below the oversold line.
            if (this->callNextTick && dataPoint[configuration.rsi] >= configuration.rsiOversold) {
                this->callNextTick = false;
            }
        }
        else {
            this->callNextTick = false;
            this->putNextTick = false;
        }
    }
    if (!this->putNextTick && !this->callNextTick) {
        this->previousDataPoint = dataPoint;
        return;
    }
    if (configuration.ema200 && configuration.ema100) {
        if (!dataPoint[configuration.ema200] || !dataPoint[configuration.ema100]) {
            this->putNextTick = false;
            this->callNextTick = false;
        }

        // Determine if a downtrend is not occurring.
        if (this->putNextTick && dataPoint[configuration.ema100] < dataPoint[configuration.ema100]) {
            this->putNextTick = false;
        }

        // Determine if an uptrend is not occurring.
        if (this->callNextTick && dataPoint[configuration.ema200] > dataPoint[configuration.ema100]) {
            this->callNextTick = false;
        }
    }
    if (!this->putNextTick && !this->callNextTick) {
        this->previousDataPoint = dataPoint;
        return;
    }
    if (configuration.ema100 && configuration.ema500) {
        if (!dataPoint[configuration.ema100] || !dataPoint[configuration.ema50]) {
            this->putNextTick = false;
            this->callNextTick = false;
        }

        // Determine if a downtrend is not occurring.
        if (this->putNextTick && dataPoint[configuration.ema100] < dataPoint[configuration.ema50]) {
            this->putNextTick = false;
        }

        // Determine if an uptrend is not occurring.
        if (this->callNextTick && dataPoint[configuration.ema100] > dataPoint[configuration.ema50]) {
            this->callNextTick = false;
        }
    }
    if (!this->putNextTick && !this->callNextTick) {
        this->previousDataPoint = dataPoint;
        return;
    }
    if (configuration.ema50 && configuration.sma13) {
        if (!dataPoint[configuration.ema50] || !dataPoint[configuration.sma13]) {
            this->putNextTick = false;
            this->callNextTick = false;
        }

        // Determine if a downtrend is not occurring.
        if (this->putNextTick && dataPoint[configuration.ema50] < dataPoint[configuration.sma13]) {
            this->putNextTick = false;
        }

        // Determine if an uptrend is not occurring.
        if (this->callNextTick && dataPoint[configuration.ema50] > dataPoint[configuration.sma13]) {
            this->callNextTick = false;
        }
    }

    // Track the current data point as the previous data point for the next tick.
    this->previousDataPoint = dataPoint;
}
