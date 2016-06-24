#include "strategies/reversalsOptimizationStrategy.cuh"

__device__ __host__ ReversalsOptimizationStrategy::ReversalsOptimizationStrategy(const char *symbol, Configuration configuration)
        : OptimizationStrategy(symbol, configuration) {
    this->configuration = configuration;
    this->putNextTick = false;
    this->callNextTick = false;
    this->expirationMinutes = 5;
}

__device__ __host__ void ReversalsOptimizationStrategy::backtest(double *dataPoint, double investment, double profitability) {
    // Tick the strategy.
    this->tick(getPreviousClose(), (int)dataPoint[configuration.timestamp]);

    // Do not create trades between 4pm - 11:30pm Central, as the payout is lower during these times.
    if (dataPoint[configuration.timestampHour] >= 16 && (dataPoint[configuration.timestampHour] < 23 || (dataPoint[configuration.timestampHour] == 23 && dataPoint[configuration.timestampMinute] < 30))) {
        setPreviousClose(dataPoint[configuration.close]);

        putNextTick = false;
        callNextTick = false;

        return;
    }

    if (getPreviousClose()) {
        if (putNextTick) {
            addPutPosition(getSymbol(), ((int)dataPoint[configuration.timestamp]), getPreviousClose(), investment, profitability, expirationMinutes);
        }
        if (callNextTick) {
            addCallPosition(getSymbol(), ((int)dataPoint[configuration.timestamp]), getPreviousClose(), investment, profitability, expirationMinutes);
        }
    }

    putNextTick = true;
    callNextTick = true;

    if (configuration.prChannelUpper && configuration.prChannelLower) {
        if (dataPoint[configuration.prChannelUpper] && dataPoint[configuration.prChannelLower]) {
            // Determine if the upper regression bound was not breached by the high price.
            if (putNextTick && (!dataPoint[configuration.prChannelUpper] || dataPoint[configuration.high] <= dataPoint[configuration.prChannelUpper])) {
                putNextTick = false;
            }

            // Determine if the lower regression bound was not breached by the low price.
            if (callNextTick && (!dataPoint[configuration.prChannelLower] || dataPoint[configuration.low] >= dataPoint[configuration.prChannelLower])) {
                callNextTick = false;
            }
        }
        else {
            putNextTick = false;
            callNextTick = false;
        }
    }
    if (!putNextTick && !callNextTick) {
        setPreviousClose(dataPoint[configuration.close]);
        return;
    }
    if (configuration.rsi) {
        if (dataPoint[configuration.rsi]) {
            // Determine if RSI is not above the overbought line.
            if (putNextTick && dataPoint[configuration.rsi] <= configuration.rsiOverbought) {
                putNextTick = false;
            }

            // Determine if RSI is not below the oversold line.
            if (callNextTick && dataPoint[configuration.rsi] >= configuration.rsiOversold) {
                callNextTick = false;
            }
        }
        else {
            callNextTick = false;
            putNextTick = false;
        }
    }
    if (!putNextTick && !callNextTick) {
        setPreviousClose(dataPoint[configuration.close]);
        return;
    }
    if (configuration.stochasticK && configuration.stochasticD) {
        if (dataPoint[configuration.stochasticK] && dataPoint[configuration.stochasticD]) {
            // Determine if stochastic is not above the overbought line.
            if (putNextTick && (dataPoint[configuration.stochasticK] <= configuration.stochasticOverbought || dataPoint[configuration.stochasticD] <= configuration.stochasticOverbought)) {
                putNextTick = false;
            }

            // Determine if stochastic is not below the oversold line.
            if (callNextTick && (dataPoint[configuration.stochasticK] >= configuration.stochasticOversold || dataPoint[configuration.stochasticD] >= configuration.stochasticOversold)) {
                callNextTick = false;
            }
        }
        else {
            putNextTick = false;
            callNextTick = false;
        }
    }
    if (!putNextTick && !callNextTick) {
        setPreviousClose(dataPoint[configuration.close]);
        return;
    }
    if (configuration.ema50 && configuration.sma13) {
        if (!dataPoint[configuration.ema50] || !dataPoint[configuration.sma13]) {
            putNextTick = false;
            callNextTick = false;
        }

        // Determine if a downtrend is not occurring.
        if (putNextTick && dataPoint[configuration.ema50] < dataPoint[configuration.sma13]) {
            putNextTick = false;
        }

        // Determine if an uptrend is not occurring.
        if (callNextTick && dataPoint[configuration.ema50] > dataPoint[configuration.sma13]) {
            callNextTick = false;
        }
    }
    if (!putNextTick && !callNextTick) {
        setPreviousClose(dataPoint[configuration.close]);
        return;
    }
    if (configuration.ema100 && configuration.ema50) {
        if (!dataPoint[configuration.ema100] || !dataPoint[configuration.ema50]) {
            putNextTick = false;
            callNextTick = false;
        }

        // Determine if a downtrend is not occurring.
        if (putNextTick && dataPoint[configuration.ema100] < dataPoint[configuration.ema50]) {
            putNextTick = false;
        }

        // Determine if an uptrend is not occurring.
        if (callNextTick && dataPoint[configuration.ema100] > dataPoint[configuration.ema50]) {
            callNextTick = false;
        }
    }
    if (!putNextTick && !callNextTick) {
        setPreviousClose(dataPoint[configuration.close]);
        return;
    }

    if (configuration.ema200 && configuration.ema100) {
        if (!dataPoint[configuration.ema200] || !dataPoint[configuration.ema100]) {
            putNextTick = false;
            callNextTick = false;
        }

        // Determine if a downtrend is not occurring.
        if (putNextTick && dataPoint[configuration.ema200] < dataPoint[configuration.ema100]) {
            putNextTick = false;
        }

        // Determine if an uptrend is not occurring.
        if (callNextTick && dataPoint[configuration.ema200] > dataPoint[configuration.ema100]) {
            callNextTick = false;
        }
    }
    if (!putNextTick && !callNextTick) {
        setPreviousClose(dataPoint[configuration.close]);
        return;
    }
    if (configuration.ema250 && configuration.ema200) {
        if (!dataPoint[configuration.ema250] || !dataPoint[configuration.ema200]) {
            putNextTick = false;
            callNextTick = false;
        }

        // Determine if a downtrend is not occurring.
        if (putNextTick && dataPoint[configuration.ema250] < dataPoint[configuration.ema200]) {
            putNextTick = false;
        }

        // Determine if an uptrend is not occurring.
        if (callNextTick && dataPoint[configuration.ema250] > dataPoint[configuration.ema200]) {
            callNextTick = false;
        }
    }
    if (!putNextTick && !callNextTick) {
        setPreviousClose(dataPoint[configuration.close]);
        return;
    }
    if (configuration.ema300 && configuration.ema250) {
        if (!dataPoint[configuration.ema300] || !dataPoint[configuration.ema250]) {
            putNextTick = false;
            callNextTick = false;
        }

        // Determine if a downtrend is not occurring.
        if (putNextTick && dataPoint[configuration.ema300] < dataPoint[configuration.ema250]) {
            putNextTick = false;
        }

        // Determine if an uptrend is not occurring.
        if (callNextTick && dataPoint[configuration.ema300] > dataPoint[configuration.ema250]) {
            callNextTick = false;
        }
    }
    if (!putNextTick && !callNextTick) {
        setPreviousClose(dataPoint[configuration.close]);
        return;
    }
    if (configuration.ema350 && configuration.ema300) {
        if (!dataPoint[configuration.ema350] || !dataPoint[configuration.ema300]) {
            putNextTick = false;
            callNextTick = false;
        }

        // Determine if a downtrend is not occurring.
        if (putNextTick && dataPoint[configuration.ema350] < dataPoint[configuration.ema300]) {
            putNextTick = false;
        }

        // Determine if an uptrend is not occurring.
        if (callNextTick && dataPoint[configuration.ema350] > dataPoint[configuration.ema300]) {
            callNextTick = false;
        }
    }
    if (!putNextTick && !callNextTick) {
        setPreviousClose(dataPoint[configuration.close]);
        return;
    }
    if (configuration.ema400 && configuration.ema350) {
        if (!dataPoint[configuration.ema400] || !dataPoint[configuration.ema350]) {
            putNextTick = false;
            callNextTick = false;
        }

        // Determine if a downtrend is not occurring.
        if (putNextTick && dataPoint[configuration.ema400] < dataPoint[configuration.ema350]) {
            putNextTick = false;
        }

        // Determine if an uptrend is not occurring.
        if (callNextTick && dataPoint[configuration.ema400] > dataPoint[configuration.ema350]) {
            callNextTick = false;
        }
    }
    if (!putNextTick && !callNextTick) {
        setPreviousClose(dataPoint[configuration.close]);
        return;
    }
    if (configuration.ema450 && configuration.ema400) {
        if (!dataPoint[configuration.ema450] || !dataPoint[configuration.ema400]) {
            putNextTick = false;
            callNextTick = false;
        }

        // Determine if a downtrend is not occurring.
        if (putNextTick && dataPoint[configuration.ema450] < dataPoint[configuration.ema400]) {
            putNextTick = false;
        }

        // Determine if an uptrend is not occurring.
        if (callNextTick && dataPoint[configuration.ema450] > dataPoint[configuration.ema400]) {
            callNextTick = false;
        }
    }
    if (!putNextTick && !callNextTick) {
        setPreviousClose(dataPoint[configuration.close]);
        return;
    }
    if (configuration.ema500 && configuration.ema450) {
        if (!dataPoint[configuration.ema500] || !dataPoint[configuration.ema450]) {
            putNextTick = false;
            callNextTick = false;
        }

        // Determine if a downtrend is not occurring.
        if (putNextTick && dataPoint[configuration.ema500] < dataPoint[configuration.ema450]) {
            putNextTick = false;
        }

        // Determine if an uptrend is not occurring.
        if (callNextTick && dataPoint[configuration.ema500] > dataPoint[configuration.ema450]) {
            callNextTick = false;
        }
    }

    setPreviousClose(dataPoint[configuration.close]);
}
