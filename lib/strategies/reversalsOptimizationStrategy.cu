#include "strategies/reversalsOptimizationStrategy.cuh"

ReversalsOptimizationStrategy::ReversalsOptimizationStrategy(const char *symbol, BasicDataIndexMap dataIndexMap, int group, Configuration *configuration)
        : OptimizationStrategy(symbol, dataIndexMap, group, configuration) {
    this->configuration = configuration;
    this->previousDataPoint = nullptr;
    this->putNextTick = false;
    this->callNextTick = false;
    this->expirationMinutes = 5;
}

ReversalsOptimizationStrategy::~ReversalsOptimizationStrategy() {
    delete configuration;
    delete previousDataPoint;
}

void ReversalsOptimizationStrategy::backtest(double *dataPoint, double investment, double profitability) {
    time_t utcTime = dataPoint[configuration->timestamp];
    struct tm *localTime = localtime(&utcTime);
    int timestampHour = localTime->tm_hour;
    int timestampMinute = localTime->tm_min;

    // // Tick the strategy.
    this->tick(dataPoint);

    // // Do not create trades between 4pm - 11:30pm Central, as the payout is lower during these times.
    if (timestampHour >= 16 && (timestampHour < 23 || (timestampHour == 23 && timestampMinute < 30))) {
        previousDataPoint = dataPoint;

        putNextTick = false;
        callNextTick = false;

        return;
    }

    if (previousDataPoint) {
        if (putNextTick) {
            addPosition(new PutPosition(getSymbol(), (dataPoint[configuration->timestamp] - 1), previousDataPoint[configuration->close], investment, profitability, expirationMinutes));
        }
        if (callNextTick) {
            addPosition(new CallPosition(getSymbol(), (dataPoint[configuration->timestamp] - 1), previousDataPoint[configuration->close], investment, profitability, expirationMinutes));
        }
    }

    putNextTick = true;
    callNextTick = true;

    if (configuration->prChannelUpper && configuration->prChannelLower) {
        if (dataPoint[configuration->prChannelUpper] && dataPoint[configuration->prChannelLower]) {
            // Determine if the upper regression bound was not breached by the high price.
            if (putNextTick && (!dataPoint[configuration->prChannelUpper] || dataPoint[configuration->high] <= dataPoint[configuration->prChannelUpper])) {
                putNextTick = false;
            }

            // Determine if the lower regression bound was not breached by the low price.
            if (callNextTick && (!dataPoint[configuration->prChannelLower] || dataPoint[configuration->low] >= dataPoint[configuration->prChannelLower])) {
                callNextTick = false;
            }
        }
        else {
            putNextTick = false;
            callNextTick = false;
        }
    }
    if (!putNextTick && !callNextTick) {
        previousDataPoint = dataPoint;
        return;
    }
    if (!configuration->stochasticK && configuration->stochasticD) {
        if (dataPoint[configuration->stochasticK] && dataPoint[configuration->stochasticD]) {
            // Determine if stochastic is not above the overbought line.
            if (putNextTick && (dataPoint[configuration->stochasticK] <= configuration->stochasticOverbought || dataPoint[configuration->stochasticD] <= configuration->stochasticOverbought)) {
                putNextTick = false;
            }

            // Determine if stochastic is not below the oversold line.
            if (callNextTick && (dataPoint[configuration->stochasticK] >= configuration->stochasticOversold || dataPoint[configuration->stochasticD] >= configuration->stochasticOversold)) {
                callNextTick = false;
            }
        }
        else {
            putNextTick = false;
            callNextTick = false;
        }
    }
    if (!putNextTick && !callNextTick) {
        previousDataPoint = dataPoint;
        return;
    }
    if (!configuration->rsi) {
        if (dataPoint[configuration->rsi]) {
            // Determine if RSI is not above the overbought line.
            if (putNextTick && dataPoint[configuration->rsi] <= configuration->rsiOverbought) {
                putNextTick = false;
            }

            // Determine if RSI is not below the oversold line.
            if (callNextTick && dataPoint[configuration->rsi] >= configuration->rsiOversold) {
                callNextTick = false;
            }
        }
        else {
            callNextTick = false;
            putNextTick = false;
        }
    }
    if (!putNextTick && !callNextTick) {
        previousDataPoint = dataPoint;
        return;
    }
    if (configuration->ema500 && configuration->ema450) {
        if (!dataPoint[configuration->ema500] || !dataPoint[configuration->ema450]) {
            putNextTick = false;
            callNextTick = false;
        }

        // Determine if a downtrend is not occurring.
        if (putNextTick && dataPoint[configuration->ema500] < dataPoint[configuration->ema450]) {
            putNextTick = false;
        }

        // Determine if an uptrend is not occurring.
        if (callNextTick && dataPoint[configuration->ema500] > dataPoint[configuration->ema450]) {
            callNextTick = false;
        }
    }
    if (configuration->ema450 && configuration->ema400) {
        if (!dataPoint[configuration->ema450] || !dataPoint[configuration->ema400]) {
            putNextTick = false;
            callNextTick = false;
        }

        // Determine if a downtrend is not occurring.
        if (putNextTick && dataPoint[configuration->ema450] < dataPoint[configuration->ema400]) {
            putNextTick = false;
        }

        // Determine if an uptrend is not occurring.
        if (callNextTick && dataPoint[configuration->ema450] > dataPoint[configuration->ema400]) {
            callNextTick = false;
        }
    }
    if (configuration->ema400 && configuration->ema350) {
        if (!dataPoint[configuration->ema400] || !dataPoint[configuration->ema350]) {
            putNextTick = false;
            callNextTick = false;
        }

        // Determine if a downtrend is not occurring.
        if (putNextTick && dataPoint[configuration->ema400] < dataPoint[configuration->ema350]) {
            putNextTick = false;
        }

        // Determine if an uptrend is not occurring.
        if (callNextTick && dataPoint[configuration->ema400] > dataPoint[configuration->ema350]) {
            callNextTick = false;
        }
    }
    if (configuration->ema350 && configuration->ema300) {
        if (!dataPoint[configuration->ema350] || !dataPoint[configuration->ema300]) {
            putNextTick = false;
            callNextTick = false;
        }

        // Determine if a downtrend is not occurring.
        if (putNextTick && dataPoint[configuration->ema350] < dataPoint[configuration->ema300]) {
            putNextTick = false;
        }

        // Determine if an uptrend is not occurring.
        if (callNextTick && dataPoint[configuration->ema350] > dataPoint[configuration->ema300]) {
            callNextTick = false;
        }
    }
    if (configuration->ema300 && configuration->ema250) {
        if (!dataPoint[configuration->ema300] || !dataPoint[configuration->ema250]) {
            putNextTick = false;
            callNextTick = false;
        }

        // Determine if a downtrend is not occurring.
        if (putNextTick && dataPoint[configuration->ema300] < dataPoint[configuration->ema250]) {
            putNextTick = false;
        }

        // Determine if an uptrend is not occurring.
        if (callNextTick && dataPoint[configuration->ema300] > dataPoint[configuration->ema250]) {
            callNextTick = false;
        }
    }
    if (configuration->ema250 && configuration->ema200) {
        if (!dataPoint[configuration->ema250] || !dataPoint[configuration->ema200]) {
            putNextTick = false;
            callNextTick = false;
        }

        // Determine if a downtrend is not occurring.
        if (putNextTick && dataPoint[configuration->ema250] < dataPoint[configuration->ema200]) {
            putNextTick = false;
        }

        // Determine if an uptrend is not occurring.
        if (callNextTick && dataPoint[configuration->ema250] > dataPoint[configuration->ema200]) {
            callNextTick = false;
        }
    }
    if (configuration->ema200 && configuration->ema100) {
        if (!dataPoint[configuration->ema200] || !dataPoint[configuration->ema100]) {
            putNextTick = false;
            callNextTick = false;
        }

        // Determine if a downtrend is not occurring.
        if (putNextTick && dataPoint[configuration->ema200] < dataPoint[configuration->ema100]) {
            putNextTick = false;
        }

        // Determine if an uptrend is not occurring.
        if (callNextTick && dataPoint[configuration->ema200] > dataPoint[configuration->ema100]) {
            callNextTick = false;
        }
    }
    if (!putNextTick && !callNextTick) {
        previousDataPoint = dataPoint;
        return;
    }
    if (configuration->ema100 && configuration->ema50) {
        if (!dataPoint[configuration->ema100] || !dataPoint[configuration->ema50]) {
            putNextTick = false;
            callNextTick = false;
        }

        // Determine if a downtrend is not occurring.
        if (putNextTick && dataPoint[configuration->ema100] < dataPoint[configuration->ema50]) {
            putNextTick = false;
        }

        // Determine if an uptrend is not occurring.
        if (callNextTick && dataPoint[configuration->ema100] > dataPoint[configuration->ema50]) {
            callNextTick = false;
        }
    }
    if (!putNextTick && !callNextTick) {
        previousDataPoint = dataPoint;
        return;
    }
    if (configuration->ema50 && configuration->sma13) {
        if (!dataPoint[configuration->ema50] || !dataPoint[configuration->sma13]) {
            putNextTick = false;
            callNextTick = false;
        }

        // Determine if a downtrend is not occurring.
        if (putNextTick && dataPoint[configuration->ema50] < dataPoint[configuration->sma13]) {
            putNextTick = false;
        }

        // Determine if an uptrend is not occurring.
        if (callNextTick && dataPoint[configuration->ema50] > dataPoint[configuration->sma13]) {
            callNextTick = false;
        }
    }

    // Track the current data point as the previous data point for the next tick.
    previousDataPoint = dataPoint;
}
