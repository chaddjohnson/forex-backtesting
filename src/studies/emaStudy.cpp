#include "emaStudy.h"

EmaStudy::EmaStudy(std::map<std::string, double> &inputs, std::map<std::string, std::string> &outputMap)
            : Study(inputs, outputMap) {
    previousEma = 0.0;
}

std::map<std::string, double> EmaStudy::tick() {
    std::map<std::string, double> valueMap;
    Tick lastDataPoint = getLast();
    double K = 0.0;
    double ema = 0.0;

    if (previousEma == 0.0) {
        // Use the last data item as the first previous EMA value.
        previousEma = lastTick.close;
    }

    // Calculate the EMA.
    K = 2 / (1 + getInput('length'));
    ema = (lastTick.close * K) + (previousEma * (1 - K));

    // Set the new EMA just calculated as the previous EMA.
    previousEma = ema;

    valueMap[getOutputMapping("ema")] = ema;

    return valueMap;
}
