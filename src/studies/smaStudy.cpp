#include "smaStudy.h"

std::map<std::string, double> SmaStudy::tick() {
    std::map<std::string, double> valueMap;
    std::vector<Tick> dataSegment;
    int dataSegmentCount = 0;
    double sum = 0.0;
    double sma = 0.0;

    dataSegment = getDataSegment(getInput("length"));
    dataSegmentCount = dataSegment.size();

    if (dataSegmentCount < getInput("length")) {
        return valueMap;
    }

    // Calculate the SMA.
    for (std::vector<Tick>::iterator iterator = dataSegment.begin(); iterator != dataSegment.end(); ++iterator) {
        sum += iterator->close;
    }
    sma = sum / dataSegmentCount;

    valueMap[getOutputMapping("sma")] = sma;

    return valueMap;
}
