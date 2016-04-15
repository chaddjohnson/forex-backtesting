#include "studies/smaStudy.h"

std::map<std::string, double> SmaStudy::tick() {
    std::map<std::string, double> valueMap;
    std::vector<Tick> dataSegment;
    int dataSegmentLength = 0;
    double sum = 0.0;
    double sma = 0.0;

    dataSegment = getDataSegment(getInput("length"));
    dataSegmentLength = dataSegment.size();

    if (dataSegmentLength < getInput("length")) {
        return valueMap;
    }

    // Calculate the SMA.
    for (std::vector<Tick>::iterator iterator = dataSegment.begin(); iterator != dataSegment.end(); ++iterator) {
        sum += (*iterator).at("close");
    }
    sma = sum / dataSegmentLength;

    valueMap[getOutputMapping("sma")] = sma;

    return valueMap;
}
