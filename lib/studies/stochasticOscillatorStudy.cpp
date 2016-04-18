#include "studies/stochasticOscillatorStudy.h"

std::map<std::string, double> StochasticOscillatorStudy::tick() {
    std::map<std::string, double> valueMap;
    std::vector<Tick> dataSegment;
    int dataSegmentLength = 0;
    std::vector<Tick> averageLengthDataSegment;
    Tick lastTick = getLastTick();
    double low = 0.0;
    double high = 0.0;
    double highLowDifference = 0.0;
    double K = 0.0;
    double DSum = 0.0;
    double D = 0.0;
    std::string KOutputName = getOutputMapping("K");

    dataSegment = getDataSegment(getInput("length"));
    dataSegmentLength = dataSegment.size();

    if (dataSegmentLength < getInput("length")) {
        return valueMap;
    }

    averageLengthDataSegment = std::vector<Tick>(dataSegment.begin() + (dataSegmentLength - getInput("averageLength")), dataSegment.begin() + dataSegmentLength);

    //low = ...
    //high = ...
    highLowDifference = high - low;
    K = highLowDifference > 0 ? 100 * ((lastTick.at("close") - low) / highLowDifference) : 0;

    // Calculate D.
    for (std::vector<Tick>::iterator iterator = averageLengthDataSegment.begin(); iterator != averageLengthDataSegment.end(); ++iterator) {
        if ((*iterator)[KOutputName] != 0) {
            DSum += (*iterator)[KOutputName];
        }
        else {
            DSum += K;
        }
    }
    D = DSum / averageLengthDataSegment.size();

    valueMap[KOutputName] = K;
    valueMap[getOutputMapping("D")] = D;

    return valueMap;
}
