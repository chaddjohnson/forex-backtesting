#include "studies/stochasticOscillatorStudy.h"

double StochasticOscillatorStudy::getLowestLow(std::vector<Tick*> *dataSegment) {
    double lowest = 99999.0;
    double current;

    for (std::vector<Tick*>::iterator iterator = dataSegment->begin(); iterator != dataSegment->end(); ++iterator) {
        current = (*iterator)->at("low");

        if (current < lowest) {
            lowest = current;
        }
    }

    return lowest;
}

double StochasticOscillatorStudy::getHighestHigh(std::vector<Tick*> *dataSegment) {
    double highest = 0.0;
    double current;

    for (std::vector<Tick*>::iterator iterator = dataSegment->begin(); iterator != dataSegment->end(); ++iterator) {
        current = (*iterator)->at("high");

        if (current > highest) {
            highest = current;
        }
    }

    return highest;
}

void StochasticOscillatorStudy::tick() {
    Tick *lastTick = getLastTick();
    std::vector<Tick*> *dataSegment = new std::vector<Tick*>();
    int dataSegmentLength = 0;
    std::vector<Tick*> averageLengthDataSegment;
    double low = 0.0;
    double high = 0.0;
    double highLowDifference = 0.0;
    double K = 0.0;
    double DSum = 0.0;
    double D = 0.0;
    std::string KOutputName = getOutputMapping("K");

    dataSegment = getDataSegment(getInput("length"));
    dataSegmentLength = dataSegment->size();

    if (dataSegmentLength < getInput("length")) {
        return;
    }

    averageLengthDataSegment = std::vector<Tick*>(dataSegment->begin() + (dataSegmentLength - getInput("averageLength")), dataSegment->begin() + dataSegmentLength);

    low = getLowestLow(dataSegment);
    high = getHighestHigh(dataSegment);
    highLowDifference = high - low;
    K = highLowDifference > 0 ? 100 * ((lastTick->at("close") - low) / highLowDifference) : lastTick->at("close");

    // Calculate D.
    for (std::vector<Tick*>::iterator iterator = averageLengthDataSegment.begin(); iterator != averageLengthDataSegment.end(); ++iterator) {
        if ((*iterator)->find(KOutputName) != (*iterator)->end()) {
            DSum += (*iterator)->at(KOutputName);
        }
        else {
            DSum += K;
        }
    }
    D = DSum / averageLengthDataSegment.size();

    setTickOutput(KOutputName, K);
    setTickOutput(getOutputMapping("D"), D);

    // Free memory.
    delete dataSegment;
}
