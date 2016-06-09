#include "studies/stochasticOscillatorStudy.cuh"

Real StochasticOscillatorStudy::getLowestLow(std::vector<Tick*> *dataSegment) {
    Real lowest = 99999.0;
    Real current;

    for (std::vector<Tick*>::iterator iterator = dataSegment->begin(); iterator != dataSegment->end(); ++iterator) {
        current = (*iterator)->at("low");

        if (current < lowest) {
            lowest = current;
        }
    }

    return lowest;
}

Real StochasticOscillatorStudy::getHighestHigh(std::vector<Tick*> *dataSegment) {
    Real highest = 0.0;
    Real current;

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
    std::vector<Tick*> *dataSegment = nullptr;
    int dataSegmentLength = 0;
    std::vector<Tick*> averageLengthDataSegment;
    Real low = 0.0;
    Real high = 0.0;
    Real highLowDifference = 0.0;
    Real K = 0.0;
    Real DSum = 0.0;
    Real D = 0.0;
    std::string KOutputName = getOutputMapping("K");

    resetTickOutputs();

    dataSegment = getDataSegment(getInput("length"));
    dataSegmentLength = dataSegment->size();

    if (dataSegmentLength < getInput("length")) {
        setTickOutput(KOutputName, 0.0);
        setTickOutput(getOutputMapping("D"), 0.0);

        delete dataSegment;
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
