#include "studies/smaStudy.cuh"

void SmaStudy::tick() {
    std::vector<Tick*> *dataSegment = nullptr;
    int dataSegmentLength = 0;
    double sum = 0.0;
    double sma = 0.0;

    resetTickOutputs();

    dataSegment = getDataSegment(getInput("length"));
    dataSegmentLength = dataSegment->size();

    if (dataSegmentLength < getInput("length")) {
        delete dataSegment;
        return;
    }

    // Calculate the SMA.
    for (std::vector<Tick*>::iterator iterator = dataSegment->begin(); iterator != dataSegment->end(); ++iterator) {
        sum += (*iterator)->at("close");
    }
    sma = sum / dataSegmentLength;

    setTickOutput(getOutputMapping("sma"), sma);

    // Free memory.
    delete dataSegment;
}
