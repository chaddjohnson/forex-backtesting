#include "studies/smaStudy.h"

void SmaStudy::tick() {
    Tick *lastTick = getLastTick();
    std::vector<Tick*> *dataSegment = new std::vector<Tick*>();
    int dataSegmentLength = 0;
    double sum = 0.0;
    double sma = 0.0;

    dataSegment = getDataSegment(getInput("length"));
    dataSegmentLength = dataSegment->size();

    if (dataSegmentLength < getInput("length")) {
        return;
    }

    // Calculate the SMA.
    for (std::vector<Tick*>::iterator iterator = dataSegment->begin(); iterator != dataSegment->end(); ++iterator) {
        sum += (*iterator)->at("close");
    }
    sma = sum / dataSegmentLength;

    (*lastTick)[getOutputMapping("sma")] = sma;

    // Free memory.
    delete dataSegment;
}
