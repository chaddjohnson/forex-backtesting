#include "studies/emaStudy.h"

EmaStudy::EmaStudy(std::map<std::string, double> inputs, std::map<std::string, std::string> outputMap)
        : Study(inputs, outputMap) {
    this->previousEma = 0.0;
}

void EmaStudy::tick() {
    Tick *lastTick = getLastTick();
    std::vector<Tick*> *dataSegment = new std::vector<Tick*>();
    int dataSegmentLength = 0;
    double K = 0.0;
    double ema = 0.0;

    resetTickOutputs();

    dataSegment = getDataSegment(getInput("length"));
    dataSegmentLength = dataSegment->size();

    if (dataSegmentLength <= 1) {
        // Reset.
        this->previousEma = 0.0;
    }

    if (!this->previousEma) {
        // Use the last data item as the first previous EMA value.
        this->previousEma = lastTick->at("close");
    }

    // Calculate the EMA.
    K = 2 / (1 + getInput("length"));
    ema = (lastTick->at("close") * K) + (this->previousEma * (1 - K));

    // Set the new EMA just calculated as the previous EMA.
    this->previousEma = ema;

    setTickOutput(getOutputMapping("ema"), ema);

    // Free memory.
    delete dataSegment;
}
