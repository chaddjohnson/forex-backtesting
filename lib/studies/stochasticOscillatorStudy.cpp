#include "studies/stochasticOscillatorStudy.h"

void StochasticOscillatorStudy::tick() {
    // Tick *lastTick = getLastTick();
    // std::vector<Tick*> *dataSegment = new std::vector<Tick*>();
    // int dataSegmentLength = 0;
    // std::vector<Tick*> averageLengthDataSegment;
    // double low = 0.0;
    // double high = 0.0;
    // double highLowDifference = 0.0;
    // double K = 0.0;
    // double DSum = 0.0;
    // double D = 0.0;
    // std::string KOutputName = getOutputMapping("K");

    // dataSegment = getDataSegment(getInput("length"));
    // dataSegmentLength = dataSegment->size();

    // if (dataSegmentLength < getInput("length")) {
    //     return;
    // }

    // averageLengthDataSegment = std::vector<Tick*>(dataSegment->begin() + (dataSegmentLength - getInput("averageLength")), dataSegment->begin() + dataSegmentLength);

    // //low = ...  // TODO
    // //high = ...  // TODO
    // highLowDifference = high - low;
    // K = highLowDifference > 0 ? 100 * ((lastTick->at("close") - low) / highLowDifference) : 0;

    // // Calculate D.
    // for (std::vector<Tick*>::iterator iterator = averageLengthDataSegment.begin(); iterator != averageLengthDataSegment.end(); ++iterator) {
    //     if ((*iterator)->at(KOutputName) != 0) {
    //         DSum += (*iterator)->at(KOutputName);
    //     }
    //     else {
    //         DSum += K;
    //     }
    // }
    // D = DSum / averageLengthDataSegment.size();

    // setTickOutput(KOutputName, K);
    // setTickOutput(getOutputMapping("D"), D);

    // // Free memory.
    // delete dataSegment;
}
