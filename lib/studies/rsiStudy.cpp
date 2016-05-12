#include "studies/rsiStudy.h"

RsiStudy::RsiStudy(std::map<std::string, double> inputs, std::map<std::string, std::string> outputMap)
        : Study(inputs, outputMap) {
    this->dataSegmentLength = 0;
    this->previousAverageGain = -1.0;
    this->previousAverageLoss = -1.0;
}

double RsiStudy::calculateInitialAverageGain(Tick *initialTick, std::vector<Tick*> *dataSegment) {
    Tick *previousTick = initialTick;
    double sum = 0.0;
    double average = 0.0;

    for (std::vector<Tick*>::iterator iterator = dataSegment->begin(); iterator != dataSegment->end(); ++iterator) {
        sum += (*iterator)->at("close") > previousTick->at("close") ? (*iterator)->at("close") - previousTick->at("close") : 0;
        previousTick = *iterator;
    }

    average = sum / this->dataSegmentLength;

    return average;
}

double RsiStudy::calculateInitialAverageLoss(Tick *initialTick, std::vector<Tick*> *dataSegment) {
    Tick *previousTick = initialTick;
    double sum = 0.0;
    double average = 0.0;

    for (std::vector<Tick*>::iterator iterator = dataSegment->begin(); iterator != dataSegment->end(); ++iterator) {
        sum += (*iterator)->at("close") < previousTick->at("close") ? previousTick->at("close") - (*iterator)->at("close") : 0;
        previousTick = *iterator;
    }

    average = sum / this->dataSegmentLength;

    return average;
}

void RsiStudy::tick() {
    Tick *lastTick = getLastTick();
    std::vector<Tick*> *dataSegment = new std::vector<Tick*>();
    Tick *previousTick = getPreviousTick();
    double currentGain = 0.0;
    double currentLoss = 0.0;
    double averageGain = 0.0;
    double averageLoss = 0.0;
    double RS = 0.0;
    double rsi = 0.0;

    resetTickOutputs();

    dataSegment = getDataSegment(getInput("length"));
    this->dataSegmentLength = dataSegment->size();

    if (this->dataSegmentLength < getInput("length")) {
        // Reset.
        this->previousAverageGain = -1.0;
        this->previousAverageLoss = -1.0;

        delete dataSegment;

        return;
    }

    // Calculate the current gain and the current loss.
    currentGain = lastTick->at("close") > previousTick->at("close") ? lastTick->at("close") - previousTick->at("close") : 0;
    currentLoss = lastTick->at("close") < previousTick->at("close") ? previousTick->at("close") - lastTick->at("close") : 0;

    // Calculate the average gain and the average loss.
    if (this->previousAverageGain == -1.0 || this->previousAverageLoss == -1.0) {
        averageGain = this->previousAverageGain = calculateInitialAverageGain(lastTick, dataSegment);
        averageLoss = this->previousAverageLoss = calculateInitialAverageLoss(lastTick, dataSegment);
    }
    else {
        averageGain = this->previousAverageGain = ((this->previousAverageGain * (getInput("length") - 1)) + currentGain) / getInput("length");
        averageLoss = this->previousAverageLoss = ((this->previousAverageLoss * (getInput("length") - 1)) + currentLoss) / getInput("length");
    }

    // Calculate RS.
    RS = averageLoss > 0 ? averageGain / averageLoss : 0;

    // Calculate RSI.
    rsi = 100 - (100 / (1 + RS));

    setTickOutput(getOutputMapping("rsi"), rsi);

    // Free memory.
    delete dataSegment;
}
