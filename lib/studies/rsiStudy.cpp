#include "studies/rsiStudy.h"

RsiStudy::RsiStudy(std::map<std::string, double> inputs, std::map<std::string, std::string> outputMap)
        : Study(inputs, outputMap) {
    dataSegmentLength = 0;
    previousAverageGain = -1.0;
    previousAverageLoss = -1.0;
}

double RsiStudy::calculateInitialAverageGain(Tick *initialTick, std::vector<Tick*> *dataSegment) {
    Tick *previousTick = initialTick;
    double sum = 0.0;
    double average = 0.0;

    for (std::vector<Tick*>::iterator iterator = dataSegment->begin(); iterator != dataSegment->end(); ++iterator) {
        sum += (*iterator)->at("close") > previousTick->at("close") ? (*iterator)->at("close") - previousTick->at("close") : 0;
        previousTick = *iterator;
    }

    average = sum / dataSegmentLength;

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

    average = sum / dataSegmentLength;

    return average;
}

std::map<std::string, double> RsiStudy::tick() {
    std::map<std::string, double> valueMap;
    std::vector<Tick*> *dataSegment = new std::vector<Tick*>();
    Tick *lastTick = getLastTick();
    Tick *previousTick = getPreviousTick();
    double currentGain = 0.0;
    double currentLoss = 0.0;
    double averageGain = 0.0;
    double averageLoss = 0.0;
    double RS = 0.0;
    double rsi = 0.0;

    dataSegment = getDataSegment(getInput("length"));
    dataSegmentLength = dataSegment->size();

    if (dataSegmentLength < getInput("length")) {
        return valueMap;
    }

    // Calculate the current gain and the current loss.
    currentGain = lastTick->at("close") > previousTick->at("close") ? lastTick->at("close") - previousTick->at("close") : 0;
    currentLoss = lastTick->at("close") < previousTick->at("close") ? previousTick->at("close") - lastTick->at("close") : 0;

    // Calculate the average gain and the average loss.
    if (previousAverageGain == -1.0 || previousAverageLoss == -1.0) {
        averageGain = previousAverageGain = calculateInitialAverageGain(lastTick, dataSegment);
        averageLoss = previousAverageLoss = calculateInitialAverageLoss(lastTick, dataSegment);
    }
    else {
        averageGain = previousAverageGain = ((previousAverageGain * (getInput("length") - 1)) + currentGain) / getInput("length");
        averageLoss = previousAverageLoss = ((previousAverageLoss * (getInput("length") - 1)) + currentLoss) / getInput("length");
    }

    // Calculate RS.
    RS = averageLoss > 0 ? averageGain / averageLoss : 0;

    // Calculate RSI.
    rsi = 100 - (100 / (1 + RS));

    valueMap[getOutputMapping("rsi")] = rsi;

    return valueMap;
}
