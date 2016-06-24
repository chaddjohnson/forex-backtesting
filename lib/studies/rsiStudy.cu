#include "studies/rsiStudy.cuh"

RsiStudy::RsiStudy(std::map<std::string, float> inputs, std::map<std::string, std::string> outputMap)
        : Study(inputs, outputMap) {
    this->dataSegmentLength = 0;
    this->previousAverageGain = -1.0;
    this->previousAverageLoss = -1.0;
}

float RsiStudy::calculateInitialAverageGain(Tick *initialTick, std::vector<Tick*> *dataSegment) {
    Tick *previousTick = initialTick;
    float sum = 0.0;
    float average = 0.0;

    for (std::vector<Tick*>::iterator iterator = dataSegment->begin(); iterator != dataSegment->end(); ++iterator) {
        sum += (*iterator)->at("close") > previousTick->at("close") ? (*iterator)->at("close") - previousTick->at("close") : 0;
        previousTick = *iterator;
    }

    average = sum / this->dataSegmentLength;

    return average;
}

float RsiStudy::calculateInitialAverageLoss(Tick *initialTick, std::vector<Tick*> *dataSegment) {
    Tick *previousTick = initialTick;
    float sum = 0.0;
    float average = 0.0;

    for (std::vector<Tick*>::iterator iterator = dataSegment->begin(); iterator != dataSegment->end(); ++iterator) {
        sum += (*iterator)->at("close") < previousTick->at("close") ? previousTick->at("close") - (*iterator)->at("close") : 0;
        previousTick = *iterator;
    }

    average = sum / this->dataSegmentLength;

    return average;
}

void RsiStudy::tick() {
    Tick *lastTick = getLastTick();
    std::vector<Tick*> *dataSegment = nullptr;
    Tick *previousTick = getPreviousTick();
    float currentGain = 0.0;
    float currentLoss = 0.0;
    float averageGain = 0.0;
    float averageLoss = 0.0;
    float RS = 0.0;
    float rsi = 0.0;

    resetTickOutputs();

    dataSegment = getDataSegment(getInput("length"));
    this->dataSegmentLength = dataSegment->size();

    if (this->dataSegmentLength < getInput("length")) {
        // Reset.
        this->previousAverageGain = -1.0;
        this->previousAverageLoss = -1.0;

        setTickOutput(getOutputMapping("rsi"), 0.0);

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
