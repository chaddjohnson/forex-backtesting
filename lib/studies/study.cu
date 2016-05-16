#include "studies/study.cuh"

Study::Study(std::map<std::string, double> inputs, std::map<std::string, std::string> outputMap) {
    this->inputs = inputs;
    this->outputMap = outputMap;
}

void Study::setData(std::vector<Tick*> *data) {
    this->data = data;
    this->dataLength = data->size();
}

double Study::getInput(std::string key) {
    return this->inputs[key];
}

std::map<std::string, std::string> &Study::getOutputMap() {
    return outputMap;
}

std::string Study::getOutputMapping(std::string key) {
    return this->outputMap[key];
}

std::vector<Tick*> *Study::getDataSegment(int length) {
    int dataSegmentLength = std::min(length, dataLength);
    std::vector<Tick*> *dataSegment = new std::vector<Tick*>(data->begin() + (dataLength - dataSegmentLength), data->end());

    return dataSegment;
}

Tick *Study::getPreviousTick() {
    return data->end()[-2];
}

Tick *Study::getLastTick() {
    return data->back();
}

std::map<std::string, double> Study::getTickOutputs() {
    return this->tickOutputs;
}

void Study::setTickOutput(std::string key, double value) {
    this->tickOutputs[key] = value;
}

void Study::resetTickOutputs() {
    std::map<std::string, double>().swap(this->tickOutputs);
}
