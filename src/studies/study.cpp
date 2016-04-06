#include "study.h"

Study::Study(std::map<std::string, double> &inputs, std::map<std::string, std::string> &outputMap) {
    this->inputs = inputs;
    this->outputMap = outputMap;
}

void Study::setData(std::vector<Tick> &data) {
    this->data = data;
    this->dataCount = data.size();
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

std::vector<Tick> Study::getDataSegment(int length) {
    int dataSegmentLength = std::min(length, dataCount);

    return std::vector<Tick>(data.begin() + (dataCount - dataSegmentLength), data.begin() + dataCount);
}

Tick Study::getPrevious() {
    return data.end()[-2];
}

Tick Study::getLast() {
    return data.back();
}
