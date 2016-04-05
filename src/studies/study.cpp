#include "study.h"

Study::Study(std::map<std::string, double> &inputs, std::map<std::string, std::string> &outputMap) {
    this->inputs = inputs;
    this->outputMap = outputMap;
}

void Study::setData(std::vector<Tick *> &data) {
    this->data = data;
}

double Study::getInput(std::string key) {
    return this->inputs[key];
}
