#include "smaStudy.h"

SmaStudy::SmaStudy(std::map<std::string, double> &inputs, std::map<std::string, std::string> &outputMap)
    : Study(inputs, outputMap) {
}

std::map<std::string, double> SmaStudy::tick() {
    std::map<std::string, double> valueMap = {{"sma13", 47.71}};

    return valueMap;
}
