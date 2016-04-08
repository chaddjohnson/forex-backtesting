#ifndef SMASTUDY_H
#define SMASTUDY_H

#include <vector>
#include <map>
#include <string>
#include "../types/tick.h"
#include "study.h"

class SmaStudy : public Study {
    public:
        SmaStudy(std::map<std::string, double> &inputs, std::map<std::string, std::string> &outputMap)
            : Study(inputs, outputMap) {}
        std::map<std::string, double> tick();
};

#endif
