#ifndef STOCHASTICOSCILLATORSTUDY_H
#define STOCHASTICOSCILLATORSTUDY_H

#include <vector>
#include <map>
#include <string>
#include "types/tick.h"
#include "study.h"

class StochasticOscillatorStudy : public Study {
    public:
        StochasticOscillatorStudy(std::map<std::string, double> inputs, std::map<std::string, std::string> outputMap)
            : Study(inputs, outputMap) {}
        void tick();
};

#endif
