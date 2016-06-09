#ifndef SMASTUDY_H
#define SMASTUDY_H

#include <vector>
#include <map>
#include <string>
#include "types/tick.cuh"
#include "study.cuh"
#include "types/real.cuh"

class SmaStudy : public Study {
    public:
        SmaStudy(std::map<std::string, Real> inputs, std::map<std::string, std::string> outputMap)
            : Study(inputs, outputMap) {}
        void tick();
};

#endif
