#ifndef STOCHASTICOSCILLATORSTUDY_H
#define STOCHASTICOSCILLATORSTUDY_H

#include <vector>
#include <map>
#include <string>
#include "types/tick.cuh"
#include "study.cuh"
#include "types/real.cuh"

class StochasticOscillatorStudy : public Study {
    private:
        Real getLowestLow(std::vector<Tick*> *dataSegment);
        Real getHighestHigh(std::vector<Tick*> *dataSegment);

    public:
        StochasticOscillatorStudy(std::map<std::string, Real> inputs, std::map<std::string, std::string> outputMap)
            : Study(inputs, outputMap) {}
        void tick();
};

#endif
