#ifndef STOCHASTICOSCILLATORSTUDY_H
#define STOCHASTICOSCILLATORSTUDY_H

#include <vector>
#include <map>
#include <string>
#include "types/tick.cuh"
#include "study.cuh"

class StochasticOscillatorStudy : public Study {
    private:
        float getLowestLow(std::vector<Tick*> *dataSegment);
        float getHighestHigh(std::vector<Tick*> *dataSegment);

    public:
        StochasticOscillatorStudy(std::map<std::string, float> inputs, std::map<std::string, std::string> outputMap)
            : Study(inputs, outputMap) {}
        void tick();
};

#endif
