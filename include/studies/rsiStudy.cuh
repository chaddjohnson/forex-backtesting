#ifndef RSISTUDY_H
#define RSISTUDY_H

#include <vector>
#include <map>
#include <string>
#include "types/tick.cuh"
#include "study.cuh"

class RsiStudy : public Study {
    private:
        int dataSegmentLength;
        float previousAverageGain;
        float previousAverageLoss;

    public:
        RsiStudy(std::map<std::string, float> inputs, std::map<std::string, std::string> outputMap);
        float calculateInitialAverageGain(Tick *initialTick, std::vector<Tick*> *dataSegment);
        float calculateInitialAverageLoss(Tick *initialTick, std::vector<Tick*> *dataSegment);
        void tick();
};

#endif
