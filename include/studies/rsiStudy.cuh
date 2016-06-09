#ifndef RSISTUDY_H
#define RSISTUDY_H

#include <vector>
#include <map>
#include <string>
#include "types/tick.cuh"
#include "study.cuh"
#include "types/real.cuh"

class RsiStudy : public Study {
    private:
        int dataSegmentLength;
        Real previousAverageGain;
        Real previousAverageLoss;

    public:
        RsiStudy(std::map<std::string, Real> inputs, std::map<std::string, std::string> outputMap);
        Real calculateInitialAverageGain(Tick *initialTick, std::vector<Tick*> *dataSegment);
        Real calculateInitialAverageLoss(Tick *initialTick, std::vector<Tick*> *dataSegment);
        void tick();
};

#endif
