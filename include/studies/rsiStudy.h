#ifndef RSISTUDY_H
#define RSISTUDY_H

#include <vector>
#include <map>
#include <string>
#include "types/tick.h"
#include "study.h"

class RsiStudy : public Study {
    private:
        int dataSegmentLength;
        double previousAverageGain;
        double previousAverageLoss;

    public:
        RsiStudy(std::map<std::string, double> inputs, std::map<std::string, std::string> outputMap);
        double calculateInitialAverageGain(Tick *initialTick, std::vector<Tick*> *dataSegment);
        double calculateInitialAverageLoss(Tick *initialTick, std::vector<Tick*> *dataSegment);
        std::map<std::string, double> tick();
};

#endif
