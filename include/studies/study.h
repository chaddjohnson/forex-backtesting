#ifndef STUDY_H
#define STUDY_H

#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include "types/tick.h"

class Study {
    private:
        std::vector<Tick> data;
        int dataLength;
        std::map<std::string, double> inputs;
        std::map<std::string, std::string> outputMap;

    public:
        Study(std::map<std::string, double> inputs, std::map<std::string, std::string> outputMap);
        void setData(std::vector<Tick> &data);
        double getInput(std::string key);
        std::map<std::string, std::string> &getOutputMap();
        std::string getOutputMapping(std::string key);
        std::vector<Tick> getDataSegment(int length);
        Tick getPreviousTick();
        Tick getLastTick();
        virtual std::map<std::string, double> tick() = 0;
};

#endif
