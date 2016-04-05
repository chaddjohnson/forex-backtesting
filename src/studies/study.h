#ifndef BASESTUDY_H
#define BASESTUDY_H

#include <map>
#include <vector>
#include <string>
#include "../types/tick.h"

class Study {
    private:
        std::vector<Tick *> data;
        std::map<std::string, double> inputs;
        std::map<std::string, std::string> outputMap;

    public:
        Study(std::map<std::string, double> &inputs, std::map<std::string, std::string> &outputMap);
        void setData(std::vector<Tick *> &data);
        double getInput(std::string key);
        virtual std::map<std::string, double> tick() = 0;
};

#endif
