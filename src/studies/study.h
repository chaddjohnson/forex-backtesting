#ifndef BASESTUDY_H
#define BASESTUDY_H

#include <map>
#include <vector>
#include "../types/tick.h"

class Study {
    Study(std::map<std::string, double>, std::map<std::string, std::string>);
    std::vector<Tick *> &getData();
    void setData(Tick ticks[]);
    double getInput(std::string key);
};

#endif
