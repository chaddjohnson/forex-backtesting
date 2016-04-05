#ifndef SMASTUDY_H
#define SMASTUDY_H

#include <map>
#include <string>
#include "study.cpp"

class SmaStudy : public Study {
    public:
        SmaStudy(std::map<std::string, double> &inputs, std::map<std::string, std::string> &outputMap);
        std::map<std::string, double> tick();
};

#endif
