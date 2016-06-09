#ifndef EMASTUDY_H
#define EMASTUDY_H

#include <map>
#include <string>
#include "types/tick.cuh"
#include "study.cuh"
#include "types/real.cuh"

class EmaStudy : public Study {
    private:
        Real previousEma;

    public:
        EmaStudy(std::map<std::string, Real> inputs, std::map<std::string, std::string> outputMap);
        void tick();
};

#endif
