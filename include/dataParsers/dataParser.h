#ifndef DATAPARSER_H
#define DATAPARSER_H

#include <vector>
#include <map>
#include <string>
#include "types/tick.h"

class DataParser {
    public:
        virtual ~DataParser() {}
        virtual std::vector<Tick*> *parse() = 0;
};

#endif
