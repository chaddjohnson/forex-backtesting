#ifndef DATAPARSER_H
#define DATAPARSER_H

#include <vector>
#include <map>
#include <string>
#include <sstream>
#include <iterator>
#include <cmath>
#include <cstdlib>
#include "types/tick.cuh"

class DataParser {
    private:
        std::string filePath;

    protected:
        std::string getFilePath();
        double parseGroups(std::string groupString);

    public:
        DataParser(std::string filePath);
        virtual ~DataParser() {}
        virtual std::vector<Tick*> parse() = 0;
};

#endif
