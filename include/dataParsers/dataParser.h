#ifndef DATAPARSER_H
#define DATAPARSER_H

#include <vector>
#include <map>
#include <string>

class DataParser {
    public:
        virtual ~DataParser() {}
        virtual std::vector<std::map<std::string, double>> parse(std::string filePath) = 0;
};

#endif
