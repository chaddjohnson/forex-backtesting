#ifndef OANDADATAPARSER_H
#define OANDADATAPARSER_H

#include <fstream>
#include <iterator>
#include <vector>
#include <sstream>
#include <iostream>
#include "dataParsers/dataParser.h"

class OandaDataParser : public DataParser {
    private:
        std::string filePath;

    public:
        OandaDataParser(std::string filePath);
        std::vector<std::map<std::string, double>> parse();
};

#endif
