#ifndef OANDADATAPARSER_H
#define OANDADATAPARSER_H

#include <fstream>
#include "dataParsers/dataParser.h"

class OandaDataParser : public DataParser {
    private:
        std::string filePath;

    public:
        OandaDataParser(std::string filePath);
        virtual std::vector<std::map<std::string, double>> parse(std::string filePath) = 0;
};

#endif
