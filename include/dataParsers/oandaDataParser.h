#ifndef OANDADATAPARSER_H
#define OANDADATAPARSER_H

#include <fstream>
#include <iterator>
#include <vector>
#include <string>
#include <sstream>
#include <cstdlib>
#include "dataParsers/dataParser.h"
#include "types/tick.h"

class OandaDataParser : public DataParser {
    public:
        OandaDataParser(std::string filePath) : DataParser(filePath) {}
        ~OandaDataParser() {}
        std::vector<Tick*> parse();
};

#endif
