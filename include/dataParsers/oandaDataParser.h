#ifndef OANDADATAPARSER_H
#define OANDADATAPARSER_H

#include <fstream>
#include <iterator>
#include <vector>
#include <sstream>
#include <stdlib.h>
#include <math.h>
#include "dataParsers/dataParser.h"
#include "types/tick.h"

class OandaDataParser : public DataParser {
    private:
        std::string filePath;

    protected:
        double parseGroups(std::string string);

    public:
        OandaDataParser(std::string filePath);
        std::vector<Tick*> parse();
};

#endif
