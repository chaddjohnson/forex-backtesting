#ifndef DATAPARSERFACTORY_H
#define DATAPARSERFACTORY_H

#include <string>
#include "dataParsers/dataParser.h"
#include "dataParsers/oandaDataParser.h"

class DataParserFactory {
    public:
        static DataParser *create(std::string name, std::string filePath);
};

#endif
