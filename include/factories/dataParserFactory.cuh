#ifndef DATAPARSERFACTORY_H
#define DATAPARSERFACTORY_H

#include <string>
#include "dataParsers/dataParser.cuh"
#include "dataParsers/oandaDataParser.cuh"

class DataParserFactory {
    public:
        static DataParser *create(char *name, std::string filePath);
};

#endif
