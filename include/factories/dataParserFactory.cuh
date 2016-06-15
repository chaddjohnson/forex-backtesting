#ifndef DATAPARSERFACTORY_H
#define DATAPARSERFACTORY_H

#include <string>
#include "dataParsers/dataParser.cuh"
#include "dataParsers/oandaDataParser.cuh"

class DataParserFactory {
    public:
        static DataParser *create(std::string name, std::string filePath, int type);
};

#endif
