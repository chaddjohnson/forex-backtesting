#include "factories/dataParserFactory.cuh"

DataParser *DataParserFactory::create(std::string name, std::string filePath, int type) {
    //if (name == "oanda") {
        return new OandaDataParser(filePath, type);
    //}
}
