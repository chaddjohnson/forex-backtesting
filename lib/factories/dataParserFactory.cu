#include "factories/dataParserFactory.cuh"

DataParser *DataParserFactory::create(std::string name, std::string filePath) {
    //if (name == "oanda") {
        return new OandaDataParser(filePath);
    //}
}
