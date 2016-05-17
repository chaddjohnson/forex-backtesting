#include "factories/dataParserFactory.cuh"

DataParser *DataParserFactory::create(char *name, std::string filePath) {
    //if (name == "oanda") {
        return new OandaDataParser(filePath);
    //}
}
