#include "factories/dataParserFactory.cuh"

DataParser DataParserFactory::create(const char *name, std::string filePath) {
    //if (name == "oanda") {
        return OandaDataParser::OandaDataParser(filePath);
    //}
}
