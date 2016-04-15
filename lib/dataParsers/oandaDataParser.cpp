#include "dataParsers/OandaDataParser.h"

OandaDataParser::OandaDataParser(std::string filePath) {
    this->filePath = filePath;
}

std::vector<std::map<std::string, double>> OandaDataParser::parse(std::string filePath) {
    std::vector<std::map<std::string, double>> formattedData;

    // ...

    return formattedData;
}
