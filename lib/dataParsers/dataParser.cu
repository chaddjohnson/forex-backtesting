#include "dataParsers/dataParser.cuh"

DataParser::DataParser(std::string filePath, int type) {
    this->filePath = filePath;
    this->type = type;
}

std::string DataParser::getFilePath() {
    return this->filePath;
}

int DataParser::getType() {
    return this->type;
}

double DataParser::parseGroups(std::string groupString) {
    double value = 0;

    std::stringstream stream(groupString);
    std::string groupItem;
    std::vector<int> groups;

    // Break the group string into individual items, and convert each item to an integer.
    while(std::getline(stream, groupItem, ';')) {
        groups.push_back(std::atoi(groupItem.c_str()));
    }

    for (std::vector<int>::iterator iterator = groups.begin(); iterator != groups.end(); ++iterator) {
        value += pow(2, *iterator);
    }

    return value;
}
