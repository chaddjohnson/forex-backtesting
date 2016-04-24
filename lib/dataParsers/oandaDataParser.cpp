#include "dataParsers/oandaDataParser.h"

OandaDataParser::OandaDataParser(std::string filePath) {
    this->filePath = filePath;
}

double OandaDataParser::parseGroups(std::string groupString) {
    double value = 0;

    std::stringstream stream(groupString);
    std::string groupItem;
    std::vector<int> groups;

    // Break the group string into individual items, and convert each item to an integer.
    while(std::getline(stream, groupItem, ';')) {
        groups.push_back(std::atoi(groupItem.c_str()));
    }

    for (std::vector<int>::iterator iterator = groups.begin(); iterator != groups.end(); ++iterator) {
        value += pow(2, (*iterator) - 1);
    }

    return value;
}

std::vector<Tick*> OandaDataParser::parse() {
    std::vector<std::string> lines;
    std::vector<Tick*> translatedLines;
    Tick *translatedLine = new Tick();
    std::ifstream dataFile(this->filePath);

    // Read lines from the file into the vector.
    // Source: http://stackoverflow.com/a/8365247/83897
    std::copy(std::istream_iterator<std::string>(dataFile),
              std::istream_iterator<std::string>(),
              std::back_inserter(lines));

    // Process each line.
    for (std::vector<std::string>::iterator iterator = lines.begin(); iterator != lines.end(); ++iterator) {
        std::stringstream line(*iterator);
        std::string lineItem;
        std::vector<std::string> lineItems;

        // Break the line into data items.
        while(std::getline(line, lineItem, ',')) {
            lineItems.push_back(lineItem);
        }

        // Initialize a new tick.
        translatedLine = new Tick();

        // Translate the data items.
        (*translatedLine)["testingGroup"] = parseGroups(lineItems.at(0));
        (*translatedLine)["validationGroup"] = parseGroups(lineItems.at(1));
        (*translatedLine)["timestamp"] = 1234567890.0;  // TODO
        (*translatedLine)["open"] = std::atof(lineItems.at(3).c_str());
        (*translatedLine)["high"] = std::atof(lineItems.at(4).c_str());
        (*translatedLine)["low"] = std::atof(lineItems.at(5).c_str());
        (*translatedLine)["close"] = std::atof(lineItems.at(6).c_str());

        // Add the translated line to the list of translated lines.
        translatedLines.push_back(translatedLine);
    }

    return translatedLines;
}
