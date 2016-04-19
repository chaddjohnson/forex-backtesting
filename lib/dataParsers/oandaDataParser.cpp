#include "dataParsers/OandaDataParser.h"

OandaDataParser::OandaDataParser(std::string filePath) {
    this->filePath = filePath;
}

std::vector<Tick*> *OandaDataParser::parse() {
    std::vector<std::string> lines;
    std::vector<Tick*> *translatedLines = new std::vector<Tick*>();
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
        std::string item;
        std::vector<std::string> items;

        // Break the line into data items.
        while(std::getline(line, item, ',')) {
            items.push_back(item);
        }

        // Translate the data items.
        (*translatedLine)["timestamp"] = 1234567890.0;  // items.at(2);  // TODO
        (*translatedLine)["open"] = std::atof(items.at(3).c_str());
        (*translatedLine)["high"] = std::atof(items.at(4).c_str());
        (*translatedLine)["low"] = std::atof(items.at(5).c_str());
        (*translatedLine)["close"] = std::atof(items.at(6).c_str());

        // Add the translated line to the list of translated lines.
        translatedLines->push_back(translatedLine);
    }

    return translatedLines;
}
