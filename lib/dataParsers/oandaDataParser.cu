#include "dataParsers/oandaDataParser.cuh"

std::vector<Tick*> OandaDataParser::parse() {
    std::vector<std::string> lines;
    std::vector<Tick*> translatedLines;
    std::ifstream dataFile(getFilePath());

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
        Tick *translatedLine = new Tick();
        int i = 0;

        // Break the line into data items.
        while (std::getline(line, lineItem, ',')) {
            lineItems.push_back(lineItem);
        }

        // Translate the data items.
        (*translatedLine)["type"] = (double)getType();
        if (getType() == DataParser::types::BACKTEST) {
            (*translatedLine)["testingGroups"] = parseGroups(lineItems.at(i++));
            (*translatedLine)["validationGroups"] = parseGroups(lineItems.at(i++));
        }
        (*translatedLine)["timestamp"] = std::atof(lineItems.at(i++).c_str());
        (*translatedLine)["open"] = std::atof(lineItems.at(i++).c_str());
        (*translatedLine)["high"] = std::atof(lineItems.at(i++).c_str());
        (*translatedLine)["low"] = std::atof(lineItems.at(i++).c_str());
        (*translatedLine)["close"] = std::atof(lineItems.at(i++).c_str());

        // Add the translated line to the list of translated lines.
        translatedLines.push_back(translatedLine);
    }

    return translatedLines;
}
