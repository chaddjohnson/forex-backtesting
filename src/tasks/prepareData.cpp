#include <iostream>
#include <vector>
#include <map>
#include <string>
#include "../types/tick.h"
#include "../studies/smaStudy.cpp"

int main() {
    // Connect to the database.
    // ...

    // Parse the raw data file.
    // ...

    std::map<std::string, double> inputs = {{"length", 13.0}};
    std::map<std::string, std::string> outputMap = {{"sma", "sma13"}};
    std::vector<Tick> data = {{{"close", 14.4}}, {{"close", 15.6}}, {{"close", 13.2}}, {{"close", 12.1}}, {{"close", 13.6}}, {{"close", 14.9}}, {{"close", 13.2}}, {{"close", 15.1}}, {{"close", 16.2}}, {{"close", 16.5}}, {{"close", 17.0}}, {{"close", 16.4}}, {{"close", 15.8}}, {{"close", 16.1}}, {{"close", 16.2}}, {{"close", 17.5}}, {{"close", 16.7}}, {{"close", 17.0}}};
    SmaStudy study = SmaStudy::SmaStudy(inputs, outputMap);

    study.setData(data);
    std::map<std::string, double> valueMap = study.tick();

    return 0;
}
