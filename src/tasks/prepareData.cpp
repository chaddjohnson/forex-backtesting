#include <iostream>
#include <map>
#include <string>
#include "../studies/smaStudy.cpp"

int main() {
    // Connect to the database.
    // ...

    // Parse the raw data file.
    // ...

    std::map<std::string, double> inputs = {{"length", 13.0}};
    std::map<std::string, std::string> outputMap = {{"sma", "sma13"}};
    SmaStudy study = SmaStudy::SmaStudy(inputs, outputMap);

    std::map<std::string, double> data = study.tick();
    std::cout << data["sma13"];

    return 0;
}
