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
    std::vector<Tick> data = {{14.4}, {15.6}, {13.2}, {12.1}, {13.6}, {14.9}, {13.2}, {15.1}, {16.2}, {16.5}, {17.0}, {16.4}, {15.8}, {16.1}, {16.2}, {17.5}, {16.7}, {17.0}};
    SmaStudy study = SmaStudy::SmaStudy(inputs, outputMap);

    study.setData(data);
    std::map<std::string, double> valueMap = study.tick();

    return 0;
}
