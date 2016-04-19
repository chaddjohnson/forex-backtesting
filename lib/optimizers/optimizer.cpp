#include "optimizers/optimizer.h"

Optimizer::Optimizer(std::string strategyName, std::string symbol, int group) {
    this->strategyName = strategyName;
    this->symbol = symbol;
    this->group = group;
}

void Optimizer::optimize(std::vector<Configuration>, double investment, double profitability) {
}

void Optimizer::prepareData(std::vector<std::map<std::string, double>> data) {
    this->prepareStudies();

    // for (std::vector<std::map<std::string, double>>::iterator iterator = parsedData.begin(); iterator != parsedData.end(); ++iterator) {
    //     std::cout << "timestamp: " << (*iterator).at("timestamp") << std::endl;
    //     std::cout << "open: " << (*iterator).at("open") << "    " << ((*iterator).at("open") + 1) << std::endl;
    //     std::cout << "high: " << (*iterator).at("high") << std::endl;
    //     std::cout << "low: " << (*iterator).at("low") << std::endl;
    //     std::cout << "close: " << (*iterator).at("close") << std::endl;
    //     std::cout << std::endl;
    // }

    // ...
}

std::vector<Configuration> Optimizer::buildConfigurations() {
    std::vector<Configuration> configurations;

    // ...

    return configurations;
}
