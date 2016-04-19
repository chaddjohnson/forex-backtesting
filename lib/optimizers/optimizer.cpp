#include "optimizers/optimizer.h"

Optimizer::Optimizer(std::string strategyName, std::string symbol, int group) {
    this->strategyName = strategyName;
    this->symbol = symbol;
    this->group = group;
}

void Optimizer::prepareData(std::vector<Tick*> *data) {
    double percentage = 0;
    int dataCount = data->size();
    std::string studyProperty;
    std::map<std::string, double> studyTickValues;
    std::map<std::string, std::string> studyOutputMap;
    std::vector<Tick*> *cumulativeData = new std::vector<Tick*>();
    int i = 0;

    // Reserve space in advance for better performance
    cumulativeData->reserve(dataCount);

    // Prepare studies for use.
    this->prepareStudies();

    printf("Preparing data...");

    // Go through the data and run studies for each data item.
    for (std::vector<Tick*>::iterator dataIterator = data->begin(); dataIterator != data->end(); ++dataIterator) {
        printf("\rPreparing data...%i", ++i);

        // Append to the cumulative data.
        cumulativeData->push_back(*dataIterator);

        for (std::vector<Study*>::iterator studyIterator = this->studies.begin(); studyIterator != this->studies.end(); ++studyIterator) {
            // Update the data for the study.
            (*studyIterator)->setData(cumulativeData);

            // Process the latest data for the study.
            studyTickValues = (*studyIterator)->tick();
        }
    }
}

std::vector<Configuration> Optimizer::buildConfigurations() {
    std::vector<Configuration> configurations;

    // ...

    return configurations;
}

void Optimizer::optimize(std::vector<Configuration>, double investment, double profitability) {
}
