#include "optimizers/optimizer.h"

Optimizer::Optimizer(std::string strategyName, std::string symbol, int group) {
    this->strategyName = strategyName;
    this->symbol = symbol;
    this->group = group;
}

void Optimizer::prepareData(std::vector<Tick*> data) {
    double percentage;
    int dataCount = data.size();
    std::vector<Tick*> cumulativeData;
    std::vector<Tick*> tempCumulativeData;
    int i = 0;
    int j = 0;
    maginatics::ThreadPool pool(1, 8, 5000);

    // If there is a significant gap, save the current data points, and start over with recording.
    // TODO

    // Reserve space in advance for better performance
    cumulativeData.reserve(dataCount);

    // Prepare studies for use.
    this->prepareStudies();

    printf("Preparing data...");

    // Go through the data and run studies for each data item.
    for (std::vector<Tick*>::iterator dataIterator = data.begin(); dataIterator != data.end(); ++dataIterator) {
        percentage = (++i / (double)dataCount) * 100.0;
        printf("\rPreparing data...%0.4f%%", percentage);

        // Append to the cumulative data.
        cumulativeData.push_back(*dataIterator);

        for (std::vector<Study*>::iterator studyIterator = studies.begin(); studyIterator != studies.end(); ++studyIterator) {
            // Update the data for the study.
            (*studyIterator)->setData(&cumulativeData);

            pool.execute([&]() {
                // Source: http://stackoverflow.com/a/7854596/83897
                auto functor = [=]() {
                    // Process the latest data for the study.
                    (*studyIterator)->tick();
                };
            });
        }

        pool.drain();

        // Periodically free up memory.
        if (cumulativeData.size() >= 2000) {
            for (j=0; j<1000; j++) {
                delete cumulativeData[j];
            }

            // Extract the last 1000 elements into a new vector.
            std::vector<Tick*> tempCumulativeData(cumulativeData.begin() + (cumulativeData.size() - 1000), cumulativeData.end());

            // Release memory for the old vector.
            std::vector<Tick*>().swap(cumulativeData);
            //cumulativeData.shrink_to_fit();

            // Set the original to be the new vector.
            cumulativeData = tempCumulativeData;
        }
    }

    printf("\n");
}

std::vector<Configuration> Optimizer::buildConfigurations() {
    std::vector<Configuration> configurations;

    // ...

    return configurations;
}

void Optimizer::optimize(std::vector<Configuration>, double investment, double profitability) {
}
