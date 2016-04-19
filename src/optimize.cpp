#include <string>
#include "optimizers/optimizer.h"
#include "factories/optimizerFactory.h"
#include "types/configuration.h"

int main(int argc, char *argv[]) {
    // Optimizer settings and objects.
    std::string optimizerName = "reversals";
    std::string symbol = "AUDJPY";
    int group = 1;
    Optimizer *optimizer;
    std::vector<Configuration> configurations;

    // Perform optimization.
    optimizer = OptimizerFactory::create(optimizerName, symbol, group);
    configurations = optimizer->buildConfigurations();
    optimizer->optimize(configurations, 1000, 0.76);

    return 0;
}
