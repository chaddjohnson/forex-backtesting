#include "factories/strategyFactory.h"

Strategy *StrategyFactory::create(std::string name, std::string symbol, int group, Configuration *configuration) {
    //if (name == "reversals") {
        return new ReversalsStrategy(symbol, group, configuration);
    //}
}
