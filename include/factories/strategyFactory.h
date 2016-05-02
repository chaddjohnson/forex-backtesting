#ifndef STRATEGYFACTORY_H
#define STRATEGYFACTORY_H

#include <string>
#include "strategies/strategy.h"
#include "strategies/reversalsStrategy.h"
#include "types/configuration.h"

class StrategyFactory {
    public:
        static Strategy *create(std::string name, std::string symbol, int group, Configuration *configuration);
};

#endif
