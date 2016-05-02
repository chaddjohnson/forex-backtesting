#include "strategies/strategy.h"

Strategy::Strategy(std::string symbol, Configuration *configuration) {
    this->symbol = symbol;
    this->configuration = configuration;
}
