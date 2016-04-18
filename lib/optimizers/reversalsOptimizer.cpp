#include "optimizers/reversalsOptimizer.h"

void ReversalsOptimizer::prepareStudies() {
    this->studies.push_back(new SmaStudy({{"length", 13.0}}, {{"sma", "sma13"}}));
    this->studies.push_back(new EmaStudy({{"length", 50.0}}, {{"ema", "ema50"}}));
    this->studies.push_back(new EmaStudy({{"length", 100.0}}, {{"ema", "ema100"}}));
    this->studies.push_back(new EmaStudy({{"length", 200.0}}, {{"ema", "ema200"}}));
    this->studies.push_back(new RsiStudy({{"length", 2.0}}, {{"rsi", "rsi2"}}));
    this->studies.push_back(new RsiStudy({{"length", 5.0}}, {{"rsi", "rsi5"}}));
    this->studies.push_back(new RsiStudy({{"length", 7.0}}, {{"rsi", "rsi7"}}));
    this->studies.push_back(new RsiStudy({{"length", 9.0}}, {{"rsi", "rsi9"}}));
    this->studies.push_back(new RsiStudy({{"length", 14.0}}, {{"rsi", "rsi14"}}));
    this->studies.push_back(new StochasticOscillatorStudy({{"length", 5.0}, {"averageLength", 3.0}}, {{"K", "stochastic5K"}, {"D", "stochastic5D"}}));
    this->studies.push_back(new StochasticOscillatorStudy({{"length", 10.0}, {"averageLength", 3.0}}, {{"K", "stochastic10K"}, {"D", "stochastic10D"}}));
    this->studies.push_back(new StochasticOscillatorStudy({{"length", 14.0}, {"averageLength", 3.0}}, {{"K", "stochastic14K"}, {"D", "stochastic14D"}}));
    this->studies.push_back(new StochasticOscillatorStudy({{"length", 21.0}, {"averageLength", 3.0}}, {{"K", "stochastic21K"}, {"D", "stochastic21D"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 200.0}, {"degree", 2.0}, {"deviations", 1.95}}, {{"regression", "prChannel200_2_195"}, {"upper", "prChannelUpper200_2_195"}, {"lower", "prChannelLower200_2_195"}}));
}
