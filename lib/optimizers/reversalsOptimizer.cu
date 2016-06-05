#include "optimizers/reversalsOptimizer.cuh"

std::vector<Study*> ReversalsOptimizer::getStudies() {
    if (this->studies.size() > 0) {
        // Studies have already been prepared.
        return this->studies;
    }

    this->studies.push_back(new SmaStudy({{"length", 13.0}}, {{"sma", "sma13"}}));
    this->studies.push_back(new EmaStudy({{"length", 50.0}}, {{"ema", "ema50"}}));
    this->studies.push_back(new EmaStudy({{"length", 100.0}}, {{"ema", "ema100"}}));
    this->studies.push_back(new EmaStudy({{"length", 200.0}}, {{"ema", "ema200"}}));
    // this->studies.push_back(new EmaStudy({{"length", 250.0}}, {{"ema", "ema250"}}));
    // this->studies.push_back(new EmaStudy({{"length", 300.0}}, {{"ema", "ema300"}}));
    // this->studies.push_back(new EmaStudy({{"length", 350.0}}, {{"ema", "ema350"}}));
    // this->studies.push_back(new EmaStudy({{"length", 400.0}}, {{"ema", "ema400"}}));
    // this->studies.push_back(new EmaStudy({{"length", 450.0}}, {{"ema", "ema450"}}));
    // this->studies.push_back(new EmaStudy({{"length", 500.0}}, {{"ema", "ema500"}}));
    this->studies.push_back(new RsiStudy({{"length", 2.0}}, {{"rsi", "rsi2"}}));
    this->studies.push_back(new RsiStudy({{"length", 5.0}}, {{"rsi", "rsi5"}}));
    this->studies.push_back(new RsiStudy({{"length", 7.0}}, {{"rsi", "rsi7"}}));
    this->studies.push_back(new RsiStudy({{"length", 9.0}}, {{"rsi", "rsi9"}}));
    this->studies.push_back(new RsiStudy({{"length", 14.0}}, {{"rsi", "rsi14"}}));
    this->studies.push_back(new StochasticOscillatorStudy({{"length", 5.0}, {"averageLength", 3.0}}, {{"K", "stochastic5K"}, {"D", "stochastic5D"}}));
    this->studies.push_back(new StochasticOscillatorStudy({{"length", 10.0}, {"averageLength", 3.0}}, {{"K", "stochastic10K"}, {"D", "stochastic10D"}}));
    this->studies.push_back(new StochasticOscillatorStudy({{"length", 14.0}, {"averageLength", 3.0}}, {{"K", "stochastic14K"}, {"D", "stochastic14D"}}));
    this->studies.push_back(new StochasticOscillatorStudy({{"length", 21.0}, {"averageLength", 3.0}}, {{"K", "stochastic21K"}, {"D", "stochastic21D"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 100.0}, {"degree", 2.0}, {"deviations", 1.90}}, {{"regression", "prChannel100_2_190"}, {"upper", "prChannelUpper100_2_190"}, {"lower", "prChannelLower100_2_190"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 100.0}, {"degree", 2.0}, {"deviations", 1.95}}, {{"regression", "prChannel100_2_195"}, {"upper", "prChannelUpper100_2_195"}, {"lower", "prChannelLower100_2_195"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 100.0}, {"degree", 2.0}, {"deviations", 2.00}}, {{"regression", "prChannel100_2_200"}, {"upper", "prChannelUpper100_2_200"}, {"lower", "prChannelLower100_2_200"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 100.0}, {"degree", 2.0}, {"deviations", 2.05}}, {{"regression", "prChannel100_2_205"}, {"upper", "prChannelUpper100_2_205"}, {"lower", "prChannelLower100_2_205"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 100.0}, {"degree", 2.0}, {"deviations", 2.10}}, {{"regression", "prChannel100_2_210"}, {"upper", "prChannelUpper100_2_210"}, {"lower", "prChannelLower100_2_210"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 100.0}, {"degree", 2.0}, {"deviations", 2.15}}, {{"regression", "prChannel100_2_215"}, {"upper", "prChannelUpper100_2_215"}, {"lower", "prChannelLower100_2_215"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 100.0}, {"degree", 2.0}, {"deviations", 2.20}}, {{"regression", "prChannel100_2_220"}, {"upper", "prChannelUpper100_2_220"}, {"lower", "prChannelLower100_2_220"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 100.0}, {"degree", 2.0}, {"deviations", 2.25}}, {{"regression", "prChannel100_2_225"}, {"upper", "prChannelUpper100_2_225"}, {"lower", "prChannelLower100_2_225"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 100.0}, {"degree", 2.0}, {"deviations", 2.30}}, {{"regression", "prChannel100_2_230"}, {"upper", "prChannelUpper100_2_230"}, {"lower", "prChannelLower100_2_230"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 100.0}, {"degree", 2.0}, {"deviations", 2.35}}, {{"regression", "prChannel100_2_235"}, {"upper", "prChannelUpper100_2_235"}, {"lower", "prChannelLower100_2_235"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 100.0}, {"degree", 2.0}, {"deviations", 2.40}}, {{"regression", "prChannel100_2_240"}, {"upper", "prChannelUpper100_2_240"}, {"lower", "prChannelLower100_2_240"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 100.0}, {"degree", 3.0}, {"deviations", 1.90}}, {{"regression", "prChannel100_3_190"}, {"upper", "prChannelUpper100_3_190"}, {"lower", "prChannelLower100_3_190"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 100.0}, {"degree", 3.0}, {"deviations", 1.95}}, {{"regression", "prChannel100_3_195"}, {"upper", "prChannelUpper100_3_195"}, {"lower", "prChannelLower100_3_195"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 100.0}, {"degree", 3.0}, {"deviations", 2.00}}, {{"regression", "prChannel100_3_200"}, {"upper", "prChannelUpper100_3_200"}, {"lower", "prChannelLower100_3_200"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 100.0}, {"degree", 3.0}, {"deviations", 2.05}}, {{"regression", "prChannel100_3_205"}, {"upper", "prChannelUpper100_3_205"}, {"lower", "prChannelLower100_3_205"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 100.0}, {"degree", 3.0}, {"deviations", 2.10}}, {{"regression", "prChannel100_3_210"}, {"upper", "prChannelUpper100_3_210"}, {"lower", "prChannelLower100_3_210"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 100.0}, {"degree", 3.0}, {"deviations", 2.15}}, {{"regression", "prChannel100_3_215"}, {"upper", "prChannelUpper100_3_215"}, {"lower", "prChannelLower100_3_215"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 100.0}, {"degree", 3.0}, {"deviations", 2.20}}, {{"regression", "prChannel100_3_220"}, {"upper", "prChannelUpper100_3_220"}, {"lower", "prChannelLower100_3_220"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 100.0}, {"degree", 3.0}, {"deviations", 2.25}}, {{"regression", "prChannel100_3_225"}, {"upper", "prChannelUpper100_3_225"}, {"lower", "prChannelLower100_3_225"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 100.0}, {"degree", 3.0}, {"deviations", 2.30}}, {{"regression", "prChannel100_3_230"}, {"upper", "prChannelUpper100_3_230"}, {"lower", "prChannelLower100_3_230"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 100.0}, {"degree", 3.0}, {"deviations", 2.35}}, {{"regression", "prChannel100_3_235"}, {"upper", "prChannelUpper100_3_235"}, {"lower", "prChannelLower100_3_235"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 100.0}, {"degree", 3.0}, {"deviations", 2.40}}, {{"regression", "prChannel100_3_240"}, {"upper", "prChannelUpper100_3_240"}, {"lower", "prChannelLower100_3_240"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 100.0}, {"degree", 4.0}, {"deviations", 1.90}}, {{"regression", "prChannel100_4_190"}, {"upper", "prChannelUpper100_4_190"}, {"lower", "prChannelLower100_4_190"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 100.0}, {"degree", 4.0}, {"deviations", 1.95}}, {{"regression", "prChannel100_4_195"}, {"upper", "prChannelUpper100_4_195"}, {"lower", "prChannelLower100_4_195"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 100.0}, {"degree", 4.0}, {"deviations", 2.00}}, {{"regression", "prChannel100_4_200"}, {"upper", "prChannelUpper100_4_200"}, {"lower", "prChannelLower100_4_200"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 100.0}, {"degree", 4.0}, {"deviations", 2.05}}, {{"regression", "prChannel100_4_205"}, {"upper", "prChannelUpper100_4_205"}, {"lower", "prChannelLower100_4_205"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 100.0}, {"degree", 4.0}, {"deviations", 2.10}}, {{"regression", "prChannel100_4_210"}, {"upper", "prChannelUpper100_4_210"}, {"lower", "prChannelLower100_4_210"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 100.0}, {"degree", 4.0}, {"deviations", 2.15}}, {{"regression", "prChannel100_4_215"}, {"upper", "prChannelUpper100_4_215"}, {"lower", "prChannelLower100_4_215"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 100.0}, {"degree", 4.0}, {"deviations", 2.20}}, {{"regression", "prChannel100_4_220"}, {"upper", "prChannelUpper100_4_220"}, {"lower", "prChannelLower100_4_220"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 100.0}, {"degree", 4.0}, {"deviations", 2.25}}, {{"regression", "prChannel100_4_225"}, {"upper", "prChannelUpper100_4_225"}, {"lower", "prChannelLower100_4_225"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 100.0}, {"degree", 4.0}, {"deviations", 2.30}}, {{"regression", "prChannel100_4_230"}, {"upper", "prChannelUpper100_4_230"}, {"lower", "prChannelLower100_4_230"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 100.0}, {"degree", 4.0}, {"deviations", 2.35}}, {{"regression", "prChannel100_4_235"}, {"upper", "prChannelUpper100_4_235"}, {"lower", "prChannelLower100_4_235"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 100.0}, {"degree", 4.0}, {"deviations", 2.40}}, {{"regression", "prChannel100_4_240"}, {"upper", "prChannelUpper100_4_240"}, {"lower", "prChannelLower100_4_240"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 100.0}, {"degree", 5.0}, {"deviations", 2.05}}, {{"regression", "prChannel100_5_205"}, {"upper", "prChannelUpper100_5_205"}, {"lower", "prChannelLower100_5_205"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 200.0}, {"degree", 2.0}, {"deviations", 1.90}}, {{"regression", "prChannel200_2_190"}, {"upper", "prChannelUpper200_2_190"}, {"lower", "prChannelLower200_2_190"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 200.0}, {"degree", 2.0}, {"deviations", 1.95}}, {{"regression", "prChannel200_2_195"}, {"upper", "prChannelUpper200_2_195"}, {"lower", "prChannelLower200_2_195"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 200.0}, {"degree", 2.0}, {"deviations", 2.00}}, {{"regression", "prChannel200_2_200"}, {"upper", "prChannelUpper200_2_200"}, {"lower", "prChannelLower200_2_200"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 200.0}, {"degree", 2.0}, {"deviations", 2.05}}, {{"regression", "prChannel200_2_205"}, {"upper", "prChannelUpper200_2_205"}, {"lower", "prChannelLower200_2_205"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 200.0}, {"degree", 2.0}, {"deviations", 2.10}}, {{"regression", "prChannel200_2_210"}, {"upper", "prChannelUpper200_2_210"}, {"lower", "prChannelLower200_2_210"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 200.0}, {"degree", 2.0}, {"deviations", 2.15}}, {{"regression", "prChannel200_2_215"}, {"upper", "prChannelUpper200_2_215"}, {"lower", "prChannelLower200_2_215"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 200.0}, {"degree", 2.0}, {"deviations", 2.20}}, {{"regression", "prChannel200_2_220"}, {"upper", "prChannelUpper200_2_220"}, {"lower", "prChannelLower200_2_220"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 200.0}, {"degree", 2.0}, {"deviations", 2.25}}, {{"regression", "prChannel200_2_225"}, {"upper", "prChannelUpper200_2_225"}, {"lower", "prChannelLower200_2_225"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 200.0}, {"degree", 2.0}, {"deviations", 2.30}}, {{"regression", "prChannel200_2_230"}, {"upper", "prChannelUpper200_2_230"}, {"lower", "prChannelLower200_2_230"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 200.0}, {"degree", 2.0}, {"deviations", 2.35}}, {{"regression", "prChannel200_2_235"}, {"upper", "prChannelUpper200_2_235"}, {"lower", "prChannelLower200_2_235"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 200.0}, {"degree", 2.0}, {"deviations", 2.40}}, {{"regression", "prChannel200_2_240"}, {"upper", "prChannelUpper200_2_240"}, {"lower", "prChannelLower200_2_240"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 200.0}, {"degree", 3.0}, {"deviations", 1.90}}, {{"regression", "prChannel200_3_190"}, {"upper", "prChannelUpper200_3_190"}, {"lower", "prChannelLower200_3_190"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 200.0}, {"degree", 3.0}, {"deviations", 1.95}}, {{"regression", "prChannel200_3_195"}, {"upper", "prChannelUpper200_3_195"}, {"lower", "prChannelLower200_3_195"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 200.0}, {"degree", 3.0}, {"deviations", 2.00}}, {{"regression", "prChannel200_3_200"}, {"upper", "prChannelUpper200_3_200"}, {"lower", "prChannelLower200_3_200"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 200.0}, {"degree", 3.0}, {"deviations", 2.05}}, {{"regression", "prChannel200_3_205"}, {"upper", "prChannelUpper200_3_205"}, {"lower", "prChannelLower200_3_205"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 200.0}, {"degree", 3.0}, {"deviations", 2.10}}, {{"regression", "prChannel200_3_210"}, {"upper", "prChannelUpper200_3_210"}, {"lower", "prChannelLower200_3_210"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 200.0}, {"degree", 3.0}, {"deviations", 2.15}}, {{"regression", "prChannel200_3_215"}, {"upper", "prChannelUpper200_3_215"}, {"lower", "prChannelLower200_3_215"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 200.0}, {"degree", 3.0}, {"deviations", 2.20}}, {{"regression", "prChannel200_3_220"}, {"upper", "prChannelUpper200_3_220"}, {"lower", "prChannelLower200_3_220"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 200.0}, {"degree", 3.0}, {"deviations", 2.25}}, {{"regression", "prChannel200_3_225"}, {"upper", "prChannelUpper200_3_225"}, {"lower", "prChannelLower200_3_225"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 200.0}, {"degree", 3.0}, {"deviations", 2.30}}, {{"regression", "prChannel200_3_230"}, {"upper", "prChannelUpper200_3_230"}, {"lower", "prChannelLower200_3_230"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 200.0}, {"degree", 3.0}, {"deviations", 2.35}}, {{"regression", "prChannel200_3_235"}, {"upper", "prChannelUpper200_3_235"}, {"lower", "prChannelLower200_3_235"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 200.0}, {"degree", 3.0}, {"deviations", 2.40}}, {{"regression", "prChannel200_3_240"}, {"upper", "prChannelUpper200_3_240"}, {"lower", "prChannelLower200_3_240"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 200.0}, {"degree", 4.0}, {"deviations", 1.90}}, {{"regression", "prChannel200_4_190"}, {"upper", "prChannelUpper200_4_190"}, {"lower", "prChannelLower200_4_190"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 200.0}, {"degree", 4.0}, {"deviations", 1.95}}, {{"regression", "prChannel200_4_195"}, {"upper", "prChannelUpper200_4_195"}, {"lower", "prChannelLower200_4_195"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 200.0}, {"degree", 4.0}, {"deviations", 2.00}}, {{"regression", "prChannel200_4_200"}, {"upper", "prChannelUpper200_4_200"}, {"lower", "prChannelLower200_4_200"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 200.0}, {"degree", 4.0}, {"deviations", 2.05}}, {{"regression", "prChannel200_4_205"}, {"upper", "prChannelUpper200_4_205"}, {"lower", "prChannelLower200_4_205"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 200.0}, {"degree", 4.0}, {"deviations", 2.10}}, {{"regression", "prChannel200_4_210"}, {"upper", "prChannelUpper200_4_210"}, {"lower", "prChannelLower200_4_210"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 200.0}, {"degree", 4.0}, {"deviations", 2.15}}, {{"regression", "prChannel200_4_215"}, {"upper", "prChannelUpper200_4_215"}, {"lower", "prChannelLower200_4_215"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 200.0}, {"degree", 4.0}, {"deviations", 2.20}}, {{"regression", "prChannel200_4_220"}, {"upper", "prChannelUpper200_4_220"}, {"lower", "prChannelLower200_4_220"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 200.0}, {"degree", 4.0}, {"deviations", 2.25}}, {{"regression", "prChannel200_4_225"}, {"upper", "prChannelUpper200_4_225"}, {"lower", "prChannelLower200_4_225"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 200.0}, {"degree", 4.0}, {"deviations", 2.30}}, {{"regression", "prChannel200_4_230"}, {"upper", "prChannelUpper200_4_230"}, {"lower", "prChannelLower200_4_230"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 200.0}, {"degree", 4.0}, {"deviations", 2.35}}, {{"regression", "prChannel200_4_235"}, {"upper", "prChannelUpper200_4_235"}, {"lower", "prChannelLower200_4_235"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 200.0}, {"degree", 4.0}, {"deviations", 2.40}}, {{"regression", "prChannel200_4_240"}, {"upper", "prChannelUpper200_4_240"}, {"lower", "prChannelLower200_4_240"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 200.0}, {"degree", 5.0}, {"deviations", 2.05}}, {{"regression", "prChannel200_5_205"}, {"upper", "prChannelUpper200_5_205"}, {"lower", "prChannelLower200_5_205"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 250.0}, {"degree", 2.0}, {"deviations", 1.90}}, {{"regression", "prChannel250_2_190"}, {"upper", "prChannelUpper250_2_190"}, {"lower", "prChannelLower250_2_190"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 250.0}, {"degree", 2.0}, {"deviations", 1.95}}, {{"regression", "prChannel250_2_195"}, {"upper", "prChannelUpper250_2_195"}, {"lower", "prChannelLower250_2_195"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 250.0}, {"degree", 2.0}, {"deviations", 2.00}}, {{"regression", "prChannel250_2_200"}, {"upper", "prChannelUpper250_2_200"}, {"lower", "prChannelLower250_2_200"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 250.0}, {"degree", 2.0}, {"deviations", 2.05}}, {{"regression", "prChannel250_2_205"}, {"upper", "prChannelUpper250_2_205"}, {"lower", "prChannelLower250_2_205"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 250.0}, {"degree", 2.0}, {"deviations", 2.10}}, {{"regression", "prChannel250_2_210"}, {"upper", "prChannelUpper250_2_210"}, {"lower", "prChannelLower250_2_210"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 250.0}, {"degree", 2.0}, {"deviations", 2.15}}, {{"regression", "prChannel250_2_215"}, {"upper", "prChannelUpper250_2_215"}, {"lower", "prChannelLower250_2_215"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 250.0}, {"degree", 2.0}, {"deviations", 2.20}}, {{"regression", "prChannel250_2_220"}, {"upper", "prChannelUpper250_2_220"}, {"lower", "prChannelLower250_2_220"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 250.0}, {"degree", 2.0}, {"deviations", 2.25}}, {{"regression", "prChannel250_2_225"}, {"upper", "prChannelUpper250_2_225"}, {"lower", "prChannelLower250_2_225"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 250.0}, {"degree", 2.0}, {"deviations", 2.30}}, {{"regression", "prChannel250_2_230"}, {"upper", "prChannelUpper250_2_230"}, {"lower", "prChannelLower250_2_230"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 250.0}, {"degree", 2.0}, {"deviations", 2.35}}, {{"regression", "prChannel250_2_235"}, {"upper", "prChannelUpper250_2_235"}, {"lower", "prChannelLower250_2_235"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 250.0}, {"degree", 2.0}, {"deviations", 2.40}}, {{"regression", "prChannel250_2_240"}, {"upper", "prChannelUpper250_2_240"}, {"lower", "prChannelLower250_2_240"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 250.0}, {"degree", 3.0}, {"deviations", 1.90}}, {{"regression", "prChannel250_3_190"}, {"upper", "prChannelUpper250_3_190"}, {"lower", "prChannelLower250_3_190"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 250.0}, {"degree", 3.0}, {"deviations", 1.95}}, {{"regression", "prChannel250_3_195"}, {"upper", "prChannelUpper250_3_195"}, {"lower", "prChannelLower250_3_195"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 250.0}, {"degree", 3.0}, {"deviations", 2.00}}, {{"regression", "prChannel250_3_200"}, {"upper", "prChannelUpper250_3_200"}, {"lower", "prChannelLower250_3_200"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 250.0}, {"degree", 3.0}, {"deviations", 2.05}}, {{"regression", "prChannel250_3_205"}, {"upper", "prChannelUpper250_3_205"}, {"lower", "prChannelLower250_3_205"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 250.0}, {"degree", 3.0}, {"deviations", 2.10}}, {{"regression", "prChannel250_3_210"}, {"upper", "prChannelUpper250_3_210"}, {"lower", "prChannelLower250_3_210"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 250.0}, {"degree", 3.0}, {"deviations", 2.15}}, {{"regression", "prChannel250_3_215"}, {"upper", "prChannelUpper250_3_215"}, {"lower", "prChannelLower250_3_215"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 250.0}, {"degree", 3.0}, {"deviations", 2.20}}, {{"regression", "prChannel250_3_220"}, {"upper", "prChannelUpper250_3_220"}, {"lower", "prChannelLower250_3_220"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 250.0}, {"degree", 3.0}, {"deviations", 2.25}}, {{"regression", "prChannel250_3_225"}, {"upper", "prChannelUpper250_3_225"}, {"lower", "prChannelLower250_3_225"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 250.0}, {"degree", 3.0}, {"deviations", 2.30}}, {{"regression", "prChannel250_3_230"}, {"upper", "prChannelUpper250_3_230"}, {"lower", "prChannelLower250_3_230"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 250.0}, {"degree", 3.0}, {"deviations", 2.35}}, {{"regression", "prChannel250_3_235"}, {"upper", "prChannelUpper250_3_235"}, {"lower", "prChannelLower250_3_235"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 250.0}, {"degree", 3.0}, {"deviations", 2.40}}, {{"regression", "prChannel250_3_240"}, {"upper", "prChannelUpper250_3_240"}, {"lower", "prChannelLower250_3_240"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 250.0}, {"degree", 4.0}, {"deviations", 1.90}}, {{"regression", "prChannel250_4_190"}, {"upper", "prChannelUpper250_4_190"}, {"lower", "prChannelLower250_4_190"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 250.0}, {"degree", 4.0}, {"deviations", 1.95}}, {{"regression", "prChannel250_4_195"}, {"upper", "prChannelUpper250_4_195"}, {"lower", "prChannelLower250_4_195"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 250.0}, {"degree", 4.0}, {"deviations", 2.00}}, {{"regression", "prChannel250_4_200"}, {"upper", "prChannelUpper250_4_200"}, {"lower", "prChannelLower250_4_200"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 250.0}, {"degree", 4.0}, {"deviations", 2.05}}, {{"regression", "prChannel250_4_205"}, {"upper", "prChannelUpper250_4_205"}, {"lower", "prChannelLower250_4_205"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 250.0}, {"degree", 4.0}, {"deviations", 2.10}}, {{"regression", "prChannel250_4_210"}, {"upper", "prChannelUpper250_4_210"}, {"lower", "prChannelLower250_4_210"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 250.0}, {"degree", 4.0}, {"deviations", 2.15}}, {{"regression", "prChannel250_4_215"}, {"upper", "prChannelUpper250_4_215"}, {"lower", "prChannelLower250_4_215"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 250.0}, {"degree", 4.0}, {"deviations", 2.20}}, {{"regression", "prChannel250_4_220"}, {"upper", "prChannelUpper250_4_220"}, {"lower", "prChannelLower250_4_220"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 250.0}, {"degree", 4.0}, {"deviations", 2.25}}, {{"regression", "prChannel250_4_225"}, {"upper", "prChannelUpper250_4_225"}, {"lower", "prChannelLower250_4_225"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 250.0}, {"degree", 4.0}, {"deviations", 2.30}}, {{"regression", "prChannel250_4_230"}, {"upper", "prChannelUpper250_4_230"}, {"lower", "prChannelLower250_4_230"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 250.0}, {"degree", 4.0}, {"deviations", 2.35}}, {{"regression", "prChannel250_4_235"}, {"upper", "prChannelUpper250_4_235"}, {"lower", "prChannelLower250_4_235"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 250.0}, {"degree", 4.0}, {"deviations", 2.40}}, {{"regression", "prChannel250_4_240"}, {"upper", "prChannelUpper250_4_240"}, {"lower", "prChannelLower250_4_240"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 250.0}, {"degree", 5.0}, {"deviations", 2.05}}, {{"regression", "prChannel250_5_205"}, {"upper", "prChannelUpper250_5_205"}, {"lower", "prChannelLower250_5_205"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 300.0}, {"degree", 2.0}, {"deviations", 1.90}}, {{"regression", "prChannel300_2_190"}, {"upper", "prChannelUpper300_2_190"}, {"lower", "prChannelLower300_2_190"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 300.0}, {"degree", 2.0}, {"deviations", 1.95}}, {{"regression", "prChannel300_2_195"}, {"upper", "prChannelUpper300_2_195"}, {"lower", "prChannelLower300_2_195"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 300.0}, {"degree", 2.0}, {"deviations", 2.00}}, {{"regression", "prChannel300_2_200"}, {"upper", "prChannelUpper300_2_200"}, {"lower", "prChannelLower300_2_200"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 300.0}, {"degree", 2.0}, {"deviations", 2.05}}, {{"regression", "prChannel300_2_205"}, {"upper", "prChannelUpper300_2_205"}, {"lower", "prChannelLower300_2_205"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 300.0}, {"degree", 2.0}, {"deviations", 2.10}}, {{"regression", "prChannel300_2_210"}, {"upper", "prChannelUpper300_2_210"}, {"lower", "prChannelLower300_2_210"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 300.0}, {"degree", 2.0}, {"deviations", 2.15}}, {{"regression", "prChannel300_2_215"}, {"upper", "prChannelUpper300_2_215"}, {"lower", "prChannelLower300_2_215"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 300.0}, {"degree", 2.0}, {"deviations", 2.20}}, {{"regression", "prChannel300_2_220"}, {"upper", "prChannelUpper300_2_220"}, {"lower", "prChannelLower300_2_220"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 300.0}, {"degree", 2.0}, {"deviations", 2.25}}, {{"regression", "prChannel300_2_225"}, {"upper", "prChannelUpper300_2_225"}, {"lower", "prChannelLower300_2_225"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 300.0}, {"degree", 2.0}, {"deviations", 2.30}}, {{"regression", "prChannel300_2_230"}, {"upper", "prChannelUpper300_2_230"}, {"lower", "prChannelLower300_2_230"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 300.0}, {"degree", 2.0}, {"deviations", 2.35}}, {{"regression", "prChannel300_2_235"}, {"upper", "prChannelUpper300_2_235"}, {"lower", "prChannelLower300_2_235"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 300.0}, {"degree", 2.0}, {"deviations", 2.40}}, {{"regression", "prChannel300_2_240"}, {"upper", "prChannelUpper300_2_240"}, {"lower", "prChannelLower300_2_240"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 300.0}, {"degree", 3.0}, {"deviations", 1.90}}, {{"regression", "prChannel300_3_190"}, {"upper", "prChannelUpper300_3_190"}, {"lower", "prChannelLower300_3_190"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 300.0}, {"degree", 3.0}, {"deviations", 1.95}}, {{"regression", "prChannel300_3_195"}, {"upper", "prChannelUpper300_3_195"}, {"lower", "prChannelLower300_3_195"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 300.0}, {"degree", 3.0}, {"deviations", 2.00}}, {{"regression", "prChannel300_3_200"}, {"upper", "prChannelUpper300_3_200"}, {"lower", "prChannelLower300_3_200"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 300.0}, {"degree", 3.0}, {"deviations", 2.05}}, {{"regression", "prChannel300_3_205"}, {"upper", "prChannelUpper300_3_205"}, {"lower", "prChannelLower300_3_205"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 300.0}, {"degree", 3.0}, {"deviations", 2.10}}, {{"regression", "prChannel300_3_210"}, {"upper", "prChannelUpper300_3_210"}, {"lower", "prChannelLower300_3_210"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 300.0}, {"degree", 3.0}, {"deviations", 2.15}}, {{"regression", "prChannel300_3_215"}, {"upper", "prChannelUpper300_3_215"}, {"lower", "prChannelLower300_3_215"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 300.0}, {"degree", 3.0}, {"deviations", 2.20}}, {{"regression", "prChannel300_3_220"}, {"upper", "prChannelUpper300_3_220"}, {"lower", "prChannelLower300_3_220"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 300.0}, {"degree", 3.0}, {"deviations", 2.25}}, {{"regression", "prChannel300_3_225"}, {"upper", "prChannelUpper300_3_225"}, {"lower", "prChannelLower300_3_225"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 300.0}, {"degree", 3.0}, {"deviations", 2.30}}, {{"regression", "prChannel300_3_230"}, {"upper", "prChannelUpper300_3_230"}, {"lower", "prChannelLower300_3_230"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 300.0}, {"degree", 3.0}, {"deviations", 2.35}}, {{"regression", "prChannel300_3_235"}, {"upper", "prChannelUpper300_3_235"}, {"lower", "prChannelLower300_3_235"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 300.0}, {"degree", 3.0}, {"deviations", 2.40}}, {{"regression", "prChannel300_3_240"}, {"upper", "prChannelUpper300_3_240"}, {"lower", "prChannelLower300_3_240"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 300.0}, {"degree", 4.0}, {"deviations", 1.90}}, {{"regression", "prChannel300_4_190"}, {"upper", "prChannelUpper300_4_190"}, {"lower", "prChannelLower300_4_190"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 300.0}, {"degree", 4.0}, {"deviations", 1.95}}, {{"regression", "prChannel300_4_195"}, {"upper", "prChannelUpper300_4_195"}, {"lower", "prChannelLower300_4_195"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 300.0}, {"degree", 4.0}, {"deviations", 2.00}}, {{"regression", "prChannel300_4_200"}, {"upper", "prChannelUpper300_4_200"}, {"lower", "prChannelLower300_4_200"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 300.0}, {"degree", 4.0}, {"deviations", 2.05}}, {{"regression", "prChannel300_4_205"}, {"upper", "prChannelUpper300_4_205"}, {"lower", "prChannelLower300_4_205"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 300.0}, {"degree", 4.0}, {"deviations", 2.10}}, {{"regression", "prChannel300_4_210"}, {"upper", "prChannelUpper300_4_210"}, {"lower", "prChannelLower300_4_210"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 300.0}, {"degree", 4.0}, {"deviations", 2.15}}, {{"regression", "prChannel300_4_215"}, {"upper", "prChannelUpper300_4_215"}, {"lower", "prChannelLower300_4_215"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 300.0}, {"degree", 4.0}, {"deviations", 2.20}}, {{"regression", "prChannel300_4_220"}, {"upper", "prChannelUpper300_4_220"}, {"lower", "prChannelLower300_4_220"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 300.0}, {"degree", 4.0}, {"deviations", 2.25}}, {{"regression", "prChannel300_4_225"}, {"upper", "prChannelUpper300_4_225"}, {"lower", "prChannelLower300_4_225"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 300.0}, {"degree", 4.0}, {"deviations", 2.30}}, {{"regression", "prChannel300_4_230"}, {"upper", "prChannelUpper300_4_230"}, {"lower", "prChannelLower300_4_230"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 300.0}, {"degree", 4.0}, {"deviations", 2.35}}, {{"regression", "prChannel300_4_235"}, {"upper", "prChannelUpper300_4_235"}, {"lower", "prChannelLower300_4_235"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 300.0}, {"degree", 4.0}, {"deviations", 2.40}}, {{"regression", "prChannel300_4_240"}, {"upper", "prChannelUpper300_4_240"}, {"lower", "prChannelLower300_4_240"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 300.0}, {"degree", 5.0}, {"deviations", 2.05}}, {{"regression", "prChannel300_5_205"}, {"upper", "prChannelUpper300_5_205"}, {"lower", "prChannelLower300_5_205"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 350.0}, {"degree", 2.0}, {"deviations", 1.90}}, {{"regression", "prChannel350_2_190"}, {"upper", "prChannelUpper350_2_190"}, {"lower", "prChannelLower350_2_190"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 350.0}, {"degree", 2.0}, {"deviations", 1.95}}, {{"regression", "prChannel350_2_195"}, {"upper", "prChannelUpper350_2_195"}, {"lower", "prChannelLower350_2_195"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 350.0}, {"degree", 2.0}, {"deviations", 2.00}}, {{"regression", "prChannel350_2_200"}, {"upper", "prChannelUpper350_2_200"}, {"lower", "prChannelLower350_2_200"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 350.0}, {"degree", 2.0}, {"deviations", 2.05}}, {{"regression", "prChannel350_2_205"}, {"upper", "prChannelUpper350_2_205"}, {"lower", "prChannelLower350_2_205"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 350.0}, {"degree", 2.0}, {"deviations", 2.10}}, {{"regression", "prChannel350_2_210"}, {"upper", "prChannelUpper350_2_210"}, {"lower", "prChannelLower350_2_210"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 350.0}, {"degree", 2.0}, {"deviations", 2.15}}, {{"regression", "prChannel350_2_215"}, {"upper", "prChannelUpper350_2_215"}, {"lower", "prChannelLower350_2_215"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 350.0}, {"degree", 2.0}, {"deviations", 2.20}}, {{"regression", "prChannel350_2_220"}, {"upper", "prChannelUpper350_2_220"}, {"lower", "prChannelLower350_2_220"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 350.0}, {"degree", 2.0}, {"deviations", 2.25}}, {{"regression", "prChannel350_2_225"}, {"upper", "prChannelUpper350_2_225"}, {"lower", "prChannelLower350_2_225"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 350.0}, {"degree", 2.0}, {"deviations", 2.30}}, {{"regression", "prChannel350_2_230"}, {"upper", "prChannelUpper350_2_230"}, {"lower", "prChannelLower350_2_230"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 350.0}, {"degree", 2.0}, {"deviations", 2.35}}, {{"regression", "prChannel350_2_235"}, {"upper", "prChannelUpper350_2_235"}, {"lower", "prChannelLower350_2_235"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 350.0}, {"degree", 2.0}, {"deviations", 2.40}}, {{"regression", "prChannel350_2_240"}, {"upper", "prChannelUpper350_2_240"}, {"lower", "prChannelLower350_2_240"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 350.0}, {"degree", 3.0}, {"deviations", 1.90}}, {{"regression", "prChannel350_3_190"}, {"upper", "prChannelUpper350_3_190"}, {"lower", "prChannelLower350_3_190"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 350.0}, {"degree", 3.0}, {"deviations", 1.95}}, {{"regression", "prChannel350_3_195"}, {"upper", "prChannelUpper350_3_195"}, {"lower", "prChannelLower350_3_195"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 350.0}, {"degree", 3.0}, {"deviations", 2.00}}, {{"regression", "prChannel350_3_200"}, {"upper", "prChannelUpper350_3_200"}, {"lower", "prChannelLower350_3_200"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 350.0}, {"degree", 3.0}, {"deviations", 2.05}}, {{"regression", "prChannel350_3_205"}, {"upper", "prChannelUpper350_3_205"}, {"lower", "prChannelLower350_3_205"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 350.0}, {"degree", 3.0}, {"deviations", 2.10}}, {{"regression", "prChannel350_3_210"}, {"upper", "prChannelUpper350_3_210"}, {"lower", "prChannelLower350_3_210"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 350.0}, {"degree", 3.0}, {"deviations", 2.15}}, {{"regression", "prChannel350_3_215"}, {"upper", "prChannelUpper350_3_215"}, {"lower", "prChannelLower350_3_215"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 350.0}, {"degree", 3.0}, {"deviations", 2.20}}, {{"regression", "prChannel350_3_220"}, {"upper", "prChannelUpper350_3_220"}, {"lower", "prChannelLower350_3_220"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 350.0}, {"degree", 3.0}, {"deviations", 2.25}}, {{"regression", "prChannel350_3_225"}, {"upper", "prChannelUpper350_3_225"}, {"lower", "prChannelLower350_3_225"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 350.0}, {"degree", 3.0}, {"deviations", 2.30}}, {{"regression", "prChannel350_3_230"}, {"upper", "prChannelUpper350_3_230"}, {"lower", "prChannelLower350_3_230"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 350.0}, {"degree", 3.0}, {"deviations", 2.35}}, {{"regression", "prChannel350_3_235"}, {"upper", "prChannelUpper350_3_235"}, {"lower", "prChannelLower350_3_235"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 350.0}, {"degree", 3.0}, {"deviations", 2.40}}, {{"regression", "prChannel350_3_240"}, {"upper", "prChannelUpper350_3_240"}, {"lower", "prChannelLower350_3_240"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 350.0}, {"degree", 4.0}, {"deviations", 1.90}}, {{"regression", "prChannel350_4_190"}, {"upper", "prChannelUpper350_4_190"}, {"lower", "prChannelLower350_4_190"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 350.0}, {"degree", 4.0}, {"deviations", 1.95}}, {{"regression", "prChannel350_4_195"}, {"upper", "prChannelUpper350_4_195"}, {"lower", "prChannelLower350_4_195"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 350.0}, {"degree", 4.0}, {"deviations", 2.00}}, {{"regression", "prChannel350_4_200"}, {"upper", "prChannelUpper350_4_200"}, {"lower", "prChannelLower350_4_200"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 350.0}, {"degree", 4.0}, {"deviations", 2.05}}, {{"regression", "prChannel350_4_205"}, {"upper", "prChannelUpper350_4_205"}, {"lower", "prChannelLower350_4_205"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 350.0}, {"degree", 4.0}, {"deviations", 2.10}}, {{"regression", "prChannel350_4_210"}, {"upper", "prChannelUpper350_4_210"}, {"lower", "prChannelLower350_4_210"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 350.0}, {"degree", 4.0}, {"deviations", 2.15}}, {{"regression", "prChannel350_4_215"}, {"upper", "prChannelUpper350_4_215"}, {"lower", "prChannelLower350_4_215"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 350.0}, {"degree", 4.0}, {"deviations", 2.20}}, {{"regression", "prChannel350_4_220"}, {"upper", "prChannelUpper350_4_220"}, {"lower", "prChannelLower350_4_220"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 350.0}, {"degree", 4.0}, {"deviations", 2.25}}, {{"regression", "prChannel350_4_225"}, {"upper", "prChannelUpper350_4_225"}, {"lower", "prChannelLower350_4_225"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 350.0}, {"degree", 4.0}, {"deviations", 2.30}}, {{"regression", "prChannel350_4_230"}, {"upper", "prChannelUpper350_4_230"}, {"lower", "prChannelLower350_4_230"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 350.0}, {"degree", 4.0}, {"deviations", 2.35}}, {{"regression", "prChannel350_4_235"}, {"upper", "prChannelUpper350_4_235"}, {"lower", "prChannelLower350_4_235"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 350.0}, {"degree", 4.0}, {"deviations", 2.40}}, {{"regression", "prChannel350_4_240"}, {"upper", "prChannelUpper350_4_240"}, {"lower", "prChannelLower350_4_240"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 350.0}, {"degree", 5.0}, {"deviations", 2.05}}, {{"regression", "prChannel350_5_205"}, {"upper", "prChannelUpper350_5_205"}, {"lower", "prChannelLower350_5_205"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 400.0}, {"degree", 2.0}, {"deviations", 1.90}}, {{"regression", "prChannel400_2_190"}, {"upper", "prChannelUpper400_2_190"}, {"lower", "prChannelLower400_2_190"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 400.0}, {"degree", 2.0}, {"deviations", 1.95}}, {{"regression", "prChannel400_2_195"}, {"upper", "prChannelUpper400_2_195"}, {"lower", "prChannelLower400_2_195"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 400.0}, {"degree", 2.0}, {"deviations", 2.00}}, {{"regression", "prChannel400_2_200"}, {"upper", "prChannelUpper400_2_200"}, {"lower", "prChannelLower400_2_200"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 400.0}, {"degree", 2.0}, {"deviations", 2.05}}, {{"regression", "prChannel400_2_205"}, {"upper", "prChannelUpper400_2_205"}, {"lower", "prChannelLower400_2_205"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 400.0}, {"degree", 2.0}, {"deviations", 2.10}}, {{"regression", "prChannel400_2_210"}, {"upper", "prChannelUpper400_2_210"}, {"lower", "prChannelLower400_2_210"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 400.0}, {"degree", 2.0}, {"deviations", 2.15}}, {{"regression", "prChannel400_2_215"}, {"upper", "prChannelUpper400_2_215"}, {"lower", "prChannelLower400_2_215"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 400.0}, {"degree", 2.0}, {"deviations", 2.20}}, {{"regression", "prChannel400_2_220"}, {"upper", "prChannelUpper400_2_220"}, {"lower", "prChannelLower400_2_220"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 400.0}, {"degree", 2.0}, {"deviations", 2.25}}, {{"regression", "prChannel400_2_225"}, {"upper", "prChannelUpper400_2_225"}, {"lower", "prChannelLower400_2_225"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 400.0}, {"degree", 2.0}, {"deviations", 2.30}}, {{"regression", "prChannel400_2_230"}, {"upper", "prChannelUpper400_2_230"}, {"lower", "prChannelLower400_2_230"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 400.0}, {"degree", 2.0}, {"deviations", 2.35}}, {{"regression", "prChannel400_2_235"}, {"upper", "prChannelUpper400_2_235"}, {"lower", "prChannelLower400_2_235"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 400.0}, {"degree", 2.0}, {"deviations", 2.40}}, {{"regression", "prChannel400_2_240"}, {"upper", "prChannelUpper400_2_240"}, {"lower", "prChannelLower400_2_240"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 400.0}, {"degree", 3.0}, {"deviations", 1.90}}, {{"regression", "prChannel400_3_190"}, {"upper", "prChannelUpper400_3_190"}, {"lower", "prChannelLower400_3_190"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 400.0}, {"degree", 3.0}, {"deviations", 1.95}}, {{"regression", "prChannel400_3_195"}, {"upper", "prChannelUpper400_3_195"}, {"lower", "prChannelLower400_3_195"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 400.0}, {"degree", 3.0}, {"deviations", 2.00}}, {{"regression", "prChannel400_3_200"}, {"upper", "prChannelUpper400_3_200"}, {"lower", "prChannelLower400_3_200"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 400.0}, {"degree", 3.0}, {"deviations", 2.05}}, {{"regression", "prChannel400_3_205"}, {"upper", "prChannelUpper400_3_205"}, {"lower", "prChannelLower400_3_205"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 400.0}, {"degree", 3.0}, {"deviations", 2.10}}, {{"regression", "prChannel400_3_210"}, {"upper", "prChannelUpper400_3_210"}, {"lower", "prChannelLower400_3_210"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 400.0}, {"degree", 3.0}, {"deviations", 2.15}}, {{"regression", "prChannel400_3_215"}, {"upper", "prChannelUpper400_3_215"}, {"lower", "prChannelLower400_3_215"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 400.0}, {"degree", 3.0}, {"deviations", 2.20}}, {{"regression", "prChannel400_3_220"}, {"upper", "prChannelUpper400_3_220"}, {"lower", "prChannelLower400_3_220"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 400.0}, {"degree", 3.0}, {"deviations", 2.25}}, {{"regression", "prChannel400_3_225"}, {"upper", "prChannelUpper400_3_225"}, {"lower", "prChannelLower400_3_225"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 400.0}, {"degree", 3.0}, {"deviations", 2.30}}, {{"regression", "prChannel400_3_230"}, {"upper", "prChannelUpper400_3_230"}, {"lower", "prChannelLower400_3_230"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 400.0}, {"degree", 3.0}, {"deviations", 2.35}}, {{"regression", "prChannel400_3_235"}, {"upper", "prChannelUpper400_3_235"}, {"lower", "prChannelLower400_3_235"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 400.0}, {"degree", 3.0}, {"deviations", 2.40}}, {{"regression", "prChannel400_3_240"}, {"upper", "prChannelUpper400_3_240"}, {"lower", "prChannelLower400_3_240"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 400.0}, {"degree", 4.0}, {"deviations", 1.90}}, {{"regression", "prChannel400_4_190"}, {"upper", "prChannelUpper400_4_190"}, {"lower", "prChannelLower400_4_190"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 400.0}, {"degree", 4.0}, {"deviations", 1.95}}, {{"regression", "prChannel400_4_195"}, {"upper", "prChannelUpper400_4_195"}, {"lower", "prChannelLower400_4_195"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 400.0}, {"degree", 4.0}, {"deviations", 2.00}}, {{"regression", "prChannel400_4_200"}, {"upper", "prChannelUpper400_4_200"}, {"lower", "prChannelLower400_4_200"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 400.0}, {"degree", 4.0}, {"deviations", 2.05}}, {{"regression", "prChannel400_4_205"}, {"upper", "prChannelUpper400_4_205"}, {"lower", "prChannelLower400_4_205"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 400.0}, {"degree", 4.0}, {"deviations", 2.10}}, {{"regression", "prChannel400_4_210"}, {"upper", "prChannelUpper400_4_210"}, {"lower", "prChannelLower400_4_210"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 400.0}, {"degree", 4.0}, {"deviations", 2.15}}, {{"regression", "prChannel400_4_215"}, {"upper", "prChannelUpper400_4_215"}, {"lower", "prChannelLower400_4_215"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 400.0}, {"degree", 4.0}, {"deviations", 2.20}}, {{"regression", "prChannel400_4_220"}, {"upper", "prChannelUpper400_4_220"}, {"lower", "prChannelLower400_4_220"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 400.0}, {"degree", 4.0}, {"deviations", 2.25}}, {{"regression", "prChannel400_4_225"}, {"upper", "prChannelUpper400_4_225"}, {"lower", "prChannelLower400_4_225"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 400.0}, {"degree", 4.0}, {"deviations", 2.30}}, {{"regression", "prChannel400_4_230"}, {"upper", "prChannelUpper400_4_230"}, {"lower", "prChannelLower400_4_230"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 400.0}, {"degree", 4.0}, {"deviations", 2.35}}, {{"regression", "prChannel400_4_235"}, {"upper", "prChannelUpper400_4_235"}, {"lower", "prChannelLower400_4_235"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 400.0}, {"degree", 4.0}, {"deviations", 2.40}}, {{"regression", "prChannel400_4_240"}, {"upper", "prChannelUpper400_4_240"}, {"lower", "prChannelLower400_4_240"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 400.0}, {"degree", 5.0}, {"deviations", 2.05}}, {{"regression", "prChannel400_5_205"}, {"upper", "prChannelUpper400_5_205"}, {"lower", "prChannelLower400_5_205"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 450.0}, {"degree", 2.0}, {"deviations", 1.90}}, {{"regression", "prChannel450_2_190"}, {"upper", "prChannelUpper450_2_190"}, {"lower", "prChannelLower450_2_190"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 450.0}, {"degree", 2.0}, {"deviations", 1.95}}, {{"regression", "prChannel450_2_195"}, {"upper", "prChannelUpper450_2_195"}, {"lower", "prChannelLower450_2_195"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 450.0}, {"degree", 2.0}, {"deviations", 2.00}}, {{"regression", "prChannel450_2_200"}, {"upper", "prChannelUpper450_2_200"}, {"lower", "prChannelLower450_2_200"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 450.0}, {"degree", 2.0}, {"deviations", 2.05}}, {{"regression", "prChannel450_2_205"}, {"upper", "prChannelUpper450_2_205"}, {"lower", "prChannelLower450_2_205"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 450.0}, {"degree", 2.0}, {"deviations", 2.10}}, {{"regression", "prChannel450_2_210"}, {"upper", "prChannelUpper450_2_210"}, {"lower", "prChannelLower450_2_210"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 450.0}, {"degree", 2.0}, {"deviations", 2.15}}, {{"regression", "prChannel450_2_215"}, {"upper", "prChannelUpper450_2_215"}, {"lower", "prChannelLower450_2_215"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 450.0}, {"degree", 2.0}, {"deviations", 2.20}}, {{"regression", "prChannel450_2_220"}, {"upper", "prChannelUpper450_2_220"}, {"lower", "prChannelLower450_2_220"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 450.0}, {"degree", 2.0}, {"deviations", 2.25}}, {{"regression", "prChannel450_2_225"}, {"upper", "prChannelUpper450_2_225"}, {"lower", "prChannelLower450_2_225"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 450.0}, {"degree", 2.0}, {"deviations", 2.30}}, {{"regression", "prChannel450_2_230"}, {"upper", "prChannelUpper450_2_230"}, {"lower", "prChannelLower450_2_230"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 450.0}, {"degree", 2.0}, {"deviations", 2.35}}, {{"regression", "prChannel450_2_235"}, {"upper", "prChannelUpper450_2_235"}, {"lower", "prChannelLower450_2_235"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 450.0}, {"degree", 2.0}, {"deviations", 2.40}}, {{"regression", "prChannel450_2_240"}, {"upper", "prChannelUpper450_2_240"}, {"lower", "prChannelLower450_2_240"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 450.0}, {"degree", 3.0}, {"deviations", 1.90}}, {{"regression", "prChannel450_3_190"}, {"upper", "prChannelUpper450_3_190"}, {"lower", "prChannelLower450_3_190"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 450.0}, {"degree", 3.0}, {"deviations", 1.95}}, {{"regression", "prChannel450_3_195"}, {"upper", "prChannelUpper450_3_195"}, {"lower", "prChannelLower450_3_195"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 450.0}, {"degree", 3.0}, {"deviations", 2.00}}, {{"regression", "prChannel450_3_200"}, {"upper", "prChannelUpper450_3_200"}, {"lower", "prChannelLower450_3_200"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 450.0}, {"degree", 3.0}, {"deviations", 2.05}}, {{"regression", "prChannel450_3_205"}, {"upper", "prChannelUpper450_3_205"}, {"lower", "prChannelLower450_3_205"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 450.0}, {"degree", 3.0}, {"deviations", 2.10}}, {{"regression", "prChannel450_3_210"}, {"upper", "prChannelUpper450_3_210"}, {"lower", "prChannelLower450_3_210"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 450.0}, {"degree", 3.0}, {"deviations", 2.15}}, {{"regression", "prChannel450_3_215"}, {"upper", "prChannelUpper450_3_215"}, {"lower", "prChannelLower450_3_215"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 450.0}, {"degree", 3.0}, {"deviations", 2.20}}, {{"regression", "prChannel450_3_220"}, {"upper", "prChannelUpper450_3_220"}, {"lower", "prChannelLower450_3_220"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 450.0}, {"degree", 3.0}, {"deviations", 2.25}}, {{"regression", "prChannel450_3_225"}, {"upper", "prChannelUpper450_3_225"}, {"lower", "prChannelLower450_3_225"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 450.0}, {"degree", 3.0}, {"deviations", 2.30}}, {{"regression", "prChannel450_3_230"}, {"upper", "prChannelUpper450_3_230"}, {"lower", "prChannelLower450_3_230"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 450.0}, {"degree", 3.0}, {"deviations", 2.35}}, {{"regression", "prChannel450_3_235"}, {"upper", "prChannelUpper450_3_235"}, {"lower", "prChannelLower450_3_235"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 450.0}, {"degree", 3.0}, {"deviations", 2.40}}, {{"regression", "prChannel450_3_240"}, {"upper", "prChannelUpper450_3_240"}, {"lower", "prChannelLower450_3_240"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 450.0}, {"degree", 4.0}, {"deviations", 1.90}}, {{"regression", "prChannel450_4_190"}, {"upper", "prChannelUpper450_4_190"}, {"lower", "prChannelLower450_4_190"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 450.0}, {"degree", 4.0}, {"deviations", 1.95}}, {{"regression", "prChannel450_4_195"}, {"upper", "prChannelUpper450_4_195"}, {"lower", "prChannelLower450_4_195"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 450.0}, {"degree", 4.0}, {"deviations", 2.00}}, {{"regression", "prChannel450_4_200"}, {"upper", "prChannelUpper450_4_200"}, {"lower", "prChannelLower450_4_200"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 450.0}, {"degree", 4.0}, {"deviations", 2.05}}, {{"regression", "prChannel450_4_205"}, {"upper", "prChannelUpper450_4_205"}, {"lower", "prChannelLower450_4_205"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 450.0}, {"degree", 4.0}, {"deviations", 2.10}}, {{"regression", "prChannel450_4_210"}, {"upper", "prChannelUpper450_4_210"}, {"lower", "prChannelLower450_4_210"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 450.0}, {"degree", 4.0}, {"deviations", 2.15}}, {{"regression", "prChannel450_4_215"}, {"upper", "prChannelUpper450_4_215"}, {"lower", "prChannelLower450_4_215"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 450.0}, {"degree", 4.0}, {"deviations", 2.20}}, {{"regression", "prChannel450_4_220"}, {"upper", "prChannelUpper450_4_220"}, {"lower", "prChannelLower450_4_220"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 450.0}, {"degree", 4.0}, {"deviations", 2.25}}, {{"regression", "prChannel450_4_225"}, {"upper", "prChannelUpper450_4_225"}, {"lower", "prChannelLower450_4_225"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 450.0}, {"degree", 4.0}, {"deviations", 2.30}}, {{"regression", "prChannel450_4_230"}, {"upper", "prChannelUpper450_4_230"}, {"lower", "prChannelLower450_4_230"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 450.0}, {"degree", 4.0}, {"deviations", 2.35}}, {{"regression", "prChannel450_4_235"}, {"upper", "prChannelUpper450_4_235"}, {"lower", "prChannelLower450_4_235"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 450.0}, {"degree", 4.0}, {"deviations", 2.40}}, {{"regression", "prChannel450_4_240"}, {"upper", "prChannelUpper450_4_240"}, {"lower", "prChannelLower450_4_240"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 450.0}, {"degree", 5.0}, {"deviations", 2.05}}, {{"regression", "prChannel450_5_205"}, {"upper", "prChannelUpper450_5_205"}, {"lower", "prChannelLower450_5_205"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 500.0}, {"degree", 2.0}, {"deviations", 1.90}}, {{"regression", "prChannel500_2_190"}, {"upper", "prChannelUpper500_2_190"}, {"lower", "prChannelLower500_2_190"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 500.0}, {"degree", 2.0}, {"deviations", 1.95}}, {{"regression", "prChannel500_2_195"}, {"upper", "prChannelUpper500_2_195"}, {"lower", "prChannelLower500_2_195"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 500.0}, {"degree", 2.0}, {"deviations", 2.00}}, {{"regression", "prChannel500_2_200"}, {"upper", "prChannelUpper500_2_200"}, {"lower", "prChannelLower500_2_200"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 500.0}, {"degree", 2.0}, {"deviations", 2.05}}, {{"regression", "prChannel500_2_205"}, {"upper", "prChannelUpper500_2_205"}, {"lower", "prChannelLower500_2_205"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 500.0}, {"degree", 2.0}, {"deviations", 2.10}}, {{"regression", "prChannel500_2_210"}, {"upper", "prChannelUpper500_2_210"}, {"lower", "prChannelLower500_2_210"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 500.0}, {"degree", 2.0}, {"deviations", 2.15}}, {{"regression", "prChannel500_2_215"}, {"upper", "prChannelUpper500_2_215"}, {"lower", "prChannelLower500_2_215"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 500.0}, {"degree", 2.0}, {"deviations", 2.20}}, {{"regression", "prChannel500_2_220"}, {"upper", "prChannelUpper500_2_220"}, {"lower", "prChannelLower500_2_220"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 500.0}, {"degree", 2.0}, {"deviations", 2.25}}, {{"regression", "prChannel500_2_225"}, {"upper", "prChannelUpper500_2_225"}, {"lower", "prChannelLower500_2_225"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 500.0}, {"degree", 2.0}, {"deviations", 2.30}}, {{"regression", "prChannel500_2_230"}, {"upper", "prChannelUpper500_2_230"}, {"lower", "prChannelLower500_2_230"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 500.0}, {"degree", 2.0}, {"deviations", 2.35}}, {{"regression", "prChannel500_2_235"}, {"upper", "prChannelUpper500_2_235"}, {"lower", "prChannelLower500_2_235"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 500.0}, {"degree", 2.0}, {"deviations", 2.40}}, {{"regression", "prChannel500_2_240"}, {"upper", "prChannelUpper500_2_240"}, {"lower", "prChannelLower500_2_240"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 500.0}, {"degree", 3.0}, {"deviations", 1.90}}, {{"regression", "prChannel500_3_190"}, {"upper", "prChannelUpper500_3_190"}, {"lower", "prChannelLower500_3_190"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 500.0}, {"degree", 3.0}, {"deviations", 1.95}}, {{"regression", "prChannel500_3_195"}, {"upper", "prChannelUpper500_3_195"}, {"lower", "prChannelLower500_3_195"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 500.0}, {"degree", 3.0}, {"deviations", 2.00}}, {{"regression", "prChannel500_3_200"}, {"upper", "prChannelUpper500_3_200"}, {"lower", "prChannelLower500_3_200"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 500.0}, {"degree", 3.0}, {"deviations", 2.05}}, {{"regression", "prChannel500_3_205"}, {"upper", "prChannelUpper500_3_205"}, {"lower", "prChannelLower500_3_205"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 500.0}, {"degree", 3.0}, {"deviations", 2.10}}, {{"regression", "prChannel500_3_210"}, {"upper", "prChannelUpper500_3_210"}, {"lower", "prChannelLower500_3_210"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 500.0}, {"degree", 3.0}, {"deviations", 2.15}}, {{"regression", "prChannel500_3_215"}, {"upper", "prChannelUpper500_3_215"}, {"lower", "prChannelLower500_3_215"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 500.0}, {"degree", 3.0}, {"deviations", 2.20}}, {{"regression", "prChannel500_3_220"}, {"upper", "prChannelUpper500_3_220"}, {"lower", "prChannelLower500_3_220"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 500.0}, {"degree", 3.0}, {"deviations", 2.25}}, {{"regression", "prChannel500_3_225"}, {"upper", "prChannelUpper500_3_225"}, {"lower", "prChannelLower500_3_225"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 500.0}, {"degree", 3.0}, {"deviations", 2.30}}, {{"regression", "prChannel500_3_230"}, {"upper", "prChannelUpper500_3_230"}, {"lower", "prChannelLower500_3_230"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 500.0}, {"degree", 3.0}, {"deviations", 2.35}}, {{"regression", "prChannel500_3_235"}, {"upper", "prChannelUpper500_3_235"}, {"lower", "prChannelLower500_3_235"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 500.0}, {"degree", 3.0}, {"deviations", 2.40}}, {{"regression", "prChannel500_3_240"}, {"upper", "prChannelUpper500_3_240"}, {"lower", "prChannelLower500_3_240"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 500.0}, {"degree", 4.0}, {"deviations", 1.90}}, {{"regression", "prChannel500_4_190"}, {"upper", "prChannelUpper500_4_190"}, {"lower", "prChannelLower500_4_190"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 500.0}, {"degree", 4.0}, {"deviations", 1.95}}, {{"regression", "prChannel500_4_195"}, {"upper", "prChannelUpper500_4_195"}, {"lower", "prChannelLower500_4_195"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 500.0}, {"degree", 4.0}, {"deviations", 2.00}}, {{"regression", "prChannel500_4_200"}, {"upper", "prChannelUpper500_4_200"}, {"lower", "prChannelLower500_4_200"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 500.0}, {"degree", 4.0}, {"deviations", 2.05}}, {{"regression", "prChannel500_4_205"}, {"upper", "prChannelUpper500_4_205"}, {"lower", "prChannelLower500_4_205"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 500.0}, {"degree", 4.0}, {"deviations", 2.10}}, {{"regression", "prChannel500_4_210"}, {"upper", "prChannelUpper500_4_210"}, {"lower", "prChannelLower500_4_210"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 500.0}, {"degree", 4.0}, {"deviations", 2.15}}, {{"regression", "prChannel500_4_215"}, {"upper", "prChannelUpper500_4_215"}, {"lower", "prChannelLower500_4_215"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 500.0}, {"degree", 4.0}, {"deviations", 2.20}}, {{"regression", "prChannel500_4_220"}, {"upper", "prChannelUpper500_4_220"}, {"lower", "prChannelLower500_4_220"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 500.0}, {"degree", 4.0}, {"deviations", 2.25}}, {{"regression", "prChannel500_4_225"}, {"upper", "prChannelUpper500_4_225"}, {"lower", "prChannelLower500_4_225"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 500.0}, {"degree", 4.0}, {"deviations", 2.30}}, {{"regression", "prChannel500_4_230"}, {"upper", "prChannelUpper500_4_230"}, {"lower", "prChannelLower500_4_230"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 500.0}, {"degree", 4.0}, {"deviations", 2.35}}, {{"regression", "prChannel500_4_235"}, {"upper", "prChannelUpper500_4_235"}, {"lower", "prChannelLower500_4_235"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 500.0}, {"degree", 4.0}, {"deviations", 2.40}}, {{"regression", "prChannel500_4_240"}, {"upper", "prChannelUpper500_4_240"}, {"lower", "prChannelLower500_4_240"}}));
    this->studies.push_back(new PolynomialRegressionChannelStudy({{"length", 500.0}, {"degree", 5.0}, {"deviations", 2.05}}, {{"regression", "prChannel500_5_205"}, {"upper", "prChannelUpper500_5_205"}, {"lower", "prChannelLower500_5_205"}}));

    return this->studies;
}

std::map<std::string, ConfigurationOption> ReversalsOptimizer::getConfigurationOptions() {
    std::map<std::string, ConfigurationOption> configurationOptions;

    // SMA 13
    ConfigurationOption sma13Options;
    sma13Options.push_back({{}});
    sma13Options.push_back({{"sma13", "sma13"}});
    configurationOptions["sma13"] = sma13Options;

    // EMA 50
    ConfigurationOption ema50Options;
    ema50Options.push_back({{}});
    ema50Options.push_back({{"ema50", "ema50"}});
    configurationOptions["ema50"] = ema50Options;

    // EMA 100
    ConfigurationOption ema100Options;
    ema100Options.push_back({{}});
    ema100Options.push_back({{"ema100", "ema100"}});
    configurationOptions["ema100"] = ema100Options;

    // EMA 200
    ConfigurationOption ema200Options;
    ema200Options.push_back({{}});
    ema200Options.push_back({{"ema200", "ema200"}});
    configurationOptions["ema200"] = ema200Options;

    // // EMA 250
    // ConfigurationOption ema250Options;
    // ema250Options.push_back({{}});
    // ema250Options.push_back({{"ema250", "ema250"}});
    // configurationOptions["ema250"] = ema250Options;

    // // EMA 300
    // ConfigurationOption ema300Options;
    // ema300Options.push_back({{}});
    // ema300Options.push_back({{"ema300", "ema300"}});
    // configurationOptions["ema300"] = ema300Options;

    // // EMA 350
    // ConfigurationOption ema350Options;
    // ema350Options.push_back({{}});
    // ema350Options.push_back({{"ema350", "ema350"}});
    // configurationOptions["ema350"] = ema350Options;

    // // EMA 400
    // ConfigurationOption ema400Options;
    // ema400Options.push_back({{}});
    // ema400Options.push_back({{"ema400", "ema400"}});
    // configurationOptions["ema400"] = ema400Options;

    // // EMA 450
    // ConfigurationOption ema450Options;
    // ema450Options.push_back({{}});
    // ema450Options.push_back({{"ema450", "ema450"}});
    // configurationOptions["ema450"] = ema450Options;

    // // EMA 500
    // ConfigurationOption ema500Options;
    // ema500Options.push_back({{}});
    // ema500Options.push_back({{"ema500", "ema500"}});
    // configurationOptions["ema500"] = ema500Options;

    // RSI
    ConfigurationOption rsiOptions;
    rsiOptions.push_back({{}});
    rsiOptions.push_back({{"rsi", "rsi2"}, {"rsiOverbought", 95.0}, {"rsiOversold", 5.0}});
    rsiOptions.push_back({{"rsi", "rsi5"}, {"rsiOverbought", 80.0}, {"rsiOversold", 20.0}});
    rsiOptions.push_back({{"rsi", "rsi7"}, {"rsiOverbought", 77.0}, {"rsiOversold", 23.0}});
    rsiOptions.push_back({{"rsi", "rsi7"}, {"rsiOverbought", 80.0}, {"rsiOversold", 20.0}});
    rsiOptions.push_back({{"rsi", "rsi9"}, {"rsiOverbought", 70.0}, {"rsiOversold", 30.0}});
    rsiOptions.push_back({{"rsi", "rsi9"}, {"rsiOverbought", 77.0}, {"rsiOversold", 23.0}});
    rsiOptions.push_back({{"rsi", "rsi14"}, {"rsiOverbought", 70.0}, {"rsiOversold", 30.0}});
    configurationOptions["rsi"] = rsiOptions;

    // Stochastic Oscillator
    ConfigurationOption stochasticOptions;
    stochasticOptions.push_back({{}});
    stochasticOptions.push_back({{"stochasticK", "stochastic5K"}, {"stochasticD", "stochastic5D"}, {"stochasticOverbought", 77.0}, {"stochasticOversold", 23.0}});
    stochasticOptions.push_back({{"stochasticK", "stochastic5K"}, {"stochasticD", "stochastic5D"}, {"stochasticOverbought", 80.0}, {"stochasticOversold", 20.0}});
    stochasticOptions.push_back({{"stochasticK", "stochastic5K"}, {"stochasticD", "stochastic5D"}, {"stochasticOverbought", 95.0}, {"stochasticOversold", 5.0}});
    stochasticOptions.push_back({{"stochasticK", "stochastic10K"}, {"stochasticD", "stochastic10D"}, {"stochasticOverbought", 77.0}, {"stochasticOversold", 23.0}});
    stochasticOptions.push_back({{"stochasticK", "stochastic10K"}, {"stochasticD", "stochastic10D"}, {"stochasticOverbought", 80.0}, {"stochasticOversold", 20.0}});
    stochasticOptions.push_back({{"stochasticK", "stochastic14K"}, {"stochasticD", "stochastic14D"}, {"stochasticOverbought", 70.0}, {"stochasticOversold", 30.0}});
    stochasticOptions.push_back({{"stochasticK", "stochastic14K"}, {"stochasticD", "stochastic14D"}, {"stochasticOverbought", 77.0}, {"stochasticOversold", 23.0}});
    stochasticOptions.push_back({{"stochasticK", "stochastic21K"}, {"stochasticD", "stochastic21D"}, {"stochasticOverbought", 70.0}, {"stochasticOversold", 30.0}});
    stochasticOptions.push_back({{"stochasticK", "stochastic21K"}, {"stochasticD", "stochastic21D"}, {"stochasticOverbought", 77.0}, {"stochasticOversold", 23.0}});
    configurationOptions["stochastic"] = stochasticOptions;

    // Polynomial Regression Channel
    ConfigurationOption prChannelOptions;
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper100_2_190"}, {"prChannelLower", "prChannelLower100_2_190"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper100_2_195"}, {"prChannelLower", "prChannelLower100_2_195"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper100_2_200"}, {"prChannelLower", "prChannelLower100_2_200"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper100_2_205"}, {"prChannelLower", "prChannelLower100_2_205"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper100_2_210"}, {"prChannelLower", "prChannelLower100_2_210"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper100_2_215"}, {"prChannelLower", "prChannelLower100_2_215"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper100_2_220"}, {"prChannelLower", "prChannelLower100_2_220"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper100_2_225"}, {"prChannelLower", "prChannelLower100_2_225"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper100_2_230"}, {"prChannelLower", "prChannelLower100_2_230"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper100_2_235"}, {"prChannelLower", "prChannelLower100_2_235"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper100_2_240"}, {"prChannelLower", "prChannelLower100_2_240"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper100_3_190"}, {"prChannelLower", "prChannelLower100_3_190"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper100_3_195"}, {"prChannelLower", "prChannelLower100_3_195"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper100_3_200"}, {"prChannelLower", "prChannelLower100_3_200"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper100_3_205"}, {"prChannelLower", "prChannelLower100_3_205"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper100_3_210"}, {"prChannelLower", "prChannelLower100_3_210"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper100_3_215"}, {"prChannelLower", "prChannelLower100_3_215"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper100_3_220"}, {"prChannelLower", "prChannelLower100_3_220"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper100_3_225"}, {"prChannelLower", "prChannelLower100_3_225"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper100_3_230"}, {"prChannelLower", "prChannelLower100_3_230"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper100_3_235"}, {"prChannelLower", "prChannelLower100_3_235"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper100_3_240"}, {"prChannelLower", "prChannelLower100_3_240"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper100_4_190"}, {"prChannelLower", "prChannelLower100_4_190"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper100_4_195"}, {"prChannelLower", "prChannelLower100_4_195"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper100_4_200"}, {"prChannelLower", "prChannelLower100_4_200"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper100_4_205"}, {"prChannelLower", "prChannelLower100_4_205"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper100_4_210"}, {"prChannelLower", "prChannelLower100_4_210"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper100_4_215"}, {"prChannelLower", "prChannelLower100_4_215"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper100_4_220"}, {"prChannelLower", "prChannelLower100_4_220"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper100_4_225"}, {"prChannelLower", "prChannelLower100_4_225"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper100_4_230"}, {"prChannelLower", "prChannelLower100_4_230"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper100_4_235"}, {"prChannelLower", "prChannelLower100_4_235"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper100_4_240"}, {"prChannelLower", "prChannelLower100_4_240"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper100_5_205"}, {"prChannelLower", "prChannelLower100_5_205"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper200_2_190"}, {"prChannelLower", "prChannelLower200_2_190"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper200_2_195"}, {"prChannelLower", "prChannelLower200_2_195"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper200_2_200"}, {"prChannelLower", "prChannelLower200_2_200"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper200_2_205"}, {"prChannelLower", "prChannelLower200_2_205"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper200_2_210"}, {"prChannelLower", "prChannelLower200_2_210"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper200_2_215"}, {"prChannelLower", "prChannelLower200_2_215"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper200_2_220"}, {"prChannelLower", "prChannelLower200_2_220"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper200_2_225"}, {"prChannelLower", "prChannelLower200_2_225"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper200_2_230"}, {"prChannelLower", "prChannelLower200_2_230"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper200_2_235"}, {"prChannelLower", "prChannelLower200_2_235"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper200_2_240"}, {"prChannelLower", "prChannelLower200_2_240"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper200_3_190"}, {"prChannelLower", "prChannelLower200_3_190"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper200_3_195"}, {"prChannelLower", "prChannelLower200_3_195"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper200_3_200"}, {"prChannelLower", "prChannelLower200_3_200"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper200_3_205"}, {"prChannelLower", "prChannelLower200_3_205"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper200_3_210"}, {"prChannelLower", "prChannelLower200_3_210"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper200_3_215"}, {"prChannelLower", "prChannelLower200_3_215"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper200_3_220"}, {"prChannelLower", "prChannelLower200_3_220"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper200_3_225"}, {"prChannelLower", "prChannelLower200_3_225"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper200_3_230"}, {"prChannelLower", "prChannelLower200_3_230"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper200_3_235"}, {"prChannelLower", "prChannelLower200_3_235"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper200_3_240"}, {"prChannelLower", "prChannelLower200_3_240"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper200_4_190"}, {"prChannelLower", "prChannelLower200_4_190"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper200_4_195"}, {"prChannelLower", "prChannelLower200_4_195"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper200_4_200"}, {"prChannelLower", "prChannelLower200_4_200"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper200_4_205"}, {"prChannelLower", "prChannelLower200_4_205"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper200_4_210"}, {"prChannelLower", "prChannelLower200_4_210"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper200_4_215"}, {"prChannelLower", "prChannelLower200_4_215"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper200_4_220"}, {"prChannelLower", "prChannelLower200_4_220"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper200_4_225"}, {"prChannelLower", "prChannelLower200_4_225"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper200_4_230"}, {"prChannelLower", "prChannelLower200_4_230"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper200_4_235"}, {"prChannelLower", "prChannelLower200_4_235"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper200_4_240"}, {"prChannelLower", "prChannelLower200_4_240"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper200_5_205"}, {"prChannelLower", "prChannelLower200_5_205"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper250_2_190"}, {"prChannelLower", "prChannelLower250_2_190"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper250_2_195"}, {"prChannelLower", "prChannelLower250_2_195"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper250_2_200"}, {"prChannelLower", "prChannelLower250_2_200"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper250_2_205"}, {"prChannelLower", "prChannelLower250_2_205"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper250_2_210"}, {"prChannelLower", "prChannelLower250_2_210"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper250_2_215"}, {"prChannelLower", "prChannelLower250_2_215"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper250_2_220"}, {"prChannelLower", "prChannelLower250_2_220"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper250_2_225"}, {"prChannelLower", "prChannelLower250_2_225"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper250_2_230"}, {"prChannelLower", "prChannelLower250_2_230"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper250_2_235"}, {"prChannelLower", "prChannelLower250_2_235"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper250_2_240"}, {"prChannelLower", "prChannelLower250_2_240"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper250_3_190"}, {"prChannelLower", "prChannelLower250_3_190"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper250_3_195"}, {"prChannelLower", "prChannelLower250_3_195"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper250_3_200"}, {"prChannelLower", "prChannelLower250_3_200"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper250_3_205"}, {"prChannelLower", "prChannelLower250_3_205"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper250_3_210"}, {"prChannelLower", "prChannelLower250_3_210"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper250_3_215"}, {"prChannelLower", "prChannelLower250_3_215"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper250_3_220"}, {"prChannelLower", "prChannelLower250_3_220"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper250_3_225"}, {"prChannelLower", "prChannelLower250_3_225"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper250_3_230"}, {"prChannelLower", "prChannelLower250_3_230"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper250_3_235"}, {"prChannelLower", "prChannelLower250_3_235"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper250_3_240"}, {"prChannelLower", "prChannelLower250_3_240"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper250_4_190"}, {"prChannelLower", "prChannelLower250_4_190"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper250_4_195"}, {"prChannelLower", "prChannelLower250_4_195"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper250_4_200"}, {"prChannelLower", "prChannelLower250_4_200"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper250_4_205"}, {"prChannelLower", "prChannelLower250_4_205"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper250_4_210"}, {"prChannelLower", "prChannelLower250_4_210"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper250_4_215"}, {"prChannelLower", "prChannelLower250_4_215"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper250_4_220"}, {"prChannelLower", "prChannelLower250_4_220"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper250_4_225"}, {"prChannelLower", "prChannelLower250_4_225"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper250_4_230"}, {"prChannelLower", "prChannelLower250_4_230"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper250_4_235"}, {"prChannelLower", "prChannelLower250_4_235"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper250_4_240"}, {"prChannelLower", "prChannelLower250_4_240"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper250_5_205"}, {"prChannelLower", "prChannelLower250_5_205"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper300_2_190"}, {"prChannelLower", "prChannelLower300_2_190"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper300_2_195"}, {"prChannelLower", "prChannelLower300_2_195"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper300_2_200"}, {"prChannelLower", "prChannelLower300_2_200"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper300_2_205"}, {"prChannelLower", "prChannelLower300_2_205"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper300_2_210"}, {"prChannelLower", "prChannelLower300_2_210"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper300_2_215"}, {"prChannelLower", "prChannelLower300_2_215"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper300_2_220"}, {"prChannelLower", "prChannelLower300_2_220"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper300_2_225"}, {"prChannelLower", "prChannelLower300_2_225"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper300_2_230"}, {"prChannelLower", "prChannelLower300_2_230"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper300_2_235"}, {"prChannelLower", "prChannelLower300_2_235"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper300_2_240"}, {"prChannelLower", "prChannelLower300_2_240"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper300_3_190"}, {"prChannelLower", "prChannelLower300_3_190"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper300_3_195"}, {"prChannelLower", "prChannelLower300_3_195"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper300_3_200"}, {"prChannelLower", "prChannelLower300_3_200"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper300_3_205"}, {"prChannelLower", "prChannelLower300_3_205"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper300_3_210"}, {"prChannelLower", "prChannelLower300_3_210"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper300_3_215"}, {"prChannelLower", "prChannelLower300_3_215"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper300_3_220"}, {"prChannelLower", "prChannelLower300_3_220"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper300_3_225"}, {"prChannelLower", "prChannelLower300_3_225"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper300_3_230"}, {"prChannelLower", "prChannelLower300_3_230"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper300_3_235"}, {"prChannelLower", "prChannelLower300_3_235"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper300_3_240"}, {"prChannelLower", "prChannelLower300_3_240"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper300_4_190"}, {"prChannelLower", "prChannelLower300_4_190"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper300_4_195"}, {"prChannelLower", "prChannelLower300_4_195"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper300_4_200"}, {"prChannelLower", "prChannelLower300_4_200"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper300_4_205"}, {"prChannelLower", "prChannelLower300_4_205"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper300_4_210"}, {"prChannelLower", "prChannelLower300_4_210"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper300_4_215"}, {"prChannelLower", "prChannelLower300_4_215"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper300_4_220"}, {"prChannelLower", "prChannelLower300_4_220"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper300_4_225"}, {"prChannelLower", "prChannelLower300_4_225"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper300_4_230"}, {"prChannelLower", "prChannelLower300_4_230"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper300_4_235"}, {"prChannelLower", "prChannelLower300_4_235"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper300_4_240"}, {"prChannelLower", "prChannelLower300_4_240"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper300_5_205"}, {"prChannelLower", "prChannelLower300_5_205"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper350_2_190"}, {"prChannelLower", "prChannelLower350_2_190"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper350_2_195"}, {"prChannelLower", "prChannelLower350_2_195"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper350_2_200"}, {"prChannelLower", "prChannelLower350_2_200"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper350_2_205"}, {"prChannelLower", "prChannelLower350_2_205"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper350_2_210"}, {"prChannelLower", "prChannelLower350_2_210"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper350_2_215"}, {"prChannelLower", "prChannelLower350_2_215"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper350_2_220"}, {"prChannelLower", "prChannelLower350_2_220"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper350_2_225"}, {"prChannelLower", "prChannelLower350_2_225"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper350_2_230"}, {"prChannelLower", "prChannelLower350_2_230"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper350_2_235"}, {"prChannelLower", "prChannelLower350_2_235"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper350_2_240"}, {"prChannelLower", "prChannelLower350_2_240"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper350_3_190"}, {"prChannelLower", "prChannelLower350_3_190"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper350_3_195"}, {"prChannelLower", "prChannelLower350_3_195"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper350_3_200"}, {"prChannelLower", "prChannelLower350_3_200"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper350_3_205"}, {"prChannelLower", "prChannelLower350_3_205"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper350_3_210"}, {"prChannelLower", "prChannelLower350_3_210"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper350_3_215"}, {"prChannelLower", "prChannelLower350_3_215"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper350_3_220"}, {"prChannelLower", "prChannelLower350_3_220"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper350_3_225"}, {"prChannelLower", "prChannelLower350_3_225"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper350_3_230"}, {"prChannelLower", "prChannelLower350_3_230"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper350_3_235"}, {"prChannelLower", "prChannelLower350_3_235"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper350_3_240"}, {"prChannelLower", "prChannelLower350_3_240"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper350_4_190"}, {"prChannelLower", "prChannelLower350_4_190"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper350_4_195"}, {"prChannelLower", "prChannelLower350_4_195"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper350_4_200"}, {"prChannelLower", "prChannelLower350_4_200"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper350_4_205"}, {"prChannelLower", "prChannelLower350_4_205"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper350_4_210"}, {"prChannelLower", "prChannelLower350_4_210"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper350_4_215"}, {"prChannelLower", "prChannelLower350_4_215"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper350_4_220"}, {"prChannelLower", "prChannelLower350_4_220"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper350_4_225"}, {"prChannelLower", "prChannelLower350_4_225"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper350_4_230"}, {"prChannelLower", "prChannelLower350_4_230"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper350_4_235"}, {"prChannelLower", "prChannelLower350_4_235"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper350_4_240"}, {"prChannelLower", "prChannelLower350_4_240"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper350_5_205"}, {"prChannelLower", "prChannelLower350_5_205"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper400_2_190"}, {"prChannelLower", "prChannelLower400_2_190"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper400_2_195"}, {"prChannelLower", "prChannelLower400_2_195"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper400_2_200"}, {"prChannelLower", "prChannelLower400_2_200"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper400_2_205"}, {"prChannelLower", "prChannelLower400_2_205"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper400_2_210"}, {"prChannelLower", "prChannelLower400_2_210"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper400_2_215"}, {"prChannelLower", "prChannelLower400_2_215"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper400_2_220"}, {"prChannelLower", "prChannelLower400_2_220"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper400_2_225"}, {"prChannelLower", "prChannelLower400_2_225"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper400_2_230"}, {"prChannelLower", "prChannelLower400_2_230"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper400_2_235"}, {"prChannelLower", "prChannelLower400_2_235"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper400_2_240"}, {"prChannelLower", "prChannelLower400_2_240"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper400_3_190"}, {"prChannelLower", "prChannelLower400_3_190"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper400_3_195"}, {"prChannelLower", "prChannelLower400_3_195"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper400_3_200"}, {"prChannelLower", "prChannelLower400_3_200"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper400_3_205"}, {"prChannelLower", "prChannelLower400_3_205"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper400_3_210"}, {"prChannelLower", "prChannelLower400_3_210"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper400_3_215"}, {"prChannelLower", "prChannelLower400_3_215"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper400_3_220"}, {"prChannelLower", "prChannelLower400_3_220"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper400_3_225"}, {"prChannelLower", "prChannelLower400_3_225"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper400_3_230"}, {"prChannelLower", "prChannelLower400_3_230"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper400_3_235"}, {"prChannelLower", "prChannelLower400_3_235"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper400_3_240"}, {"prChannelLower", "prChannelLower400_3_240"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper400_4_190"}, {"prChannelLower", "prChannelLower400_4_190"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper400_4_195"}, {"prChannelLower", "prChannelLower400_4_195"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper400_4_200"}, {"prChannelLower", "prChannelLower400_4_200"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper400_4_205"}, {"prChannelLower", "prChannelLower400_4_205"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper400_4_210"}, {"prChannelLower", "prChannelLower400_4_210"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper400_4_215"}, {"prChannelLower", "prChannelLower400_4_215"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper400_4_220"}, {"prChannelLower", "prChannelLower400_4_220"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper400_4_225"}, {"prChannelLower", "prChannelLower400_4_225"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper400_4_230"}, {"prChannelLower", "prChannelLower400_4_230"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper400_4_235"}, {"prChannelLower", "prChannelLower400_4_235"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper400_4_240"}, {"prChannelLower", "prChannelLower400_4_240"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper400_5_205"}, {"prChannelLower", "prChannelLower400_5_205"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper450_2_190"}, {"prChannelLower", "prChannelLower450_2_190"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper450_2_195"}, {"prChannelLower", "prChannelLower450_2_195"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper450_2_200"}, {"prChannelLower", "prChannelLower450_2_200"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper450_2_205"}, {"prChannelLower", "prChannelLower450_2_205"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper450_2_210"}, {"prChannelLower", "prChannelLower450_2_210"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper450_2_215"}, {"prChannelLower", "prChannelLower450_2_215"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper450_2_220"}, {"prChannelLower", "prChannelLower450_2_220"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper450_2_225"}, {"prChannelLower", "prChannelLower450_2_225"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper450_2_230"}, {"prChannelLower", "prChannelLower450_2_230"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper450_2_235"}, {"prChannelLower", "prChannelLower450_2_235"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper450_2_240"}, {"prChannelLower", "prChannelLower450_2_240"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper450_3_190"}, {"prChannelLower", "prChannelLower450_3_190"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper450_3_195"}, {"prChannelLower", "prChannelLower450_3_195"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper450_3_200"}, {"prChannelLower", "prChannelLower450_3_200"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper450_3_205"}, {"prChannelLower", "prChannelLower450_3_205"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper450_3_210"}, {"prChannelLower", "prChannelLower450_3_210"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper450_3_215"}, {"prChannelLower", "prChannelLower450_3_215"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper450_3_220"}, {"prChannelLower", "prChannelLower450_3_220"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper450_3_225"}, {"prChannelLower", "prChannelLower450_3_225"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper450_3_230"}, {"prChannelLower", "prChannelLower450_3_230"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper450_3_235"}, {"prChannelLower", "prChannelLower450_3_235"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper450_3_240"}, {"prChannelLower", "prChannelLower450_3_240"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper450_4_190"}, {"prChannelLower", "prChannelLower450_4_190"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper450_4_195"}, {"prChannelLower", "prChannelLower450_4_195"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper450_4_200"}, {"prChannelLower", "prChannelLower450_4_200"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper450_4_205"}, {"prChannelLower", "prChannelLower450_4_205"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper450_4_210"}, {"prChannelLower", "prChannelLower450_4_210"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper450_4_215"}, {"prChannelLower", "prChannelLower450_4_215"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper450_4_220"}, {"prChannelLower", "prChannelLower450_4_220"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper450_4_225"}, {"prChannelLower", "prChannelLower450_4_225"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper450_4_230"}, {"prChannelLower", "prChannelLower450_4_230"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper450_4_235"}, {"prChannelLower", "prChannelLower450_4_235"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper450_4_240"}, {"prChannelLower", "prChannelLower450_4_240"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper450_5_205"}, {"prChannelLower", "prChannelLower450_5_205"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper500_2_190"}, {"prChannelLower", "prChannelLower500_2_190"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper500_2_195"}, {"prChannelLower", "prChannelLower500_2_195"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper500_2_200"}, {"prChannelLower", "prChannelLower500_2_200"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper500_2_205"}, {"prChannelLower", "prChannelLower500_2_205"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper500_2_210"}, {"prChannelLower", "prChannelLower500_2_210"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper500_2_215"}, {"prChannelLower", "prChannelLower500_2_215"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper500_2_220"}, {"prChannelLower", "prChannelLower500_2_220"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper500_2_225"}, {"prChannelLower", "prChannelLower500_2_225"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper500_2_230"}, {"prChannelLower", "prChannelLower500_2_230"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper500_2_235"}, {"prChannelLower", "prChannelLower500_2_235"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper500_2_240"}, {"prChannelLower", "prChannelLower500_2_240"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper500_3_190"}, {"prChannelLower", "prChannelLower500_3_190"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper500_3_195"}, {"prChannelLower", "prChannelLower500_3_195"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper500_3_200"}, {"prChannelLower", "prChannelLower500_3_200"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper500_3_205"}, {"prChannelLower", "prChannelLower500_3_205"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper500_3_210"}, {"prChannelLower", "prChannelLower500_3_210"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper500_3_215"}, {"prChannelLower", "prChannelLower500_3_215"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper500_3_220"}, {"prChannelLower", "prChannelLower500_3_220"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper500_3_225"}, {"prChannelLower", "prChannelLower500_3_225"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper500_3_230"}, {"prChannelLower", "prChannelLower500_3_230"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper500_3_235"}, {"prChannelLower", "prChannelLower500_3_235"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper500_3_240"}, {"prChannelLower", "prChannelLower500_3_240"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper500_4_190"}, {"prChannelLower", "prChannelLower500_4_190"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper500_4_195"}, {"prChannelLower", "prChannelLower500_4_195"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper500_4_200"}, {"prChannelLower", "prChannelLower500_4_200"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper500_4_205"}, {"prChannelLower", "prChannelLower500_4_205"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper500_4_210"}, {"prChannelLower", "prChannelLower500_4_210"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper500_4_215"}, {"prChannelLower", "prChannelLower500_4_215"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper500_4_220"}, {"prChannelLower", "prChannelLower500_4_220"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper500_4_225"}, {"prChannelLower", "prChannelLower500_4_225"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper500_4_230"}, {"prChannelLower", "prChannelLower500_4_230"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper500_4_235"}, {"prChannelLower", "prChannelLower500_4_235"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper500_4_240"}, {"prChannelLower", "prChannelLower500_4_240"}});
    prChannelOptions.push_back({{"prChannelUpper", "prChannelUpper500_5_205"}, {"prChannelLower", "prChannelLower500_5_205"}});
    configurationOptions["prChannel"] = prChannelOptions;

    return configurationOptions;
}

std::vector<Configuration*> ReversalsOptimizer::buildConfigurations() {
    printf("Building configurations...");

    std::map<std::string, ConfigurationOption> options = getConfigurationOptions();
    std::map<std::string, int> *tempDataIndexMap = getDataIndexMap();
    std::vector<MapConfiguration> *mapConfigurations = buildMapConfigurations(options);
    std::vector<Configuration*> configurations;
    Configuration *configuration = nullptr;

    // Reserve space in advance for better performance.
    configurations.reserve(mapConfigurations->size());

    // Convert map representations of maps into structs of type Configuration.
    for (std::vector<MapConfiguration>::iterator mapConfigurationIterator = mapConfigurations->begin(); mapConfigurationIterator != mapConfigurations->end(); ++mapConfigurationIterator) {
        // Set up a new, empty configuration.
        configuration = new Configuration();

        // Set basic properties.
        configuration->timestamp = (*tempDataIndexMap)["timestamp"];
        configuration->timestampHour = (*tempDataIndexMap)["timestampHour"];
        configuration->timestampMinute = (*tempDataIndexMap)["timestampMinute"];
        configuration->open = (*tempDataIndexMap)["open"];
        configuration->high = (*tempDataIndexMap)["high"];
        configuration->low = (*tempDataIndexMap)["low"];
        configuration->close = (*tempDataIndexMap)["close"];

        // Set index mappings.
        if ((*mapConfigurationIterator).find("sma13") != (*mapConfigurationIterator).end()) {
            configuration->sma13 = boost::get<int>((*mapConfigurationIterator)["sma13"]);
        }
        if ((*mapConfigurationIterator).find("ema50") != (*mapConfigurationIterator).end()) {
            configuration->ema50 = boost::get<int>((*mapConfigurationIterator)["ema50"]);
        }
        if ((*mapConfigurationIterator).find("ema100") != (*mapConfigurationIterator).end()) {
            configuration->ema100 = boost::get<int>((*mapConfigurationIterator)["ema100"]);
        }
        if ((*mapConfigurationIterator).find("ema200") != (*mapConfigurationIterator).end()) {
            configuration->ema200 = boost::get<int>((*mapConfigurationIterator)["ema200"]);
        }
        // if ((*mapConfigurationIterator).find("ema250") != (*mapConfigurationIterator).end()) {
        //     configuration->ema250 = boost::get<int>((*mapConfigurationIterator)["ema250"]);
        // }
        // if ((*mapConfigurationIterator).find("ema300") != (*mapConfigurationIterator).end()) {
        //     configuration->ema300 = boost::get<int>((*mapConfigurationIterator)["ema300"]);
        // }
        // if ((*mapConfigurationIterator).find("ema350") != (*mapConfigurationIterator).end()) {
        //     configuration->ema350 = boost::get<int>((*mapConfigurationIterator)["ema350"]);
        // }
        // if ((*mapConfigurationIterator).find("ema400") != (*mapConfigurationIterator).end()) {
        //     configuration->ema400 = boost::get<int>((*mapConfigurationIterator)["ema400"]);
        // }
        // if ((*mapConfigurationIterator).find("ema450") != (*mapConfigurationIterator).end()) {
        //     configuration->ema450 = boost::get<int>((*mapConfigurationIterator)["ema450"]);
        // }
        // if ((*mapConfigurationIterator).find("ema500") != (*mapConfigurationIterator).end()) {
        //     configuration->ema500 = boost::get<int>((*mapConfigurationIterator)["ema500"]);
        // }
        if ((*mapConfigurationIterator).find("rsi") != (*mapConfigurationIterator).end()) {
            configuration->rsi = boost::get<int>((*mapConfigurationIterator)["rsi"]);
        }
        if ((*mapConfigurationIterator).find("stochasticD") != (*mapConfigurationIterator).end()) {
            configuration->stochasticD = boost::get<int>((*mapConfigurationIterator)["stochasticD"]);
        }
        if ((*mapConfigurationIterator).find("stochasticK") != (*mapConfigurationIterator).end()) {
            configuration->stochasticK = boost::get<int>((*mapConfigurationIterator)["stochasticK"]);
        }
        if ((*mapConfigurationIterator).find("prChannelUpper") != (*mapConfigurationIterator).end()) {
            configuration->prChannelUpper = boost::get<int>((*mapConfigurationIterator)["prChannelUpper"]);
        }
        if ((*mapConfigurationIterator).find("prChannelLower") != (*mapConfigurationIterator).end()) {
            configuration->prChannelLower = boost::get<int>((*mapConfigurationIterator)["prChannelLower"]);
        }

        // Set values.
        if ((*mapConfigurationIterator).find("rsiOverbought") != (*mapConfigurationIterator).end()) {
            configuration->rsiOverbought = boost::get<double>((*mapConfigurationIterator)["rsiOverbought"]);
        }
        if ((*mapConfigurationIterator).find("rsiOversold") != (*mapConfigurationIterator).end()) {
            configuration->rsiOversold = boost::get<double>((*mapConfigurationIterator)["rsiOversold"]);
        }
        if ((*mapConfigurationIterator).find("stochasticOverbought") != (*mapConfigurationIterator).end()) {
            configuration->stochasticOverbought = boost::get<double>((*mapConfigurationIterator)["stochasticOverbought"]);
        }
        if ((*mapConfigurationIterator).find("stochasticOversold") != (*mapConfigurationIterator).end()) {
            configuration->stochasticOversold = boost::get<double>((*mapConfigurationIterator)["stochasticOversold"]);
        }

        configurations.push_back(configuration);
    }

    printf("%i configurations built\n", (int)configurations.size());

    return configurations;
}

bson_t *ReversalsOptimizer::convertResultToBson(StrategyResult &result) {
    bson_t *document;
    bson_t configurationDocument;

    document = bson_new();

    // Include basic information.
    BSON_APPEND_UTF8(document, "symbol", getSymbol().c_str());
    BSON_APPEND_INT32(document, "group", getGroup());
    BSON_APPEND_UTF8(document, "strategyName", getStrategyName().c_str());

    // Include stats.
    BSON_APPEND_DOUBLE(document, "profitLoss", result.profitLoss);
    BSON_APPEND_INT32(document, "winCount", result.winCount);
    BSON_APPEND_INT32(document, "loseCount", result.loseCount);
    BSON_APPEND_INT32(document, "tradeCount", result.tradeCount);
    BSON_APPEND_DOUBLE(document, "winRate", result.winRate);
    BSON_APPEND_INT32(document, "maximumConsecutiveLosses", result.maximumConsecutiveLosses);
    BSON_APPEND_INT32(document, "minimumProfitLoss", result.minimumProfitLoss);
    BSON_APPEND_DOCUMENT_BEGIN(document, "configuration", &configurationDocument);

    // Include study settings.
    BSON_APPEND_BOOL(&configurationDocument, "sma13", result.configuration->sma13 > 0);
    BSON_APPEND_BOOL(&configurationDocument, "ema50", result.configuration->ema50 > 0);
    BSON_APPEND_BOOL(&configurationDocument, "ema100", result.configuration->ema100 > 0);
    BSON_APPEND_BOOL(&configurationDocument, "ema200", result.configuration->ema200 > 0);
    BSON_APPEND_BOOL(&configurationDocument, "ema250", result.configuration->ema250 > 0);
    BSON_APPEND_BOOL(&configurationDocument, "ema300", result.configuration->ema300 > 0);
    BSON_APPEND_BOOL(&configurationDocument, "ema350", result.configuration->ema350 > 0);
    BSON_APPEND_BOOL(&configurationDocument, "ema400", result.configuration->ema400 > 0);
    BSON_APPEND_BOOL(&configurationDocument, "ema450", result.configuration->ema450 > 0);
    BSON_APPEND_BOOL(&configurationDocument, "ema500", result.configuration->ema500 > 0);
    if (result.configuration->rsi > 0) {
        BSON_APPEND_UTF8(&configurationDocument, "rsi", findDataIndexMapKeyByValue(result.configuration->rsi).c_str());
        BSON_APPEND_DOUBLE(&configurationDocument, "rsiOverbought", result.configuration->rsiOverbought);
        BSON_APPEND_DOUBLE(&configurationDocument, "rsiOversold", result.configuration->rsiOversold);
    }
    else {
        BSON_APPEND_BOOL(&configurationDocument, "rsi", false);
    }
    if (result.configuration->stochasticD > 0 && result.configuration->stochasticK > 0) {
        BSON_APPEND_UTF8(&configurationDocument, "stochasticD", findDataIndexMapKeyByValue(result.configuration->stochasticD).c_str());
        BSON_APPEND_UTF8(&configurationDocument, "stochasticK", findDataIndexMapKeyByValue(result.configuration->stochasticK).c_str());
        BSON_APPEND_DOUBLE(&configurationDocument, "stochasticOverbought", result.configuration->stochasticOverbought);
        BSON_APPEND_DOUBLE(&configurationDocument, "stochasticOversold", result.configuration->stochasticOversold);
    }
    else {
        BSON_APPEND_BOOL(&configurationDocument, "stochasticD", false);
        BSON_APPEND_BOOL(&configurationDocument, "stochasticK", false);
    }
    if (result.configuration->prChannelUpper > 0 && result.configuration->prChannelLower > 0) {
        BSON_APPEND_UTF8(&configurationDocument, "prChannelUpper", findDataIndexMapKeyByValue(result.configuration->prChannelUpper).c_str());
        BSON_APPEND_UTF8(&configurationDocument, "prChannelLower", findDataIndexMapKeyByValue(result.configuration->prChannelLower).c_str());
    }
    else {
        BSON_APPEND_BOOL(&configurationDocument, "prChannelUpper", false);
        BSON_APPEND_BOOL(&configurationDocument, "prChannelLower", false);
    }

    bson_append_document_end(document, &configurationDocument);

    return document;
}
