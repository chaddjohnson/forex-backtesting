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

std::vector<Configuration*> ReversalsOptimizer::buildBaseConfigurations() {
    printf("Building base configurations...");

    std::map<std::string, ConfigurationOption> options = getConfigurationOptions();
    std::map<std::string, int> *tempDataIndexMap = getDataIndexMap();
    std::vector<MapConfiguration> *mapConfigurations = buildMapConfigurations(options);
    std::vector<Configuration*> configurations;
    Configuration *configuration = nullptr;

    // Reserve space in advance for better performance.
    configurations.reserve(mapConfigurations->size());

    // Convert map representations of maps into structs of type Configuration.
    for (std::vector<MapConfiguration>::iterator mapvalueIterator = mapConfigurations->begin(); mapvalueIterator != mapConfigurations->end(); ++mapvalueIterator) {
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
        if ((*mapvalueIterator).find("sma13") != (*mapvalueIterator).end()) {
            configuration->sma13 = boost::get<int>((*mapvalueIterator)["sma13"]);
        }
        if ((*mapvalueIterator).find("ema50") != (*mapvalueIterator).end()) {
            configuration->ema50 = boost::get<int>((*mapvalueIterator)["ema50"]);
        }
        if ((*mapvalueIterator).find("ema100") != (*mapvalueIterator).end()) {
            configuration->ema100 = boost::get<int>((*mapvalueIterator)["ema100"]);
        }
        if ((*mapvalueIterator).find("ema200") != (*mapvalueIterator).end()) {
            configuration->ema200 = boost::get<int>((*mapvalueIterator)["ema200"]);
        }
        // if ((*mapvalueIterator).find("ema250") != (*mapvalueIterator).end()) {
        //     configuration->ema250 = boost::get<int>((*mapvalueIterator)["ema250"]);
        // }
        // if ((*mapvalueIterator).find("ema300") != (*mapvalueIterator).end()) {
        //     configuration->ema300 = boost::get<int>((*mapvalueIterator)["ema300"]);
        // }
        // if ((*mapvalueIterator).find("ema350") != (*mapvalueIterator).end()) {
        //     configuration->ema350 = boost::get<int>((*mapvalueIterator)["ema350"]);
        // }
        // if ((*mapvalueIterator).find("ema400") != (*mapvalueIterator).end()) {
        //     configuration->ema400 = boost::get<int>((*mapvalueIterator)["ema400"]);
        // }
        // if ((*mapvalueIterator).find("ema450") != (*mapvalueIterator).end()) {
        //     configuration->ema450 = boost::get<int>((*mapvalueIterator)["ema450"]);
        // }
        // if ((*mapvalueIterator).find("ema500") != (*mapvalueIterator).end()) {
        //     configuration->ema500 = boost::get<int>((*mapvalueIterator)["ema500"]);
        // }
        if ((*mapvalueIterator).find("rsi") != (*mapvalueIterator).end()) {
            configuration->rsi = boost::get<int>((*mapvalueIterator)["rsi"]);
        }
        if ((*mapvalueIterator).find("stochasticD") != (*mapvalueIterator).end()) {
            configuration->stochasticD = boost::get<int>((*mapvalueIterator)["stochasticD"]);
        }
        if ((*mapvalueIterator).find("stochasticK") != (*mapvalueIterator).end()) {
            configuration->stochasticK = boost::get<int>((*mapvalueIterator)["stochasticK"]);
        }
        if ((*mapvalueIterator).find("prChannelUpper") != (*mapvalueIterator).end()) {
            configuration->prChannelUpper = boost::get<int>((*mapvalueIterator)["prChannelUpper"]);
        }
        if ((*mapvalueIterator).find("prChannelLower") != (*mapvalueIterator).end()) {
            configuration->prChannelLower = boost::get<int>((*mapvalueIterator)["prChannelLower"]);
        }

        // Set values.
        if ((*mapvalueIterator).find("rsiOverbought") != (*mapvalueIterator).end()) {
            configuration->rsiOverbought = boost::get<double>((*mapvalueIterator)["rsiOverbought"]);
        }
        if ((*mapvalueIterator).find("rsiOversold") != (*mapvalueIterator).end()) {
            configuration->rsiOversold = boost::get<double>((*mapvalueIterator)["rsiOversold"]);
        }
        if ((*mapvalueIterator).find("stochasticOverbought") != (*mapvalueIterator).end()) {
            configuration->stochasticOverbought = boost::get<double>((*mapvalueIterator)["stochasticOverbought"]);
        }
        if ((*mapvalueIterator).find("stochasticOversold") != (*mapvalueIterator).end()) {
            configuration->stochasticOversold = boost::get<double>((*mapvalueIterator)["stochasticOversold"]);
        }

        configurations.push_back(configuration);
    }

    printf("%i configurations built\n", (int)configurations.size());

    return configurations;
}

std::vector<Configuration*> ReversalsOptimizer::buildGroupConfigurations() {
    std::vector<Configuration*> configurations;
    bson_t *query;

    // Default to using the current group.
    int group = getGroup();

    if (getType() == Optimizer::types::TEST) {
        // Testing is being performed, so use configurations from the previous test.
        // If validation is being performed, on the other hand, then we want to
        // validate the current group rather than the previous group.
        group = group - 1;
    }

    printf("Building group %i configurations...", group);

    // Query the database for configurations belonging to the previous (testing) or current (validation) group.
    if (getType() == Optimizer::types::TEST) {
        query = BCON_NEW(
            "$query", "{",
                "symbol", BCON_UTF8(getSymbol().c_str()),
                "group", BCON_INT32(group),
                "strategyName", BCON_UTF8(getStrategyName().c_str()),
                "winRate", "{", "$gte", BCON_DOUBLE(0.60), "}",
            "}"
        );
    }
    else {
        query = BCON_NEW(
            "$query", "{",
                "symbol", BCON_UTF8(getSymbol().c_str()),
                "group", BCON_INT32(group),
                "strategyName", BCON_UTF8(getStrategyName().c_str()),
            "}"
        );
    }

    configurations = loadConfigurations("tests", query);

    // Cleanup.
    bson_destroy(query);

    printf("%i configurations built\n", (int)configurations.size());

    return configurations;
}

std::vector<Configuration*> ReversalsOptimizer::buildSavedConfigurations() {
    std::vector<Configuration*> configurations;
    bson_t *query;

    // Default to using the current group.
    int group = getGroup();

    if (getType() == Optimizer::types::TEST) {
        // Testing is being performed, so use configurations from the previous test.
        // If validation is being performed, on the other hand, then we want to
        // validate the current group rather than the previous group.
        group = group - 1;
    }

    printf("Building saved configurations...");

    // Query the database for configurations belonging to the previous (testing) or current (validation) group.
    query = BCON_NEW(
        "$query", "{",
            "symbol", BCON_UTF8(getSymbol().c_str()),
            "strategyName", BCON_UTF8(getStrategyName().c_str()),
        "}"
    );
    configurations = loadConfigurations("configurations", query);

    // Cleanup.
    bson_destroy(query);

    printf("%i configurations built\n", (int)configurations.size());

    return configurations;
}

std::vector<Configuration*> ReversalsOptimizer::loadConfigurations(const char *collectionName, bson_t *query) {
    std::vector<Configuration*> configurations;
    std::map<std::string, int> *tempDataIndexMap = getDataIndexMap();
    mongoc_collection_t *collection;
    mongoc_cursor_t *cursor;
    const bson_t *document;
    bson_iter_t documentIterator;
    bson_iter_t configurationIterator;
    std::string propertyName;
    const bson_value_t *propertyValue;

    // Get a reference to the database collection.
    collection = mongoc_client_get_collection(getDbClient(), "forex-backtesting", collectionName);

    // Run the query and get a cursor.
    cursor = mongoc_collection_find(collection, MONGOC_QUERY_NONE, 0, 0, 1000, query, NULL, NULL);

    while (mongoc_cursor_next(cursor, &document)) {
        if (bson_iter_init(&documentIterator, document)) {
            // Find the "data" subdocument.
            if (bson_iter_init_find(&documentIterator, document, "configuration") &&
                BSON_ITER_HOLDS_DOCUMENT(&documentIterator) &&
                bson_iter_recurse(&documentIterator, &configurationIterator))
            {
                Configuration *resultConfiguration = new Configuration();

                // basic properties
                resultConfiguration->timestamp = (*tempDataIndexMap)["timestamp"];
                resultConfiguration->timestampHour = (*tempDataIndexMap)["timestampHour"];
                resultConfiguration->timestampMinute = (*tempDataIndexMap)["timestampMinute"];
                resultConfiguration->open = (*tempDataIndexMap)["open"];
                resultConfiguration->high = (*tempDataIndexMap)["high"];
                resultConfiguration->low = (*tempDataIndexMap)["low"];
                resultConfiguration->close = (*tempDataIndexMap)["close"];

                // Iterate through the configuration properties.
                while (bson_iter_next(&configurationIterator)) {
                    // Get the property name and value.
                    propertyName = std::string(bson_iter_key(&configurationIterator));
                    propertyValue = bson_iter_value(&configurationIterator);

                    if (propertyName == "sma13") {
                        if (propertyValue->value.v_bool) {
                            resultConfiguration->sma13 = (*tempDataIndexMap)["sma13"];
                        }
                    }
                    else if (propertyName == "ema50") {
                        if (propertyValue->value.v_bool) {
                            resultConfiguration->ema50 = (*tempDataIndexMap)["ema50"];
                        }
                    }
                    else if (propertyName == "ema100") {
                        if (propertyValue->value.v_bool) {
                            resultConfiguration->ema100 = (*tempDataIndexMap)["ema100"];
                        }
                    }
                    else if (propertyName == "ema200") {
                        if (propertyValue->value.v_bool) {
                            resultConfiguration->ema200 = (*tempDataIndexMap)["ema200"];
                        }
                    }
                    // else if (propertyName == "ema250") {
                    //     if (propertyValue->value.v_bool) {
                    //         resultConfiguration->ema250 = (*tempDataIndexMap)["ema250"];
                    //     }
                    // }
                    // else if (propertyName == "ema300") {
                    //     if (propertyValue->value.v_bool) {
                    //         resultConfiguration->ema300 = (*tempDataIndexMap)["ema300"];
                    //     }
                    // }
                    // else if (propertyName == "ema350") {
                    //     if (propertyValue->value.v_bool) {
                    //         resultConfiguration->ema350 = (*tempDataIndexMap)["ema350"];
                    //     }
                    // }
                    // else if (propertyName == "ema400") {
                    //     if (propertyValue->value.v_bool) {
                    //         resultConfiguration->ema400 = (*tempDataIndexMap)["ema400"];
                    //     }
                    // }
                    // else if (propertyName == "ema450") {
                    //     if (propertyValue->value.v_bool) {
                    //         resultConfiguration->ema450 = (*tempDataIndexMap)["ema450"];
                    //     }
                    // }
                    // else if (propertyName == "ema500") {
                    //     if (propertyValue->value.v_bool) {
                    //         resultConfiguration->ema500 = (*tempDataIndexMap)["ema500"];
                    //     }
                    // }
                    else if (propertyName == "rsi") {
                        if (propertyValue->value_type != BSON_TYPE_BOOL) {
                            resultConfiguration->rsi = (*tempDataIndexMap)[propertyValue->value.v_utf8.str];
                        }
                    }
                    else if (propertyName == "rsiOverbought") {
                        resultConfiguration->rsiOverbought = propertyValue->value.v_double;
                    }
                    else if (propertyName == "rsiOversold") {
                        resultConfiguration->rsiOversold = propertyValue->value.v_double;
                    }
                    else if (propertyName == "stochasticD") {
                        if (propertyValue->value_type != BSON_TYPE_BOOL) {
                            resultConfiguration->stochasticD = (*tempDataIndexMap)[propertyValue->value.v_utf8.str];
                        }
                    }
                    else if (propertyName == "stochasticK") {
                        if (propertyValue->value_type != BSON_TYPE_BOOL) {
                            resultConfiguration->stochasticK = (*tempDataIndexMap)[propertyValue->value.v_utf8.str];
                        }
                    }
                    else if (propertyName == "stochasticOverbought") {
                        resultConfiguration->stochasticOverbought = propertyValue->value.v_double;
                    }
                    else if (propertyName == "stochasticOversold") {
                        resultConfiguration->stochasticOversold = propertyValue->value.v_double;
                    }
                    else if (propertyName == "prChannelUpper") {
                        resultConfiguration->prChannelUpper = (*tempDataIndexMap)[propertyValue->value.v_utf8.str];
                    }
                    else if (propertyName == "prChannelLower") {
                        resultConfiguration->prChannelLower = (*tempDataIndexMap)[propertyValue->value.v_utf8.str];
                    }
                }

                // Add the configuration to the list of configurations.
                configurations.push_back(resultConfiguration);
            }
        }
    }

    // Cleanup.
    mongoc_collection_destroy(collection);
    mongoc_cursor_destroy(cursor);

    return configurations;
}

bson_t *ReversalsOptimizer::convertResultToBson(StrategyResult &result) {
    bson_t *document;
    bson_t configurationDocument;

    document = bson_new();

    // Include basic information.
    BSON_APPEND_UTF8(document, "symbol", getSymbol().c_str());
    BSON_APPEND_UTF8(document, "strategyName", getStrategyName().c_str());

    if (getType() == Optimizer::types::TEST || getType() == Optimizer::types::VALIDATION) {
        BSON_APPEND_INT32(document, "group", getGroup());
    }

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
