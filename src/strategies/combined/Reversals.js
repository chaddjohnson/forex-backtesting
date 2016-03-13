var Base = require('./Base');
var Call = require('../../positions/Call');
var Put = require('../../positions/Put');
var studies = require('../../studies');

var studyDefinitions = [
    {study: studies.Sma, inputs: {length: 13}, outputMap: {sma: 'sma13'}},
    {study: studies.Ema, inputs: {length: 50}, outputMap: {ema: 'ema50'}},
    {study: studies.Ema, inputs: {length: 100}, outputMap: {ema: 'ema100'}},
    {study: studies.Ema, inputs: {length: 200}, outputMap: {ema: 'ema200'}},
    {study: studies.Rsi, inputs: {length: 14}, outputMap: {rsi: 'rsi14'}},
    {study: studies.Rsi, inputs: {length: 2}, outputMap: {rsi: 'rsi2'}},
    {study: studies.Rsi, inputs: {length: 5}, outputMap: {rsi: 'rsi5'}},
    {study: studies.Rsi, inputs: {length: 7}, outputMap: {rsi: 'rsi7'}},
    {study: studies.Rsi, inputs: {length: 9}, outputMap: {rsi: 'rsi9'}},
    {study: studies.StochasticOscillator, inputs: {length: 10, averageLength: 3}, outputMap: {K: 'stochastic10K', D: 'stochastic10D'}},
    {study: studies.StochasticOscillator, inputs: {length: 14, averageLength: 3}, outputMap: {K: 'stochastic14K', D: 'stochastic14D'}},
    {study: studies.StochasticOscillator, inputs: {length: 21, averageLength: 3}, outputMap: {K: 'stochastic21K', D: 'stochastic21D'}},
    {study: studies.StochasticOscillator, inputs: {length: 5, averageLength: 3}, outputMap: {K: 'stochastic5K', D: 'stochastic5D'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 100, degree: 2, deviations: 1.85}, outputMap: {regression: 'prChannel100_2_185', upper: 'prChannelUpper100_2_185', lower: 'prChannelLower100_2_185'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 100, degree: 2, deviations: 1.95}, outputMap: {regression: 'prChannel100_2_195', upper: 'prChannelUpper100_2_195', lower: 'prChannelLower100_2_195'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 100, degree: 2, deviations: 1.9}, outputMap: {regression: 'prChannel100_2_19', upper: 'prChannelUpper100_2_19', lower: 'prChannelLower100_2_19'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 100, degree: 2, deviations: 2.05}, outputMap: {regression: 'prChannel100_2_205', upper: 'prChannelUpper100_2_205', lower: 'prChannelLower100_2_205'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 100, degree: 2, deviations: 2.0}, outputMap: {regression: 'prChannel100_2_20', upper: 'prChannelUpper100_2_20', lower: 'prChannelLower100_2_20'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 100, degree: 2, deviations: 2.15}, outputMap: {regression: 'prChannel100_2_215', upper: 'prChannelUpper100_2_215', lower: 'prChannelLower100_2_215'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 100, degree: 2, deviations: 2.1}, outputMap: {regression: 'prChannel100_2_21', upper: 'prChannelUpper100_2_21', lower: 'prChannelLower100_2_21'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 100, degree: 3, deviations: 1.85}, outputMap: {regression: 'prChannel100_3_185', upper: 'prChannelUpper100_3_185', lower: 'prChannelLower100_3_185'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 100, degree: 3, deviations: 1.95}, outputMap: {regression: 'prChannel100_3_195', upper: 'prChannelUpper100_3_195', lower: 'prChannelLower100_3_195'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 100, degree: 3, deviations: 1.9}, outputMap: {regression: 'prChannel100_3_19', upper: 'prChannelUpper100_3_19', lower: 'prChannelLower100_3_19'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 100, degree: 3, deviations: 2.05}, outputMap: {regression: 'prChannel100_3_205', upper: 'prChannelUpper100_3_205', lower: 'prChannelLower100_3_205'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 100, degree: 3, deviations: 2.0}, outputMap: {regression: 'prChannel100_3_20', upper: 'prChannelUpper100_3_20', lower: 'prChannelLower100_3_20'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 100, degree: 3, deviations: 2.15}, outputMap: {regression: 'prChannel100_3_215', upper: 'prChannelUpper100_3_215', lower: 'prChannelLower100_3_215'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 100, degree: 3, deviations: 2.1}, outputMap: {regression: 'prChannel100_3_21', upper: 'prChannelUpper100_3_21', lower: 'prChannelLower100_3_21'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 100, degree: 4, deviations: 1.85}, outputMap: {regression: 'prChannel100_4_185', upper: 'prChannelUpper100_4_185', lower: 'prChannelLower100_4_185'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 100, degree: 4, deviations: 1.95}, outputMap: {regression: 'prChannel100_4_195', upper: 'prChannelUpper100_4_195', lower: 'prChannelLower100_4_195'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 100, degree: 4, deviations: 1.9}, outputMap: {regression: 'prChannel100_4_19', upper: 'prChannelUpper100_4_19', lower: 'prChannelLower100_4_19'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 100, degree: 4, deviations: 2.05}, outputMap: {regression: 'prChannel100_4_205', upper: 'prChannelUpper100_4_205', lower: 'prChannelLower100_4_205'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 100, degree: 4, deviations: 2.0}, outputMap: {regression: 'prChannel100_4_20', upper: 'prChannelUpper100_4_20', lower: 'prChannelLower100_4_20'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 100, degree: 4, deviations: 2.15}, outputMap: {regression: 'prChannel100_4_215', upper: 'prChannelUpper100_4_215', lower: 'prChannelLower100_4_215'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 100, degree: 4, deviations: 2.1}, outputMap: {regression: 'prChannel100_4_21', upper: 'prChannelUpper100_4_21', lower: 'prChannelLower100_4_21'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 100, degree: 5, deviations: 1.85}, outputMap: {regression: 'prChannel100_5_185', upper: 'prChannelUpper100_5_185', lower: 'prChannelLower100_5_185'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 100, degree: 5, deviations: 2.05}, outputMap: {regression: 'prChannel100_5_205', upper: 'prChannelUpper100_5_205', lower: 'prChannelLower100_5_205'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 200, degree: 2, deviations: 1.85}, outputMap: {regression: 'prChannel200_2_185', upper: 'prChannelUpper200_2_185', lower: 'prChannelLower200_2_185'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 200, degree: 2, deviations: 1.95}, outputMap: {regression: 'prChannel200_2_195', upper: 'prChannelUpper200_2_195', lower: 'prChannelLower200_2_195'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 200, degree: 2, deviations: 1.9}, outputMap: {regression: 'prChannel200_2_19', upper: 'prChannelUpper200_2_19', lower: 'prChannelLower200_2_19'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 200, degree: 2, deviations: 2.05}, outputMap: {regression: 'prChannel200_2_205', upper: 'prChannelUpper200_2_205', lower: 'prChannelLower200_2_205'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 200, degree: 2, deviations: 2.0}, outputMap: {regression: 'prChannel200_2_20', upper: 'prChannelUpper200_2_20', lower: 'prChannelLower200_2_20'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 200, degree: 2, deviations: 2.15}, outputMap: {regression: 'prChannel200_2_215', upper: 'prChannelUpper200_2_215', lower: 'prChannelLower200_2_215'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 200, degree: 2, deviations: 2.1}, outputMap: {regression: 'prChannel200_2_21', upper: 'prChannelUpper200_2_21', lower: 'prChannelLower200_2_21'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 200, degree: 3, deviations: 1.85}, outputMap: {regression: 'prChannel200_3_185', upper: 'prChannelUpper200_3_185', lower: 'prChannelLower200_3_185'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 200, degree: 3, deviations: 1.95}, outputMap: {regression: 'prChannel200_3_195', upper: 'prChannelUpper200_3_195', lower: 'prChannelLower200_3_195'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 200, degree: 3, deviations: 1.9}, outputMap: {regression: 'prChannel200_3_19', upper: 'prChannelUpper200_3_19', lower: 'prChannelLower200_3_19'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 200, degree: 3, deviations: 2.05}, outputMap: {regression: 'prChannel200_3_205', upper: 'prChannelUpper200_3_205', lower: 'prChannelLower200_3_205'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 200, degree: 3, deviations: 2.0}, outputMap: {regression: 'prChannel200_3_20', upper: 'prChannelUpper200_3_20', lower: 'prChannelLower200_3_20'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 200, degree: 3, deviations: 2.15}, outputMap: {regression: 'prChannel200_3_215', upper: 'prChannelUpper200_3_215', lower: 'prChannelLower200_3_215'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 200, degree: 3, deviations: 2.1}, outputMap: {regression: 'prChannel200_3_21', upper: 'prChannelUpper200_3_21', lower: 'prChannelLower200_3_21'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 200, degree: 4, deviations: 1.85}, outputMap: {regression: 'prChannel200_4_185', upper: 'prChannelUpper200_4_185', lower: 'prChannelLower200_4_185'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 200, degree: 4, deviations: 1.95}, outputMap: {regression: 'prChannel200_4_195', upper: 'prChannelUpper200_4_195', lower: 'prChannelLower200_4_195'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 200, degree: 4, deviations: 1.9}, outputMap: {regression: 'prChannel200_4_19', upper: 'prChannelUpper200_4_19', lower: 'prChannelLower200_4_19'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 200, degree: 4, deviations: 2.05}, outputMap: {regression: 'prChannel200_4_205', upper: 'prChannelUpper200_4_205', lower: 'prChannelLower200_4_205'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 200, degree: 4, deviations: 2.0}, outputMap: {regression: 'prChannel200_4_20', upper: 'prChannelUpper200_4_20', lower: 'prChannelLower200_4_20'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 200, degree: 4, deviations: 2.15}, outputMap: {regression: 'prChannel200_4_215', upper: 'prChannelUpper200_4_215', lower: 'prChannelLower200_4_215'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 200, degree: 4, deviations: 2.1}, outputMap: {regression: 'prChannel200_4_21', upper: 'prChannelUpper200_4_21', lower: 'prChannelLower200_4_21'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 200, degree: 5, deviations: 1.85}, outputMap: {regression: 'prChannel200_5_185', upper: 'prChannelUpper200_5_185', lower: 'prChannelLower200_5_185'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 200, degree: 5, deviations: 2.05}, outputMap: {regression: 'prChannel200_5_205', upper: 'prChannelUpper200_5_205', lower: 'prChannelLower200_5_205'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 250, degree: 2, deviations: 1.85}, outputMap: {regression: 'prChannel250_2_185', upper: 'prChannelUpper250_2_185', lower: 'prChannelLower250_2_185'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 250, degree: 2, deviations: 1.95}, outputMap: {regression: 'prChannel250_2_195', upper: 'prChannelUpper250_2_195', lower: 'prChannelLower250_2_195'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 250, degree: 2, deviations: 1.9}, outputMap: {regression: 'prChannel250_2_19', upper: 'prChannelUpper250_2_19', lower: 'prChannelLower250_2_19'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 250, degree: 2, deviations: 2.05}, outputMap: {regression: 'prChannel250_2_205', upper: 'prChannelUpper250_2_205', lower: 'prChannelLower250_2_205'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 250, degree: 2, deviations: 2.0}, outputMap: {regression: 'prChannel250_2_20', upper: 'prChannelUpper250_2_20', lower: 'prChannelLower250_2_20'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 250, degree: 2, deviations: 2.15}, outputMap: {regression: 'prChannel250_2_215', upper: 'prChannelUpper250_2_215', lower: 'prChannelLower250_2_215'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 250, degree: 2, deviations: 2.1}, outputMap: {regression: 'prChannel250_2_21', upper: 'prChannelUpper250_2_21', lower: 'prChannelLower250_2_21'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 250, degree: 3, deviations: 1.85}, outputMap: {regression: 'prChannel250_3_185', upper: 'prChannelUpper250_3_185', lower: 'prChannelLower250_3_185'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 250, degree: 3, deviations: 1.95}, outputMap: {regression: 'prChannel250_3_195', upper: 'prChannelUpper250_3_195', lower: 'prChannelLower250_3_195'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 250, degree: 3, deviations: 1.9}, outputMap: {regression: 'prChannel250_3_19', upper: 'prChannelUpper250_3_19', lower: 'prChannelLower250_3_19'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 250, degree: 3, deviations: 2.05}, outputMap: {regression: 'prChannel250_3_205', upper: 'prChannelUpper250_3_205', lower: 'prChannelLower250_3_205'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 250, degree: 3, deviations: 2.0}, outputMap: {regression: 'prChannel250_3_20', upper: 'prChannelUpper250_3_20', lower: 'prChannelLower250_3_20'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 250, degree: 3, deviations: 2.15}, outputMap: {regression: 'prChannel250_3_215', upper: 'prChannelUpper250_3_215', lower: 'prChannelLower250_3_215'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 250, degree: 3, deviations: 2.1}, outputMap: {regression: 'prChannel250_3_21', upper: 'prChannelUpper250_3_21', lower: 'prChannelLower250_3_21'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 250, degree: 4, deviations: 1.85}, outputMap: {regression: 'prChannel250_4_185', upper: 'prChannelUpper250_4_185', lower: 'prChannelLower250_4_185'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 250, degree: 4, deviations: 1.95}, outputMap: {regression: 'prChannel250_4_195', upper: 'prChannelUpper250_4_195', lower: 'prChannelLower250_4_195'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 250, degree: 4, deviations: 1.9}, outputMap: {regression: 'prChannel250_4_19', upper: 'prChannelUpper250_4_19', lower: 'prChannelLower250_4_19'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 250, degree: 4, deviations: 2.05}, outputMap: {regression: 'prChannel250_4_205', upper: 'prChannelUpper250_4_205', lower: 'prChannelLower250_4_205'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 250, degree: 4, deviations: 2.0}, outputMap: {regression: 'prChannel250_4_20', upper: 'prChannelUpper250_4_20', lower: 'prChannelLower250_4_20'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 250, degree: 4, deviations: 2.15}, outputMap: {regression: 'prChannel250_4_215', upper: 'prChannelUpper250_4_215', lower: 'prChannelLower250_4_215'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 250, degree: 4, deviations: 2.1}, outputMap: {regression: 'prChannel250_4_21', upper: 'prChannelUpper250_4_21', lower: 'prChannelLower250_4_21'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 250, degree: 5, deviations: 1.85}, outputMap: {regression: 'prChannel250_5_185', upper: 'prChannelUpper250_5_185', lower: 'prChannelLower250_5_185'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 250, degree: 5, deviations: 2.05}, outputMap: {regression: 'prChannel250_5_205', upper: 'prChannelUpper250_5_205', lower: 'prChannelLower250_5_205'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 300, degree: 2, deviations: 1.85}, outputMap: {regression: 'prChannel300_2_185', upper: 'prChannelUpper300_2_185', lower: 'prChannelLower300_2_185'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 300, degree: 2, deviations: 1.95}, outputMap: {regression: 'prChannel300_2_195', upper: 'prChannelUpper300_2_195', lower: 'prChannelLower300_2_195'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 300, degree: 2, deviations: 1.9}, outputMap: {regression: 'prChannel300_2_19', upper: 'prChannelUpper300_2_19', lower: 'prChannelLower300_2_19'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 300, degree: 2, deviations: 2.05}, outputMap: {regression: 'prChannel300_2_205', upper: 'prChannelUpper300_2_205', lower: 'prChannelLower300_2_205'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 300, degree: 2, deviations: 2.0}, outputMap: {regression: 'prChannel300_2_20', upper: 'prChannelUpper300_2_20', lower: 'prChannelLower300_2_20'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 300, degree: 2, deviations: 2.15}, outputMap: {regression: 'prChannel300_2_215', upper: 'prChannelUpper300_2_215', lower: 'prChannelLower300_2_215'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 300, degree: 2, deviations: 2.1}, outputMap: {regression: 'prChannel300_2_21', upper: 'prChannelUpper300_2_21', lower: 'prChannelLower300_2_21'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 300, degree: 3, deviations: 1.85}, outputMap: {regression: 'prChannel300_3_185', upper: 'prChannelUpper300_3_185', lower: 'prChannelLower300_3_185'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 300, degree: 3, deviations: 1.95}, outputMap: {regression: 'prChannel300_3_195', upper: 'prChannelUpper300_3_195', lower: 'prChannelLower300_3_195'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 300, degree: 3, deviations: 1.9}, outputMap: {regression: 'prChannel300_3_19', upper: 'prChannelUpper300_3_19', lower: 'prChannelLower300_3_19'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 300, degree: 3, deviations: 2.05}, outputMap: {regression: 'prChannel300_3_205', upper: 'prChannelUpper300_3_205', lower: 'prChannelLower300_3_205'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 300, degree: 3, deviations: 2.0}, outputMap: {regression: 'prChannel300_3_20', upper: 'prChannelUpper300_3_20', lower: 'prChannelLower300_3_20'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 300, degree: 3, deviations: 2.15}, outputMap: {regression: 'prChannel300_3_215', upper: 'prChannelUpper300_3_215', lower: 'prChannelLower300_3_215'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 300, degree: 3, deviations: 2.1}, outputMap: {regression: 'prChannel300_3_21', upper: 'prChannelUpper300_3_21', lower: 'prChannelLower300_3_21'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 300, degree: 4, deviations: 1.85}, outputMap: {regression: 'prChannel300_4_185', upper: 'prChannelUpper300_4_185', lower: 'prChannelLower300_4_185'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 300, degree: 4, deviations: 1.95}, outputMap: {regression: 'prChannel300_4_195', upper: 'prChannelUpper300_4_195', lower: 'prChannelLower300_4_195'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 300, degree: 4, deviations: 1.9}, outputMap: {regression: 'prChannel300_4_19', upper: 'prChannelUpper300_4_19', lower: 'prChannelLower300_4_19'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 300, degree: 4, deviations: 2.05}, outputMap: {regression: 'prChannel300_4_205', upper: 'prChannelUpper300_4_205', lower: 'prChannelLower300_4_205'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 300, degree: 4, deviations: 2.0}, outputMap: {regression: 'prChannel300_4_20', upper: 'prChannelUpper300_4_20', lower: 'prChannelLower300_4_20'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 300, degree: 4, deviations: 2.15}, outputMap: {regression: 'prChannel300_4_215', upper: 'prChannelUpper300_4_215', lower: 'prChannelLower300_4_215'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 300, degree: 4, deviations: 2.1}, outputMap: {regression: 'prChannel300_4_21', upper: 'prChannelUpper300_4_21', lower: 'prChannelLower300_4_21'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 300, degree: 5, deviations: 1.85}, outputMap: {regression: 'prChannel300_5_185', upper: 'prChannelUpper300_5_185', lower: 'prChannelLower300_5_185'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 300, degree: 5, deviations: 2.05}, outputMap: {regression: 'prChannel300_5_205', upper: 'prChannelUpper300_5_205', lower: 'prChannelLower300_5_205'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 350, degree: 2, deviations: 1.85}, outputMap: {regression: 'prChannel350_2_185', upper: 'prChannelUpper350_2_185', lower: 'prChannelLower350_2_185'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 350, degree: 2, deviations: 1.95}, outputMap: {regression: 'prChannel350_2_195', upper: 'prChannelUpper350_2_195', lower: 'prChannelLower350_2_195'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 350, degree: 2, deviations: 1.9}, outputMap: {regression: 'prChannel350_2_19', upper: 'prChannelUpper350_2_19', lower: 'prChannelLower350_2_19'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 350, degree: 2, deviations: 2.05}, outputMap: {regression: 'prChannel350_2_205', upper: 'prChannelUpper350_2_205', lower: 'prChannelLower350_2_205'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 350, degree: 2, deviations: 2.0}, outputMap: {regression: 'prChannel350_2_20', upper: 'prChannelUpper350_2_20', lower: 'prChannelLower350_2_20'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 350, degree: 2, deviations: 2.15}, outputMap: {regression: 'prChannel350_2_215', upper: 'prChannelUpper350_2_215', lower: 'prChannelLower350_2_215'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 350, degree: 2, deviations: 2.1}, outputMap: {regression: 'prChannel350_2_21', upper: 'prChannelUpper350_2_21', lower: 'prChannelLower350_2_21'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 350, degree: 3, deviations: 1.85}, outputMap: {regression: 'prChannel350_3_185', upper: 'prChannelUpper350_3_185', lower: 'prChannelLower350_3_185'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 350, degree: 3, deviations: 1.95}, outputMap: {regression: 'prChannel350_3_195', upper: 'prChannelUpper350_3_195', lower: 'prChannelLower350_3_195'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 350, degree: 3, deviations: 1.9}, outputMap: {regression: 'prChannel350_3_19', upper: 'prChannelUpper350_3_19', lower: 'prChannelLower350_3_19'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 350, degree: 3, deviations: 2.05}, outputMap: {regression: 'prChannel350_3_205', upper: 'prChannelUpper350_3_205', lower: 'prChannelLower350_3_205'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 350, degree: 3, deviations: 2.0}, outputMap: {regression: 'prChannel350_3_20', upper: 'prChannelUpper350_3_20', lower: 'prChannelLower350_3_20'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 350, degree: 3, deviations: 2.15}, outputMap: {regression: 'prChannel350_3_215', upper: 'prChannelUpper350_3_215', lower: 'prChannelLower350_3_215'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 350, degree: 3, deviations: 2.1}, outputMap: {regression: 'prChannel350_3_21', upper: 'prChannelUpper350_3_21', lower: 'prChannelLower350_3_21'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 350, degree: 4, deviations: 1.85}, outputMap: {regression: 'prChannel350_4_185', upper: 'prChannelUpper350_4_185', lower: 'prChannelLower350_4_185'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 350, degree: 4, deviations: 1.95}, outputMap: {regression: 'prChannel350_4_195', upper: 'prChannelUpper350_4_195', lower: 'prChannelLower350_4_195'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 350, degree: 4, deviations: 1.9}, outputMap: {regression: 'prChannel350_4_19', upper: 'prChannelUpper350_4_19', lower: 'prChannelLower350_4_19'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 350, degree: 4, deviations: 2.05}, outputMap: {regression: 'prChannel350_4_205', upper: 'prChannelUpper350_4_205', lower: 'prChannelLower350_4_205'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 350, degree: 4, deviations: 2.0}, outputMap: {regression: 'prChannel350_4_20', upper: 'prChannelUpper350_4_20', lower: 'prChannelLower350_4_20'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 350, degree: 4, deviations: 2.15}, outputMap: {regression: 'prChannel350_4_215', upper: 'prChannelUpper350_4_215', lower: 'prChannelLower350_4_215'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 350, degree: 4, deviations: 2.1}, outputMap: {regression: 'prChannel350_4_21', upper: 'prChannelUpper350_4_21', lower: 'prChannelLower350_4_21'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 350, degree: 5, deviations: 1.85}, outputMap: {regression: 'prChannel350_5_185', upper: 'prChannelUpper350_5_185', lower: 'prChannelLower350_5_185'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 350, degree: 5, deviations: 2.05}, outputMap: {regression: 'prChannel350_5_205', upper: 'prChannelUpper350_5_205', lower: 'prChannelLower350_5_205'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 400, degree: 2, deviations: 1.85}, outputMap: {regression: 'prChannel400_2_185', upper: 'prChannelUpper400_2_185', lower: 'prChannelLower400_2_185'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 400, degree: 2, deviations: 1.95}, outputMap: {regression: 'prChannel400_2_195', upper: 'prChannelUpper400_2_195', lower: 'prChannelLower400_2_195'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 400, degree: 2, deviations: 1.9}, outputMap: {regression: 'prChannel400_2_19', upper: 'prChannelUpper400_2_19', lower: 'prChannelLower400_2_19'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 400, degree: 2, deviations: 2.05}, outputMap: {regression: 'prChannel400_2_205', upper: 'prChannelUpper400_2_205', lower: 'prChannelLower400_2_205'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 400, degree: 2, deviations: 2.0}, outputMap: {regression: 'prChannel400_2_20', upper: 'prChannelUpper400_2_20', lower: 'prChannelLower400_2_20'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 400, degree: 2, deviations: 2.15}, outputMap: {regression: 'prChannel400_2_215', upper: 'prChannelUpper400_2_215', lower: 'prChannelLower400_2_215'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 400, degree: 2, deviations: 2.1}, outputMap: {regression: 'prChannel400_2_21', upper: 'prChannelUpper400_2_21', lower: 'prChannelLower400_2_21'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 400, degree: 3, deviations: 1.85}, outputMap: {regression: 'prChannel400_3_185', upper: 'prChannelUpper400_3_185', lower: 'prChannelLower400_3_185'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 400, degree: 3, deviations: 1.95}, outputMap: {regression: 'prChannel400_3_195', upper: 'prChannelUpper400_3_195', lower: 'prChannelLower400_3_195'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 400, degree: 3, deviations: 1.9}, outputMap: {regression: 'prChannel400_3_19', upper: 'prChannelUpper400_3_19', lower: 'prChannelLower400_3_19'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 400, degree: 3, deviations: 2.05}, outputMap: {regression: 'prChannel400_3_205', upper: 'prChannelUpper400_3_205', lower: 'prChannelLower400_3_205'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 400, degree: 3, deviations: 2.0}, outputMap: {regression: 'prChannel400_3_20', upper: 'prChannelUpper400_3_20', lower: 'prChannelLower400_3_20'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 400, degree: 3, deviations: 2.15}, outputMap: {regression: 'prChannel400_3_215', upper: 'prChannelUpper400_3_215', lower: 'prChannelLower400_3_215'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 400, degree: 3, deviations: 2.1}, outputMap: {regression: 'prChannel400_3_21', upper: 'prChannelUpper400_3_21', lower: 'prChannelLower400_3_21'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 400, degree: 4, deviations: 1.85}, outputMap: {regression: 'prChannel400_4_185', upper: 'prChannelUpper400_4_185', lower: 'prChannelLower400_4_185'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 400, degree: 4, deviations: 1.95}, outputMap: {regression: 'prChannel400_4_195', upper: 'prChannelUpper400_4_195', lower: 'prChannelLower400_4_195'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 400, degree: 4, deviations: 1.9}, outputMap: {regression: 'prChannel400_4_19', upper: 'prChannelUpper400_4_19', lower: 'prChannelLower400_4_19'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 400, degree: 4, deviations: 2.05}, outputMap: {regression: 'prChannel400_4_205', upper: 'prChannelUpper400_4_205', lower: 'prChannelLower400_4_205'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 400, degree: 4, deviations: 2.0}, outputMap: {regression: 'prChannel400_4_20', upper: 'prChannelUpper400_4_20', lower: 'prChannelLower400_4_20'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 400, degree: 4, deviations: 2.15}, outputMap: {regression: 'prChannel400_4_215', upper: 'prChannelUpper400_4_215', lower: 'prChannelLower400_4_215'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 400, degree: 4, deviations: 2.1}, outputMap: {regression: 'prChannel400_4_21', upper: 'prChannelUpper400_4_21', lower: 'prChannelLower400_4_21'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 400, degree: 5, deviations: 1.85}, outputMap: {regression: 'prChannel400_5_185', upper: 'prChannelUpper400_5_185', lower: 'prChannelLower400_5_185'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 400, degree: 5, deviations: 2.05}, outputMap: {regression: 'prChannel400_5_205', upper: 'prChannelUpper400_5_205', lower: 'prChannelLower400_5_205'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 450, degree: 2, deviations: 1.85}, outputMap: {regression: 'prChannel450_2_185', upper: 'prChannelUpper450_2_185', lower: 'prChannelLower450_2_185'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 450, degree: 2, deviations: 1.95}, outputMap: {regression: 'prChannel450_2_195', upper: 'prChannelUpper450_2_195', lower: 'prChannelLower450_2_195'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 450, degree: 2, deviations: 1.9}, outputMap: {regression: 'prChannel450_2_19', upper: 'prChannelUpper450_2_19', lower: 'prChannelLower450_2_19'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 450, degree: 2, deviations: 2.05}, outputMap: {regression: 'prChannel450_2_205', upper: 'prChannelUpper450_2_205', lower: 'prChannelLower450_2_205'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 450, degree: 2, deviations: 2.0}, outputMap: {regression: 'prChannel450_2_20', upper: 'prChannelUpper450_2_20', lower: 'prChannelLower450_2_20'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 450, degree: 2, deviations: 2.15}, outputMap: {regression: 'prChannel450_2_215', upper: 'prChannelUpper450_2_215', lower: 'prChannelLower450_2_215'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 450, degree: 2, deviations: 2.1}, outputMap: {regression: 'prChannel450_2_21', upper: 'prChannelUpper450_2_21', lower: 'prChannelLower450_2_21'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 450, degree: 3, deviations: 1.85}, outputMap: {regression: 'prChannel450_3_185', upper: 'prChannelUpper450_3_185', lower: 'prChannelLower450_3_185'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 450, degree: 3, deviations: 1.95}, outputMap: {regression: 'prChannel450_3_195', upper: 'prChannelUpper450_3_195', lower: 'prChannelLower450_3_195'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 450, degree: 3, deviations: 1.9}, outputMap: {regression: 'prChannel450_3_19', upper: 'prChannelUpper450_3_19', lower: 'prChannelLower450_3_19'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 450, degree: 3, deviations: 2.05}, outputMap: {regression: 'prChannel450_3_205', upper: 'prChannelUpper450_3_205', lower: 'prChannelLower450_3_205'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 450, degree: 3, deviations: 2.0}, outputMap: {regression: 'prChannel450_3_20', upper: 'prChannelUpper450_3_20', lower: 'prChannelLower450_3_20'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 450, degree: 3, deviations: 2.15}, outputMap: {regression: 'prChannel450_3_215', upper: 'prChannelUpper450_3_215', lower: 'prChannelLower450_3_215'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 450, degree: 3, deviations: 2.1}, outputMap: {regression: 'prChannel450_3_21', upper: 'prChannelUpper450_3_21', lower: 'prChannelLower450_3_21'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 450, degree: 4, deviations: 1.85}, outputMap: {regression: 'prChannel450_4_185', upper: 'prChannelUpper450_4_185', lower: 'prChannelLower450_4_185'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 450, degree: 4, deviations: 1.95}, outputMap: {regression: 'prChannel450_4_195', upper: 'prChannelUpper450_4_195', lower: 'prChannelLower450_4_195'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 450, degree: 4, deviations: 1.9}, outputMap: {regression: 'prChannel450_4_19', upper: 'prChannelUpper450_4_19', lower: 'prChannelLower450_4_19'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 450, degree: 4, deviations: 2.05}, outputMap: {regression: 'prChannel450_4_205', upper: 'prChannelUpper450_4_205', lower: 'prChannelLower450_4_205'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 450, degree: 4, deviations: 2.0}, outputMap: {regression: 'prChannel450_4_20', upper: 'prChannelUpper450_4_20', lower: 'prChannelLower450_4_20'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 450, degree: 4, deviations: 2.15}, outputMap: {regression: 'prChannel450_4_215', upper: 'prChannelUpper450_4_215', lower: 'prChannelLower450_4_215'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 450, degree: 4, deviations: 2.1}, outputMap: {regression: 'prChannel450_4_21', upper: 'prChannelUpper450_4_21', lower: 'prChannelLower450_4_21'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 450, degree: 5, deviations: 1.85}, outputMap: {regression: 'prChannel450_5_185', upper: 'prChannelUpper450_5_185', lower: 'prChannelLower450_5_185'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 450, degree: 5, deviations: 2.05}, outputMap: {regression: 'prChannel450_5_205', upper: 'prChannelUpper450_5_205', lower: 'prChannelLower450_5_205'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 500, degree: 2, deviations: 1.85}, outputMap: {regression: 'prChannel500_2_185', upper: 'prChannelUpper500_2_185', lower: 'prChannelLower500_2_185'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 500, degree: 2, deviations: 1.95}, outputMap: {regression: 'prChannel500_2_195', upper: 'prChannelUpper500_2_195', lower: 'prChannelLower500_2_195'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 500, degree: 2, deviations: 1.9}, outputMap: {regression: 'prChannel500_2_19', upper: 'prChannelUpper500_2_19', lower: 'prChannelLower500_2_19'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 500, degree: 2, deviations: 2.05}, outputMap: {regression: 'prChannel500_2_205', upper: 'prChannelUpper500_2_205', lower: 'prChannelLower500_2_205'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 500, degree: 2, deviations: 2.0}, outputMap: {regression: 'prChannel500_2_20', upper: 'prChannelUpper500_2_20', lower: 'prChannelLower500_2_20'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 500, degree: 2, deviations: 2.15}, outputMap: {regression: 'prChannel500_2_215', upper: 'prChannelUpper500_2_215', lower: 'prChannelLower500_2_215'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 500, degree: 2, deviations: 2.1}, outputMap: {regression: 'prChannel500_2_21', upper: 'prChannelUpper500_2_21', lower: 'prChannelLower500_2_21'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 500, degree: 3, deviations: 1.85}, outputMap: {regression: 'prChannel500_3_185', upper: 'prChannelUpper500_3_185', lower: 'prChannelLower500_3_185'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 500, degree: 3, deviations: 1.95}, outputMap: {regression: 'prChannel500_3_195', upper: 'prChannelUpper500_3_195', lower: 'prChannelLower500_3_195'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 500, degree: 3, deviations: 1.9}, outputMap: {regression: 'prChannel500_3_19', upper: 'prChannelUpper500_3_19', lower: 'prChannelLower500_3_19'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 500, degree: 3, deviations: 2.05}, outputMap: {regression: 'prChannel500_3_205', upper: 'prChannelUpper500_3_205', lower: 'prChannelLower500_3_205'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 500, degree: 3, deviations: 2.0}, outputMap: {regression: 'prChannel500_3_20', upper: 'prChannelUpper500_3_20', lower: 'prChannelLower500_3_20'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 500, degree: 3, deviations: 2.15}, outputMap: {regression: 'prChannel500_3_215', upper: 'prChannelUpper500_3_215', lower: 'prChannelLower500_3_215'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 500, degree: 3, deviations: 2.1}, outputMap: {regression: 'prChannel500_3_21', upper: 'prChannelUpper500_3_21', lower: 'prChannelLower500_3_21'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 500, degree: 4, deviations: 1.85}, outputMap: {regression: 'prChannel500_4_185', upper: 'prChannelUpper500_4_185', lower: 'prChannelLower500_4_185'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 500, degree: 4, deviations: 1.95}, outputMap: {regression: 'prChannel500_4_195', upper: 'prChannelUpper500_4_195', lower: 'prChannelLower500_4_195'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 500, degree: 4, deviations: 1.9}, outputMap: {regression: 'prChannel500_4_19', upper: 'prChannelUpper500_4_19', lower: 'prChannelLower500_4_19'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 500, degree: 4, deviations: 2.05}, outputMap: {regression: 'prChannel500_4_205', upper: 'prChannelUpper500_4_205', lower: 'prChannelLower500_4_205'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 500, degree: 4, deviations: 2.0}, outputMap: {regression: 'prChannel500_4_20', upper: 'prChannelUpper500_4_20', lower: 'prChannelLower500_4_20'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 500, degree: 4, deviations: 2.15}, outputMap: {regression: 'prChannel500_4_215', upper: 'prChannelUpper500_4_215', lower: 'prChannelLower500_4_215'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 500, degree: 4, deviations: 2.1}, outputMap: {regression: 'prChannel500_4_21', upper: 'prChannelUpper500_4_21', lower: 'prChannelLower500_4_21'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 500, degree: 5, deviations: 1.85}, outputMap: {regression: 'prChannel500_5_185', upper: 'prChannelUpper500_5_185', lower: 'prChannelLower500_5_185'}},
    // {study: studies.PolynomialRegressionChannel, inputs: {length: 500, degree: 5, deviations: 2.05}, outputMap: {regression: 'prChannel500_5_205', upper: 'prChannelUpper500_5_205', lower: 'prChannelLower500_5_205'}}
];

function ReversalsCombined(symbol, initialInvestment, configurations) {
    this.constructor = ReversalsCombined;
    Base.call(this, symbol, initialInvestment, configurations);

    this.configurations = configurations;

    this.prepareStudies(studyDefinitions);
}

ReversalsCombined.prototype = Object.create(Base.prototype);

ReversalsCombined.prototype.backtest = function(data, profitability) {
    var self = this;
    var expirationMinutes = 5;
    var putNextTick = false;
    var callNextTick = false;
    var putThisConfiguration = false;
    var callThisConfiguration = false;
    var previousDataPoint;
    var dataPointCount = data.length;
    var previousBalance = 0;
    var previousDay = -1;
    var currentDay = -1;
    var putCount = 0;
    var callCount = 0;

    // For every data point...
    data.forEach(function(dataPoint, index) {
        var position = null;
        var timestampHour = new Date(dataPoint.timestamp).getHours();
        var timestampMinute = new Date(dataPoint.timestamp).getMinutes();

        // Adjust CTOption data to be like MetaTrader data.
        dataPoint.open = dataPoint.open * 0.9997805329;
        dataPoint.high = dataPoint.open * 0.9997805329;
        dataPoint.low = dataPoint.low * 0.9997805329;
        dataPoint.close = dataPoint.close * 0.9997805329;

        currentDay = new Date(dataPoint.timestamp).getDay();

        if (currentDay !== previousDay) {
            self.setInvestment(Math.round(self.profitLoss * 0.005));
        }

        previousDay = currentDay;

        // Simulate the next tick.
        self.tick(dataPoint);

        // Only trade when the profitability is highest (11:30pm - 4pm CST).
        // Note that MetaTrader automatically converts timestamps to the current timezone in exported CSV files.
        if (timestampHour >= 16 && (timestampHour < 23 || (timestampHour === 23 && timestampMinute < 30))) {
            // Track the current data point as the previous data point for the next tick.
            previousDataPoint = dataPoint;

            putNextTick = false;
            callNextTick = false;

            return;
        }

        if (previousDataPoint && index < dataPointCount - 1) {
            if (putNextTick) {
                // Create a new position.
                position = new Put(self.getSymbol(), (dataPoint.timestamp - 1000), previousDataPoint.close, self.getInvestment() * putCount, profitability, expirationMinutes);
                position.setShowTrades(self.getShowTrades());
                self.addPosition(position);
            }

            if (callNextTick) {
                // Create a new position.
                position = new Call(self.getSymbol(), (dataPoint.timestamp - 1000), previousDataPoint.close, self.getInvestment() * callCount, profitability, expirationMinutes)
                position.setShowTrades(self.getShowTrades());
                self.addPosition(position);
            }
        }

        putNextTick = false;
        callNextTick = false;

        putCount = 0;
        callCount = 0;

        // For every configuration...
        self.configurations.forEach(function(configuration) {
            putThisConfiguration = true;
            callThisConfiguration = true;

            if (configuration.ema200 && configuration.ema100) {
                if (!dataPoint.ema200 || !dataPoint.ema100) {
                    putThisConfiguration = false;
                    callThisConfiguration = false;
                }

                // Determine if a downtrend is not occurring.
                if (putThisConfiguration && dataPoint.ema200 < dataPoint.ema100) {
                    putThisConfiguration = false;
                }

                // Determine if an uptrend is not occurring.
                if (callThisConfiguration && dataPoint.ema200 > dataPoint.ema100) {
                    callThisConfiguration = false;
                }
            }
            if (configuration.ema100 && configuration.ema50) {
                if (!dataPoint.ema100 || !dataPoint.ema50) {
                    putThisConfiguration = false;
                    callThisConfiguration = false;
                }

                // Determine if a downtrend is not occurring.
                if (putThisConfiguration && dataPoint.ema100 < dataPoint.ema50) {
                    putThisConfiguration = false;
                }

                // Determine if an uptrend is not occurring.
                if (callThisConfiguration && dataPoint.ema100 > dataPoint.ema50) {
                    callThisConfiguration = false;
                }
            }
            if (configuration.ema50 && configuration.sma13) {
                if (!dataPoint.ema50 || !dataPoint.sma13) {
                    putThisConfiguration = false;
                    callThisConfiguration = false;
                }

                // Determine if a downtrend is not occurring.
                if (putThisConfiguration && dataPoint.ema50 < dataPoint.sma13) {
                    putThisConfiguration = false;
                }

                // Determine if an uptrend is not occurring.
                if (callThisConfiguration && dataPoint.ema50 > dataPoint.sma13) {
                    callThisConfiguration = false;
                }
            }
            if (configuration.rsi) {
                if (typeof dataPoint[configuration.rsi.rsi] === 'number') {
                    // Determine if RSI is not above the overbought line.
                    if (putThisConfiguration && dataPoint[configuration.rsi.rsi] <= configuration.rsi.overbought) {
                        putThisConfiguration = false;
                    }

                    // Determine if RSI is not below the oversold line.
                    if (callThisConfiguration && dataPoint[configuration.rsi.rsi] >= configuration.rsi.oversold) {
                        callThisConfiguration = false;
                    }
                }
                else {
                    putThisConfiguration = false;
                    callThisConfiguration = false;
                }
            }
            if (configuration.stochastic) {
                if (typeof dataPoint[configuration.stochastic.K] === 'number' && typeof dataPoint[configuration.stochastic.D] === 'number') {
                    // Determine if stochastic is not above the overbought line.
                    if (putThisConfiguration && (dataPoint[configuration.stochastic.K] <= configuration.stochastic.overbought || dataPoint[configuration.stochastic.D] <= configuration.stochastic.overbought)) {
                        putThisConfiguration = false;
                    }

                    // Determine if stochastic is not below the oversold line.
                    if (callThisConfiguration && (dataPoint[configuration.stochastic.K] >= configuration.stochastic.oversold || dataPoint[configuration.stochastic.D] >= configuration.stochastic.oversold)) {
                        callThisConfiguration = false;
                    }
                }
                else {
                    putThisConfiguration = false;
                    callThisConfiguration = false;
                }
            }
            if (configuration.prChannel) {
                if (dataPoint[configuration.prChannel.upper] && dataPoint[configuration.prChannel.lower]) {
                    // Determine if the upper regression bound was not breached by the high price.
                    if (putThisConfiguration && (!dataPoint[configuration.prChannel.upper] || dataPoint.high <= dataPoint[configuration.prChannel.upper])) {
                        putThisConfiguration = false;
                    }

                    // Determine if the lower regression bound was not breached by the low price.
                    if (callThisConfiguration && (!dataPoint[configuration.prChannel.lower] || dataPoint.low >= dataPoint[configuration.prChannel.lower])) {
                        callThisConfiguration = false;
                    }
                }
                else {
                    putThisConfiguration = false;
                    callThisConfiguration = false;
                }
            }

            if (putThisConfiguration) {
                putCount++;
            }
            if (callThisConfiguration) {
                callCount++;
            }

            // Determine whether to trade next tick.
            putNextTick = putNextTick || putThisConfiguration;
            callNextTick = callNextTick || callThisConfiguration;
        });

        // Track the current data point as the previous data point for the next tick.
        previousDataPoint = dataPoint;

        if (putNextTick) {
            console.log('PUT for $' + self.getInvestment() + ' at ' + new Date(dataPoint.timestamp + 1000));
        }

        if (callNextTick) {
            console.log('CALL for $' + self.getInvestment() + ' at ' + new Date(dataPoint.timestamp + 1000));
        }
    });

    console.log(self.getResults());
};

module.exports = ReversalsCombined;
