var Base = require('./Base');
var studies = require('../studies');

var studyDefinitions = [
    {study: studies.Ema, inputs: {length: 200}, outputMap: {ema: 'ema200'}},
    {study: studies.Ema, inputs: {length: 100}, outputMap: {ema: 'ema100'}},
    {study: studies.Ema, inputs: {length: 50}, outputMap: {ema: 'ema50'}},
    {study: studies.Sma, inputs: {length: 13}, outputMap: {sma: 'sma13'}},
    {study: studies.Rsi, inputs: {length: 7}, outputMap: {rsi: 'rsi7'}},
    {study: studies.Rsi, inputs: {length: 5}, outputMap: {rsi: 'rsi5'}},
    {study: studies.Rsi, inputs: {length: 2}, outputMap: {rsi: 'rsi2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 50, degree: 2, deviations: 1.9}, outputMap: {regression: 'prChannel50_2_19', upper: 'prChannelUpper50_2_19', lower: 'prChannelLower50_2_19', upper2: 'prChannelUpper50_2_19-2', lower2: 'prChannelLower50_2_19-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 50, degree: 3, deviations: 1.9}, outputMap: {regression: 'prChannel50_3_19', upper: 'prChannelUpper50_3_19', lower: 'prChannelLower50_3_19', upper2: 'prChannelUpper50_3_19-2', lower2: 'prChannelLower50_3_19-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 50, degree: 4, deviations: 1.9}, outputMap: {regression: 'prChannel50_4_19', upper: 'prChannelUpper50_4_19', lower: 'prChannelLower50_4_19', upper2: 'prChannelUpper50_4_19-2', lower2: 'prChannelLower50_4_19-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 50, degree: 2, deviations: 1.95}, outputMap: {regression: 'prChannel50_2_195', upper: 'prChannelUpper50_2_195', lower: 'prChannelLower50_2_195', upper2: 'prChannelUpper50_2_195-2', lower2: 'prChannelLower50_2_195-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 50, degree: 3, deviations: 1.95}, outputMap: {regression: 'prChannel50_3_195', upper: 'prChannelUpper50_3_195', lower: 'prChannelLower50_3_195', upper2: 'prChannelUpper50_3_195-2', lower2: 'prChannelLower50_3_195-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 50, degree: 4, deviations: 1.95}, outputMap: {regression: 'prChannel50_4_195', upper: 'prChannelUpper50_4_195', lower: 'prChannelLower50_4_195', upper2: 'prChannelUpper50_4_195-2', lower2: 'prChannelLower50_4_195-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 50, degree: 2, deviations: 2.0}, outputMap: {regression: 'prChannel50_2_20', upper: 'prChannelUpper50_2_20', lower: 'prChannelLower50_2_20', upper2: 'prChannelUpper50_2_20-2', lower2: 'prChannelLower50_2_20-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 50, degree: 3, deviations: 2.0}, outputMap: {regression: 'prChannel50_3_20', upper: 'prChannelUpper50_3_20', lower: 'prChannelLower50_3_20', upper2: 'prChannelUpper50_3_20-2', lower2: 'prChannelLower50_3_20-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 50, degree: 4, deviations: 2.0}, outputMap: {regression: 'prChannel50_4_20', upper: 'prChannelUpper50_4_20', lower: 'prChannelLower50_4_20', upper2: 'prChannelUpper50_4_20-2', lower2: 'prChannelLower50_4_20-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 50, degree: 2, deviations: 2.1}, outputMap: {regression: 'prChannel50_2_21', upper: 'prChannelUpper50_2_21', lower: 'prChannelLower50_2_21', upper2: 'prChannelUpper50_2_21-2', lower2: 'prChannelLower50_2_21-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 50, degree: 3, deviations: 2.1}, outputMap: {regression: 'prChannel50_3_21', upper: 'prChannelUpper50_3_21', lower: 'prChannelLower50_3_21', upper2: 'prChannelUpper50_3_21-2', lower2: 'prChannelLower50_3_21-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 50, degree: 4, deviations: 2.1}, outputMap: {regression: 'prChannel50_4_21', upper: 'prChannelUpper50_4_21', lower: 'prChannelLower50_4_21', upper2: 'prChannelUpper50_4_21-2', lower2: 'prChannelLower50_4_21-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 50, degree: 2, deviations: 2.15}, outputMap: {regression: 'prChannel50_2_215', upper: 'prChannelUpper50_2_215', lower: 'prChannelLower50_2_215', upper2: 'prChannelUpper50_2_215-2', lower2: 'prChannelLower50_2_215-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 50, degree: 3, deviations: 2.15}, outputMap: {regression: 'prChannel50_3_215', upper: 'prChannelUpper50_3_215', lower: 'prChannelLower50_3_215', upper2: 'prChannelUpper50_3_215-2', lower2: 'prChannelLower50_3_215-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 50, degree: 4, deviations: 2.15}, outputMap: {regression: 'prChannel50_4_215', upper: 'prChannelUpper50_4_215', lower: 'prChannelLower50_4_215', upper2: 'prChannelUpper50_4_215-2', lower2: 'prChannelLower50_4_215-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 100, degree: 2, deviations: 1.9}, outputMap: {regression: 'prChannel100_2_19', upper: 'prChannelUpper100_2_19', lower: 'prChannelLower100_2_19', upper2: 'prChannelUpper100_2_19-2', lower2: 'prChannelLower100_2_19-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 100, degree: 3, deviations: 1.9}, outputMap: {regression: 'prChannel100_3_19', upper: 'prChannelUpper100_3_19', lower: 'prChannelLower100_3_19', upper2: 'prChannelUpper100_3_19-2', lower2: 'prChannelLower100_3_19-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 100, degree: 4, deviations: 1.9}, outputMap: {regression: 'prChannel100_4_19', upper: 'prChannelUpper100_4_19', lower: 'prChannelLower100_4_19', upper2: 'prChannelUpper100_4_19-2', lower2: 'prChannelLower100_4_19-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 100, degree: 2, deviations: 1.95}, outputMap: {regression: 'prChannel100_2_195', upper: 'prChannelUpper100_2_195', lower: 'prChannelLower100_2_195', upper2: 'prChannelUpper100_2_195-2', lower2: 'prChannelLower100_2_195-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 100, degree: 3, deviations: 1.95}, outputMap: {regression: 'prChannel100_3_195', upper: 'prChannelUpper100_3_195', lower: 'prChannelLower100_3_195', upper2: 'prChannelUpper100_3_195-2', lower2: 'prChannelLower100_3_195-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 100, degree: 4, deviations: 1.95}, outputMap: {regression: 'prChannel100_4_195', upper: 'prChannelUpper100_4_195', lower: 'prChannelLower100_4_195', upper2: 'prChannelUpper100_4_195-2', lower2: 'prChannelLower100_4_195-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 100, degree: 2, deviations: 2.0}, outputMap: {regression: 'prChannel100_2_20', upper: 'prChannelUpper100_2_20', lower: 'prChannelLower100_2_20', upper2: 'prChannelUpper100_2_20-2', lower2: 'prChannelLower100_2_20-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 100, degree: 3, deviations: 2.0}, outputMap: {regression: 'prChannel100_3_20', upper: 'prChannelUpper100_3_20', lower: 'prChannelLower100_3_20', upper2: 'prChannelUpper100_3_20-2', lower2: 'prChannelLower100_3_20-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 100, degree: 4, deviations: 2.0}, outputMap: {regression: 'prChannel100_4_20', upper: 'prChannelUpper100_4_20', lower: 'prChannelLower100_4_20', upper2: 'prChannelUpper100_4_20-2', lower2: 'prChannelLower100_4_20-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 100, degree: 2, deviations: 2.1}, outputMap: {regression: 'prChannel100_2_21', upper: 'prChannelUpper100_2_21', lower: 'prChannelLower100_2_21', upper2: 'prChannelUpper100_2_21-2', lower2: 'prChannelLower100_2_21-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 100, degree: 3, deviations: 2.1}, outputMap: {regression: 'prChannel100_3_21', upper: 'prChannelUpper100_3_21', lower: 'prChannelLower100_3_21', upper2: 'prChannelUpper100_3_21-2', lower2: 'prChannelLower100_3_21-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 100, degree: 4, deviations: 2.1}, outputMap: {regression: 'prChannel100_4_21', upper: 'prChannelUpper100_4_21', lower: 'prChannelLower100_4_21', upper2: 'prChannelUpper100_4_21-2', lower2: 'prChannelLower100_4_21-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 100, degree: 2, deviations: 2.15}, outputMap: {regression: 'prChannel100_2_215', upper: 'prChannelUpper100_2_215', lower: 'prChannelLower100_2_215', upper2: 'prChannelUpper100_2_215-2', lower2: 'prChannelLower100_2_215-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 100, degree: 3, deviations: 2.15}, outputMap: {regression: 'prChannel100_3_215', upper: 'prChannelUpper100_3_215', lower: 'prChannelLower100_3_215', upper2: 'prChannelUpper100_3_215-2', lower2: 'prChannelLower100_3_215-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 100, degree: 4, deviations: 2.15}, outputMap: {regression: 'prChannel100_4_215', upper: 'prChannelUpper100_4_215', lower: 'prChannelLower100_4_215', upper2: 'prChannelUpper100_4_215-2', lower2: 'prChannelLower100_4_215-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 200, degree: 2, deviations: 2.15}, outputMap: {regression: 'prChannel200_2_215', upper: 'prChannelUpper200_2_215', lower: 'prChannelLower200_2_215', upper2: 'prChannelUpper200_2_215-2', lower2: 'prChannelLower200_2_215-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 200, degree: 3, deviations: 2.15}, outputMap: {regression: 'prChannel200_3_215', upper: 'prChannelUpper200_3_215', lower: 'prChannelLower200_3_215', upper2: 'prChannelUpper200_3_215-2', lower2: 'prChannelLower200_3_215-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 200, degree: 4, deviations: 2.15}, outputMap: {regression: 'prChannel200_4_215', upper: 'prChannelUpper200_4_215', lower: 'prChannelLower200_4_215', upper2: 'prChannelUpper200_4_215-2', lower2: 'prChannelLower200_4_215-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 200, degree: 2, deviations: 1.9}, outputMap: {regression: 'prChannel200_2_19', upper: 'prChannelUpper200_2_19', lower: 'prChannelLower200_2_19', upper2: 'prChannelUpper200_2_19-2', lower2: 'prChannelLower200_2_19-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 200, degree: 3, deviations: 1.9}, outputMap: {regression: 'prChannel200_3_19', upper: 'prChannelUpper200_3_19', lower: 'prChannelLower200_3_19', upper2: 'prChannelUpper200_3_19-2', lower2: 'prChannelLower200_3_19-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 200, degree: 4, deviations: 1.9}, outputMap: {regression: 'prChannel200_4_19', upper: 'prChannelUpper200_4_19', lower: 'prChannelLower200_4_19', upper2: 'prChannelUpper200_4_19-2', lower2: 'prChannelLower200_4_19-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 200, degree: 2, deviations: 1.95}, outputMap: {regression: 'prChannel200_2_195', upper: 'prChannelUpper200_2_195', lower: 'prChannelLower200_2_195', upper2: 'prChannelUpper200_2_195-2', lower2: 'prChannelLower200_2_195-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 200, degree: 3, deviations: 1.95}, outputMap: {regression: 'prChannel200_3_195', upper: 'prChannelUpper200_3_195', lower: 'prChannelLower200_3_195', upper2: 'prChannelUpper200_3_195-2', lower2: 'prChannelLower200_3_195-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 200, degree: 4, deviations: 1.95}, outputMap: {regression: 'prChannel200_4_195', upper: 'prChannelUpper200_4_195', lower: 'prChannelLower200_4_195', upper2: 'prChannelUpper200_4_195-2', lower2: 'prChannelLower200_4_195-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 200, degree: 2, deviations: 2.0}, outputMap: {regression: 'prChannel200_2_20', upper: 'prChannelUpper200_2_20', lower: 'prChannelLower200_2_20', upper2: 'prChannelUpper200_2_20-2', lower2: 'prChannelLower200_2_20-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 200, degree: 3, deviations: 2.0}, outputMap: {regression: 'prChannel200_3_20', upper: 'prChannelUpper200_3_20', lower: 'prChannelLower200_3_20', upper2: 'prChannelUpper200_3_20-2', lower2: 'prChannelLower200_3_20-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 200, degree: 4, deviations: 2.0}, outputMap: {regression: 'prChannel200_4_20', upper: 'prChannelUpper200_4_20', lower: 'prChannelLower200_4_20', upper2: 'prChannelUpper200_4_20-2', lower2: 'prChannelLower200_4_20-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 200, degree: 2, deviations: 2.1}, outputMap: {regression: 'prChannel200_2_21', upper: 'prChannelUpper200_2_21', lower: 'prChannelLower200_2_21', upper2: 'prChannelUpper200_2_21-2', lower2: 'prChannelLower200_2_21-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 200, degree: 3, deviations: 2.1}, outputMap: {regression: 'prChannel200_3_21', upper: 'prChannelUpper200_3_21', lower: 'prChannelLower200_3_21', upper2: 'prChannelUpper200_3_21-2', lower2: 'prChannelLower200_3_21-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 200, degree: 4, deviations: 2.1}, outputMap: {regression: 'prChannel200_4_21', upper: 'prChannelUpper200_4_21', lower: 'prChannelLower200_4_21', upper2: 'prChannelUpper200_4_21-2', lower2: 'prChannelLower200_4_21-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 250, degree: 2, deviations: 1.9}, outputMap: {regression: 'prChannel250_2_19', upper: 'prChannelUpper250_2_19', lower: 'prChannelLower250_2_19', upper2: 'prChannelUpper250_2_19-2', lower2: 'prChannelLower250_2_19-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 250, degree: 3, deviations: 1.9}, outputMap: {regression: 'prChannel250_3_19', upper: 'prChannelUpper250_3_19', lower: 'prChannelLower250_3_19', upper2: 'prChannelUpper250_3_19-2', lower2: 'prChannelLower250_3_19-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 250, degree: 4, deviations: 1.9}, outputMap: {regression: 'prChannel250_4_19', upper: 'prChannelUpper250_4_19', lower: 'prChannelLower250_4_19', upper2: 'prChannelUpper250_4_19-2', lower2: 'prChannelLower250_4_19-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 250, degree: 2, deviations: 1.95}, outputMap: {regression: 'prChannel250_2_195', upper: 'prChannelUpper250_2_195', lower: 'prChannelLower250_2_195', upper2: 'prChannelUpper250_2_195-2', lower2: 'prChannelLower250_2_195-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 250, degree: 3, deviations: 1.95}, outputMap: {regression: 'prChannel250_3_195', upper: 'prChannelUpper250_3_195', lower: 'prChannelLower250_3_195', upper2: 'prChannelUpper250_3_195-2', lower2: 'prChannelLower250_3_195-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 250, degree: 4, deviations: 1.95}, outputMap: {regression: 'prChannel250_4_195', upper: 'prChannelUpper250_4_195', lower: 'prChannelLower250_4_195', upper2: 'prChannelUpper250_4_195-2', lower2: 'prChannelLower250_4_195-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 250, degree: 2, deviations: 2.0}, outputMap: {regression: 'prChannel250_2_20', upper: 'prChannelUpper250_2_20', lower: 'prChannelLower250_2_20', upper2: 'prChannelUpper250_2_20-2', lower2: 'prChannelLower250_2_20-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 250, degree: 3, deviations: 2.0}, outputMap: {regression: 'prChannel250_3_20', upper: 'prChannelUpper250_3_20', lower: 'prChannelLower250_3_20', upper2: 'prChannelUpper250_3_20-2', lower2: 'prChannelLower250_3_20-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 250, degree: 4, deviations: 2.0}, outputMap: {regression: 'prChannel250_4_20', upper: 'prChannelUpper250_4_20', lower: 'prChannelLower250_4_20', upper2: 'prChannelUpper250_4_20-2', lower2: 'prChannelLower250_4_20-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 250, degree: 2, deviations: 2.1}, outputMap: {regression: 'prChannel250_2_21', upper: 'prChannelUpper250_2_21', lower: 'prChannelLower250_2_21', upper2: 'prChannelUpper250_2_21-2', lower2: 'prChannelLower250_2_21-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 250, degree: 3, deviations: 2.1}, outputMap: {regression: 'prChannel250_3_21', upper: 'prChannelUpper250_3_21', lower: 'prChannelLower250_3_21', upper2: 'prChannelUpper250_3_21-2', lower2: 'prChannelLower250_3_21-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 250, degree: 4, deviations: 2.1}, outputMap: {regression: 'prChannel250_4_21', upper: 'prChannelUpper250_4_21', lower: 'prChannelLower250_4_21', upper2: 'prChannelUpper250_4_21-2', lower2: 'prChannelLower250_4_21-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 250, degree: 2, deviations: 2.15}, outputMap: {regression: 'prChannel250_2_215', upper: 'prChannelUpper250_2_215', lower: 'prChannelLower250_2_215', upper2: 'prChannelUpper250_2_215-2', lower2: 'prChannelLower250_2_215-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 250, degree: 3, deviations: 2.15}, outputMap: {regression: 'prChannel250_3_215', upper: 'prChannelUpper250_3_215', lower: 'prChannelLower250_3_215', upper2: 'prChannelUpper250_3_215-2', lower2: 'prChannelLower250_3_215-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 250, degree: 4, deviations: 2.15}, outputMap: {regression: 'prChannel250_4_215', upper: 'prChannelUpper250_4_215', lower: 'prChannelLower250_4_215', upper2: 'prChannelUpper250_4_215-2', lower2: 'prChannelLower250_4_215-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 300, degree: 2, deviations: 2.15}, outputMap: {regression: 'prChannel300_2_215', upper: 'prChannelUpper300_2_215', lower: 'prChannelLower300_2_215', upper2: 'prChannelUpper300_2_215-2', lower2: 'prChannelLower300_2_215-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 300, degree: 3, deviations: 2.15}, outputMap: {regression: 'prChannel300_3_215', upper: 'prChannelUpper300_3_215', lower: 'prChannelLower300_3_215', upper2: 'prChannelUpper300_3_215-2', lower2: 'prChannelLower300_3_215-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 300, degree: 4, deviations: 2.15}, outputMap: {regression: 'prChannel300_4_215', upper: 'prChannelUpper300_4_215', lower: 'prChannelLower300_4_215', upper2: 'prChannelUpper300_4_215-2', lower2: 'prChannelLower300_4_215-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 300, degree: 2, deviations: 1.9}, outputMap: {regression: 'prChannel300_2_19', upper: 'prChannelUpper300_2_19', lower: 'prChannelLower300_2_19', upper2: 'prChannelUpper300_2_19-2', lower2: 'prChannelLower300_2_19-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 300, degree: 3, deviations: 1.9}, outputMap: {regression: 'prChannel300_3_19', upper: 'prChannelUpper300_3_19', lower: 'prChannelLower300_3_19', upper2: 'prChannelUpper300_3_19-2', lower2: 'prChannelLower300_3_19-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 300, degree: 4, deviations: 1.9}, outputMap: {regression: 'prChannel300_4_19', upper: 'prChannelUpper300_4_19', lower: 'prChannelLower300_4_19', upper2: 'prChannelUpper300_4_19-2', lower2: 'prChannelLower300_4_19-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 300, degree: 2, deviations: 1.95}, outputMap: {regression: 'prChannel300_2_195', upper: 'prChannelUpper300_2_195', lower: 'prChannelLower300_2_195', upper2: 'prChannelUpper300_2_195-2', lower2: 'prChannelLower300_2_195-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 300, degree: 3, deviations: 1.95}, outputMap: {regression: 'prChannel300_3_195', upper: 'prChannelUpper300_3_195', lower: 'prChannelLower300_3_195', upper2: 'prChannelUpper300_3_195-2', lower2: 'prChannelLower300_3_195-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 300, degree: 4, deviations: 1.95}, outputMap: {regression: 'prChannel300_4_195', upper: 'prChannelUpper300_4_195', lower: 'prChannelLower300_4_195', upper2: 'prChannelUpper300_4_195-2', lower2: 'prChannelLower300_4_195-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 300, degree: 2, deviations: 2.0}, outputMap: {regression: 'prChannel300_2_20', upper: 'prChannelUpper300_2_20', lower: 'prChannelLower300_2_20', upper2: 'prChannelUpper300_2_20-2', lower2: 'prChannelLower300_2_20-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 300, degree: 3, deviations: 2.0}, outputMap: {regression: 'prChannel300_3_20', upper: 'prChannelUpper300_3_20', lower: 'prChannelLower300_3_20', upper2: 'prChannelUpper300_3_20-2', lower2: 'prChannelLower300_3_20-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 300, degree: 4, deviations: 2.0}, outputMap: {regression: 'prChannel300_4_20', upper: 'prChannelUpper300_4_20', lower: 'prChannelLower300_4_20', upper2: 'prChannelUpper300_4_20-2', lower2: 'prChannelLower300_4_20-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 300, degree: 2, deviations: 2.1}, outputMap: {regression: 'prChannel300_2_21', upper: 'prChannelUpper300_2_21', lower: 'prChannelLower300_2_21', upper2: 'prChannelUpper300_2_21-2', lower2: 'prChannelLower300_2_21-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 300, degree: 3, deviations: 2.1}, outputMap: {regression: 'prChannel300_3_21', upper: 'prChannelUpper300_3_21', lower: 'prChannelLower300_3_21', upper2: 'prChannelUpper300_3_21-2', lower2: 'prChannelLower300_3_21-2'}},
    {study: studies.PolynomialRegressionChannel, inputs: {length: 300, degree: 4, deviations: 2.1}, outputMap: {regression: 'prChannel300_4_21', upper: 'prChannelUpper300_4_21', lower: 'prChannelLower300_4_21', upper2: 'prChannelUpper300_4_21-2', lower2: 'prChannelLower300_4_21-2'}}
];

var configurationOptions = {
    ema200: [true, false],
    ema100: [true, false],
    ema50: [true, false],
    sma13: [true, false],
    rsi: [
        null,
        {rsi: 'rsi7', overbought: 77, oversold: 23},
        {rsi: 'rsi7', overbought: 80, oversold: 20},
        {rsi: 'rsi5', overbought: 80, oversold: 20},
        {rsi: 'rsi2', overbought: 95, oversold: 5}
    ],
    prChannel: [
        null,
        {upper: 'prChannelUpper50_2_19', lower: 'prChannelLower50_2_19', upper2: 'prChannelUpper50_2_19-2', lower2: 'prChannelLower50_2_19-2'},
        {upper: 'prChannelUpper50_3_19', lower: 'prChannelLower50_3_19', upper2: 'prChannelUpper50_3_19-2', lower2: 'prChannelLower50_3_19-2'},
        {upper: 'prChannelUpper50_4_19', lower: 'prChannelLower50_4_19', upper2: 'prChannelUpper50_4_19-2', lower2: 'prChannelLower50_4_19-2'},
        {upper: 'prChannelUpper50_2_195', lower: 'prChannelLower50_2_195', upper2: 'prChannelUpper50_2_195-2', lower2: 'prChannelLower50_2_195-2'},
        {upper: 'prChannelUpper50_3_195', lower: 'prChannelLower50_3_195', upper2: 'prChannelUpper50_3_195-2', lower2: 'prChannelLower50_3_195-2'},
        {upper: 'prChannelUpper50_4_195', lower: 'prChannelLower50_4_195', upper2: 'prChannelUpper50_4_195-2', lower2: 'prChannelLower50_4_195-2'},
        {upper: 'prChannelUpper50_2_20', lower: 'prChannelLower50_2_20', upper2: 'prChannelUpper50_2_20-2', lower2: 'prChannelLower50_2_20-2'},
        {upper: 'prChannelUpper50_3_20', lower: 'prChannelLower50_3_20', upper2: 'prChannelUpper50_3_20-2', lower2: 'prChannelLower50_3_20-2'},
        {upper: 'prChannelUpper50_4_20', lower: 'prChannelLower50_4_20', upper2: 'prChannelUpper50_4_20-2', lower2: 'prChannelLower50_4_20-2'},
        {upper: 'prChannelUpper50_2_21', lower: 'prChannelLower50_2_21', upper2: 'prChannelUpper50_2_21-2', lower2: 'prChannelLower50_2_21-2'},
        {upper: 'prChannelUpper50_3_21', lower: 'prChannelLower50_3_21', upper2: 'prChannelUpper50_3_21-2', lower2: 'prChannelLower50_3_21-2'},
        {upper: 'prChannelUpper50_4_21', lower: 'prChannelLower50_4_21', upper2: 'prChannelUpper50_4_21-2', lower2: 'prChannelLower50_4_21-2'},
        {upper: 'prChannelUpper50_2_215', lower: 'prChannelLower50_2_215', upper2: 'prChannelUpper50_2_215-2', lower2: 'prChannelLower50_2_215-2'},
        {upper: 'prChannelUpper50_3_215', lower: 'prChannelLower50_3_215', upper2: 'prChannelUpper50_3_215-2', lower2: 'prChannelLower50_3_215-2'},
        {upper: 'prChannelUpper100_2_19', lower: 'prChannelLower100_2_19', upper2: 'prChannelUpper100_2_19-2', lower2: 'prChannelLower100_2_19-2'},
        {upper: 'prChannelUpper100_3_19', lower: 'prChannelLower100_3_19', upper2: 'prChannelUpper100_3_19-2', lower2: 'prChannelLower100_3_19-2'},
        {upper: 'prChannelUpper100_4_19', lower: 'prChannelLower100_4_19', upper2: 'prChannelUpper100_4_19-2', lower2: 'prChannelLower100_4_19-2'},
        {upper: 'prChannelUpper100_2_195', lower: 'prChannelLower100_2_195', upper2: 'prChannelUpper100_2_195-2', lower2: 'prChannelLower100_2_195-2'},
        {upper: 'prChannelUpper100_3_195', lower: 'prChannelLower100_3_195', upper2: 'prChannelUpper100_3_195-2', lower2: 'prChannelLower100_3_195-2'},
        {upper: 'prChannelUpper100_4_195', lower: 'prChannelLower100_4_195', upper2: 'prChannelUpper100_4_195-2', lower2: 'prChannelLower100_4_195-2'},
        {upper: 'prChannelUpper100_2_20', lower: 'prChannelLower100_2_20', upper2: 'prChannelUpper100_2_20-2', lower2: 'prChannelLower100_2_20-2'},
        {upper: 'prChannelUpper100_3_20', lower: 'prChannelLower100_3_20', upper2: 'prChannelUpper100_3_20-2', lower2: 'prChannelLower100_3_20-2'},
        {upper: 'prChannelUpper100_4_20', lower: 'prChannelLower100_4_20', upper2: 'prChannelUpper100_4_20-2', lower2: 'prChannelLower100_4_20-2'},
        {upper: 'prChannelUpper100_2_21', lower: 'prChannelLower100_2_21', upper2: 'prChannelUpper100_2_21-2', lower2: 'prChannelLower100_2_21-2'},
        {upper: 'prChannelUpper100_3_21', lower: 'prChannelLower100_3_21', upper2: 'prChannelUpper100_3_21-2', lower2: 'prChannelLower100_3_21-2'},
        {upper: 'prChannelUpper100_4_21', lower: 'prChannelLower100_4_21', upper2: 'prChannelUpper100_4_21-2', lower2: 'prChannelLower100_4_21-2'},
        {upper: 'prChannelUpper100_2_215', lower: 'prChannelLower100_2_215', upper2: 'prChannelUpper100_2_215-2', lower2: 'prChannelLower100_2_215-2'},
        {upper: 'prChannelUpper100_3_215', lower: 'prChannelLower100_3_215', upper2: 'prChannelUpper100_3_215-2', lower2: 'prChannelLower100_3_215-2'},
        {upper: 'prChannelUpper100_4_215', lower: 'prChannelLower100_4_215', upper2: 'prChannelUpper100_4_215-2', lower2: 'prChannelLower100_4_215-2'},
        {upper: 'prChannelUpper200_2_215', lower: 'prChannelLower200_2_215', upper2: 'prChannelUpper200_2_215-2', lower2: 'prChannelLower200_2_215-2'},
        {upper: 'prChannelUpper200_3_215', lower: 'prChannelLower200_3_215', upper2: 'prChannelUpper200_3_215-2', lower2: 'prChannelLower200_3_215-2'},
        {upper: 'prChannelUpper200_4_215', lower: 'prChannelLower200_4_215', upper2: 'prChannelUpper200_4_215-2', lower2: 'prChannelLower200_4_215-2'},
        {upper: 'prChannelUpper200_2_19', lower: 'prChannelLower200_2_19', upper2: 'prChannelUpper200_2_19-2', lower2: 'prChannelLower200_2_19-2'},
        {upper: 'prChannelUpper200_3_19', lower: 'prChannelLower200_3_19', upper2: 'prChannelUpper200_3_19-2', lower2: 'prChannelLower200_3_19-2'},
        {upper: 'prChannelUpper200_4_19', lower: 'prChannelLower200_4_19', upper2: 'prChannelUpper200_4_19-2', lower2: 'prChannelLower200_4_19-2'},
        {upper: 'prChannelUpper200_2_195', lower: 'prChannelLower200_2_195', upper2: 'prChannelUpper200_2_195-2', lower2: 'prChannelLower200_2_195-2'},
        {upper: 'prChannelUpper200_3_195', lower: 'prChannelLower200_3_195', upper2: 'prChannelUpper200_3_195-2', lower2: 'prChannelLower200_3_195-2'},
        {upper: 'prChannelUpper200_4_195', lower: 'prChannelLower200_4_195', upper2: 'prChannelUpper200_4_195-2', lower2: 'prChannelLower200_4_195-2'},
        {upper: 'prChannelUpper200_2_20', lower: 'prChannelLower200_2_20', upper2: 'prChannelUpper200_2_20-2', lower2: 'prChannelLower200_2_20-2'},
        {upper: 'prChannelUpper200_3_20', lower: 'prChannelLower200_3_20', upper2: 'prChannelUpper200_3_20-2', lower2: 'prChannelLower200_3_20-2'},
        {upper: 'prChannelUpper200_4_20', lower: 'prChannelLower200_4_20', upper2: 'prChannelUpper200_4_20-2', lower2: 'prChannelLower200_4_20-2'},
        {upper: 'prChannelUpper200_2_21', lower: 'prChannelLower200_2_21', upper2: 'prChannelUpper200_2_21-2', lower2: 'prChannelLower200_2_21-2'},
        {upper: 'prChannelUpper200_3_21', lower: 'prChannelLower200_3_21', upper2: 'prChannelUpper200_3_21-2', lower2: 'prChannelLower200_3_21-2'},
        {upper: 'prChannelUpper200_4_21', lower: 'prChannelLower200_4_21', upper2: 'prChannelUpper200_4_21-2', lower2: 'prChannelLower200_4_21-2'},
        {upper: 'prChannelUpper250_2_19', lower: 'prChannelLower250_2_19', upper2: 'prChannelUpper250_2_19-2', lower2: 'prChannelLower250_2_19-2'},
        {upper: 'prChannelUpper250_3_19', lower: 'prChannelLower250_3_19', upper2: 'prChannelUpper250_3_19-2', lower2: 'prChannelLower250_3_19-2'},
        {upper: 'prChannelUpper250_4_19', lower: 'prChannelLower250_4_19', upper2: 'prChannelUpper250_4_19-2', lower2: 'prChannelLower250_4_19-2'},
        {upper: 'prChannelUpper250_2_195', lower: 'prChannelLower250_2_195', upper2: 'prChannelUpper250_2_195-2', lower2: 'prChannelLower250_2_195-2'},
        {upper: 'prChannelUpper250_3_195', lower: 'prChannelLower250_3_195', upper2: 'prChannelUpper250_3_195-2', lower2: 'prChannelLower250_3_195-2'},
        {upper: 'prChannelUpper250_4_195', lower: 'prChannelLower250_4_195', upper2: 'prChannelUpper250_4_195-2', lower2: 'prChannelLower250_4_195-2'},
        {upper: 'prChannelUpper250_2_20', lower: 'prChannelLower250_2_20', upper2: 'prChannelUpper250_2_20-2', lower2: 'prChannelLower250_2_20-2'},
        {upper: 'prChannelUpper250_3_20', lower: 'prChannelLower250_3_20', upper2: 'prChannelUpper250_3_20-2', lower2: 'prChannelLower250_3_20-2'},
        {upper: 'prChannelUpper250_4_20', lower: 'prChannelLower250_4_20', upper2: 'prChannelUpper250_4_20-2', lower2: 'prChannelLower250_4_20-2'},
        {upper: 'prChannelUpper250_2_21', lower: 'prChannelLower250_2_21', upper2: 'prChannelUpper250_2_21-2', lower2: 'prChannelLower250_2_21-2'},
        {upper: 'prChannelUpper250_3_21', lower: 'prChannelLower250_3_21', upper2: 'prChannelUpper250_3_21-2', lower2: 'prChannelLower250_3_21-2'},
        {upper: 'prChannelUpper250_4_21', lower: 'prChannelLower250_4_21', upper2: 'prChannelUpper250_4_21-2', lower2: 'prChannelLower250_4_21-2'},
        {upper: 'prChannelUpper250_2_215', lower: 'prChannelLower250_2_215', upper2: 'prChannelUpper250_2_215-2', lower2: 'prChannelLower250_2_215-2'},
        {upper: 'prChannelUpper250_3_215', lower: 'prChannelLower250_3_215', upper2: 'prChannelUpper250_3_215-2', lower2: 'prChannelLower250_3_215-2'},
        {upper: 'prChannelUpper250_4_215', lower: 'prChannelLower250_4_215', upper2: 'prChannelUpper250_4_215-2', lower2: 'prChannelLower250_4_215-2'},
        {upper: 'prChannelUpper300_2_215', lower: 'prChannelLower300_2_215', upper2: 'prChannelUpper300_2_215-2', lower2: 'prChannelLower300_2_215-2'},
        {upper: 'prChannelUpper300_3_215', lower: 'prChannelLower300_3_215', upper2: 'prChannelUpper300_3_215-2', lower2: 'prChannelLower300_3_215-2'},
        {upper: 'prChannelUpper300_4_215', lower: 'prChannelLower300_4_215', upper2: 'prChannelUpper300_4_215-2', lower2: 'prChannelLower300_4_215-2'},
        {upper: 'prChannelUpper300_2_19', lower: 'prChannelLower300_2_19', upper2: 'prChannelUpper300_2_19-2', lower2: 'prChannelLower300_2_19-2'},
        {upper: 'prChannelUpper300_3_19', lower: 'prChannelLower300_3_19', upper2: 'prChannelUpper300_3_19-2', lower2: 'prChannelLower300_3_19-2'},
        {upper: 'prChannelUpper300_4_19', lower: 'prChannelLower300_4_19', upper2: 'prChannelUpper300_4_19-2', lower2: 'prChannelLower300_4_19-2'},
        {upper: 'prChannelUpper300_2_195', lower: 'prChannelLower300_2_195', upper2: 'prChannelUpper300_2_195-2', lower2: 'prChannelLower300_2_195-2'},
        {upper: 'prChannelUpper300_3_195', lower: 'prChannelLower300_3_195', upper2: 'prChannelUpper300_3_195-2', lower2: 'prChannelLower300_3_195-2'},
        {upper: 'prChannelUpper300_4_195', lower: 'prChannelLower300_4_195', upper2: 'prChannelUpper300_4_195-2', lower2: 'prChannelLower300_4_195-2'},
        {upper: 'prChannelUpper300_2_20', lower: 'prChannelLower300_2_20', upper2: 'prChannelUpper300_2_20-2', lower2: 'prChannelLower300_2_20-2'},
        {upper: 'prChannelUpper300_3_20', lower: 'prChannelLower300_3_20', upper2: 'prChannelUpper300_3_20-2', lower2: 'prChannelLower300_3_20-2'},
        {upper: 'prChannelUpper300_4_20', lower: 'prChannelLower300_4_20', upper2: 'prChannelUpper300_4_20-2', lower2: 'prChannelLower300_4_20-2'},
        {upper: 'prChannelUpper300_2_21', lower: 'prChannelLower300_2_21', upper2: 'prChannelUpper300_2_21-2', lower2: 'prChannelLower300_2_21-2'},
        {upper: 'prChannelUpper300_3_21', lower: 'prChannelLower300_3_21', upper2: 'prChannelUpper300_3_21-2', lower2: 'prChannelLower300_3_21-2'},
        {upper: 'prChannelUpper300_4_21', lower: 'prChannelLower300_4_21', upper2: 'prChannelUpper300_4_21-2', lower2: 'prChannelLower300_4_21-2'}
    ]
};

function Reversals(symbol) {
    this.constructor = Reversals;
    Base.call(this, 'Reversals', symbol);

    // Prepare studies for use.
    this.prepareStudies(studyDefinitions);

    // Prepare all optimization configurations.
    this.configurations = this.buildConfigurations(configurationOptions);
}

// Create a copy of the Base "class" prototype for use in this "class."
Reversals.prototype = Object.create(Base.prototype);

Reversals.prototype.optimize = function(data, investment, profitability, done) {
    var self = this;

    // Prepare all data in advance for use.
    self.prepareStudyData(data, function() {
        // Ensure memory is released.
        data = null;

        Base.prototype.optimize.call(self, self.configurations, investment, profitability, done);
    });
};

module.exports = Reversals;
