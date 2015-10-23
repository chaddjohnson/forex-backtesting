module.exports.test = function(positions) {
    var profitLoss = 0;
    var winCount = 0;
    var loseCount = 0;
    var winRate = 0;
    var maximumConsecutiveLosses = 0;
    var minimumProfitLoss = 99999;
    var consecutiveLosses = 0;

    positions.forEach(function(position) {
        var positionProfitLoss = position.profitLoss;

        profitLoss -= position.investment;
        profitLoss += positionProfitLoss;

        if (positionProfitLoss > position.investment) {
            winCount++;
            consecutiveLosses = 0;
        }
        if (positionProfitLoss === 0) {
            loseCount++;
            consecutiveLosses++;
        }

        // Track minimum profit/loss.
        if (positionProfitLoss < minimumProfitLoss) {
            minimumProfitLoss = positionProfitLoss;
        }

        // Track the maximum consecutive losses.
        if (consecutiveLosses > maximumConsecutiveLosses) {
            maximumConsecutiveLosses = consecutiveLosses;
        }
    });

    if (winCount + loseCount === 0) {
        winRate = 0;
    }
    else {
        winRate = winCount / (winCount + loseCount);
    }

    return {
        profitLoss: profitLoss,
        winCount: winCount,
        loseCount: loseCount,
        winRate: winRate,
        tradeCount: winCount + loseCount,
        maximumConsecutiveLosses: maximumConsecutiveLosses,
        minimumProfitLoss: minimumProfitLoss
    };
};
