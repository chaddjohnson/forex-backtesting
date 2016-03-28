var Base = require('./Base');
var _ = require('lodash');

function AverageDirectionalIndex(inputs, outputMap) {
    this.constructor = AverageDirectionalIndex;
    Base.call(this, inputs, outputMap);

    this.tickIndex = 0;

    this.pastValues = {};
    this.pastValues.TR = [];
    this.pastValues.ADX = 0.0;
    this.pastValues.TR2 = 0.0;
    this.pastValues.pDM = [];
    this.pastValues.mDM = [];
    this.pastValues.pDM2 = 0.0;
    this.pastValues.mDM2 = 0.0;
    this.pastValues.DX = [];

    if (!inputs.length) {
        throw 'No length input parameter provided to study.';
    }
}

// Create a copy of the Base "class" prototype for use in this "class."
AverageDirectionalIndex.prototype = Object.create(Base.prototype);

AverageDirectionalIndex.prototype.tick = function() {
    var dataSegment = this.getDataSegment(this.getInput('length'));
    var dataSegmentLength = dataSegment.length;
    var returnValue = {};
    var previousDataPoint = this.getPrevious();
    var lastDataPoint = this.getLast();
    var TR = 0.0;
    var TR2 = 0.0;
    var pDM = 0.0;
    var mDM = 0.0;
    var pDM2 = 0.0;
    var mDM2 = 0.0;
    var pDI = 0.0;
    var mDI = 0.0;
    var DIDiff = 0.0;
    var DISum = 0.0;
    var DX = 0.0;
    var ADX = 0.0;

    if (!previousDataPoint) {
        this.tickIndex = 0;
    }

    this.tickIndex++;

    if (this.tickIndex < 2) {
        return returnValue;
    }

    if (this.tickIndex > 1) {
        TR = Math.max(lastDataPoint.high - lastDataPoint.low, Math.abs(lastDataPoint.high - previousDataPoint.close), Math.abs(lastDataPoint.low - previousDataPoint.close));

        if (lastDataPoint.high - previousDataPoint.high > previousDataPoint.low - lastDataPoint.low) {
            pDM = Math.max(lastDataPoint.high - previousDataPoint.high, 0);
        }
        else {
            pDM = 0;
        }

        if (previousDataPoint.low - lastDataPoint.low > lastDataPoint.high - previousDataPoint.high) {
            mDM = Math.max(previousDataPoint.low - lastDataPoint.low, 0);
        }
        else {
            mDM = 0;
        }

        this.pastValues.TR.push(TR);
        this.pastValues.pDM.push(pDM);
        this.pastValues.mDM.push(mDM);
    }

    if (this.tickIndex > this.getInput('length')) {
        if (this.tickIndex === this.getInput('length') + 1) {
            TR2 = _.reduce(this.pastValues.TR, function(memo, pastTR) {
                return memo + pastTR;
            }, 0);
            pDM2 = _.reduce(this.pastValues.pDM, function(memo, pastPDM) {
                return memo + pastPDM;
            }, 0);
            mDM2 = _.reduce(this.pastValues.mDM, function(memo, pastMDM) {
                return memo + pastMDM;
            }, 0);
        }
        else {
            TR2 = this.pastValues.TR2 - (this.pastValues.TR2 / this.getInput('length')) + TR;
            pDM2 = this.pastValues.pDM2 - (this.pastValues.pDM2 / this.getInput('length')) + pDM;
            mDM2 = this.pastValues.mDM2 - (this.pastValues.mDM2 / this.getInput('length')) + mDM;
        }

        pDI = 100 * (pDM2 / TR2);
        mDI = 100 * (mDM2 / TR2);
        DIDiff = Math.abs(pDI - mDI);
        DISum = pDI + mDI;
        DX = 100 * (DIDiff / DISum);

        this.pastValues.TR2 = TR2;
        this.pastValues.pDM2 = pDM2;
        this.pastValues.mDM2 = mDM2;
        this.pastValues.DX.push(DX);
    }

    if (this.tickIndex >= this.getInput('length') * 2) {
        if (this.tickIndex === this.getInput('length') * 2) {
            ADX = _.reduce(this.pastValues.DX, function(memo, pastDX) {
                return memo + pastDX;
            }, 0) / this.getInput('length');
        }
        else {
            ADX = ((this.pastValues.ADX * (this.getInput('length') - 1)) + DX) / this.getInput('length');
        }

        this.pastValues.ADX = ADX;
    }

    returnValue[this.getOutputMapping('pDI')] = +pDI.toFixed(2);
    returnValue[this.getOutputMapping('mDI')] = +mDI.toFixed(2);
    returnValue[this.getOutputMapping('ADX')] = +ADX.toFixed(2);

    return returnValue;
};

module.exports = AverageDirectionalIndex;
