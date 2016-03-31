forex-backtesting
=================
Small Node.js-based framework for backtesting Forex trading strategies. Allows creation of strategies and studies, and makes inclusion of studies within strategies easy.

### Setup

1. Install Node.js: https://nodejs.org.
2. Install Gulp globally: `npm install -g gulp`.
3. Install Node modules: `npm install`.
4. Download minute tick data for one security from a supported data provider.
    1. MetaTrader link: http://www.fxdd.com/us/en/forex-resources/forex-trading-tools/metatrader-1-minute-data/
5. Open MetaTrader, and open the History Center. Then import the data and export it into CSV format.
6. Put the data into ./data/:
    1. mkdir ./data/metatrader
    1. mv AUDJPY.csv ./data/metatrader
7. Create database indexes:
```javascript
db.positions.createIndex({symbol: 1});
db.positions.createIndex({strategyUuid: 1});
db.backtests.createIndex({symbol: 1});
db.datapoints.createIndex({symbol: 1});
db.datapoints.createIndex({"data.timestamp": 1});
db.datapoints.createIndex({"data.groups.testing": 1});
db.datapoints.createIndex({"data.groups.validation": 1});
db.forwardtests.createIndex({symbol: 1});
db.forwardtests.createIndex({group: 1});
db.validations.createIndex({symbol: 1});
db.validations.createIndex({winRate: 1});
db.validations.createIndex({symbol: 1, configuration: 1});
```
8. Run `./backtest.sh AUDJPY`.
