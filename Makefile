CC = g++
NVCC = nvcc
CFLAGS = -std=c++11 -Wall -pedantic
LFLAGS = -L/usr/local/lib -Llib
INCLUDES = -I/usr/local/include -Iinclude
LIBS = -lgsl -lcblas
BIN = ./bin
BUILD = ./build

all: prepareData optimize

prepareData: src/prepareData.cpp optimizerFactory.o reversalsOptimizer.o optimizer.o oandaDataParser.o study.o smaStudy.o emaStudy.o rsiStudy.o stochasticOscillatorStudy.o polynomialRegressionChannelStudy.o
	$(CC) $(CFLAGS) $(LFLAGS) $(INCLUDES) $(LIBS) -o $(BIN)/$@ src/prepareData.cpp $(BUILD)/optimizerFactory.o $(BUILD)/reversalsOptimizer.o $(BUILD)/optimizer.o $(BUILD)/oandaDataParser.o $(BUILD)/study.o $(BUILD)/smaStudy.o $(BUILD)/emaStudy.o $(BUILD)/rsiStudy.o $(BUILD)/stochasticOscillatorStudy.o $(BUILD)/polynomialRegressionChannelStudy.o

optimize: src/optimize.cpp optimizerFactory.o reversalsOptimizer.o optimizer.o oandaDataParser.o study.o smaStudy.o emaStudy.o rsiStudy.o stochasticOscillatorStudy.o polynomialRegressionChannelStudy.o
	$(CC) $(CFLAGS) $(LFLAGS) $(INCLUDES) $(LIBS) -o $(BIN)/$@ src/optimize.cpp $(BUILD)/optimizerFactory.o $(BUILD)/reversalsOptimizer.o $(BUILD)/optimizer.o $(BUILD)/oandaDataParser.o $(BUILD)/study.o $(BUILD)/smaStudy.o $(BUILD)/emaStudy.o $(BUILD)/rsiStudy.o $(BUILD)/stochasticOscillatorStudy.o $(BUILD)/polynomialRegressionChannelStudy.o

optimizerFactory.o: lib/optimizers/optimizerFactory.cpp
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $(BUILD)/$@ lib/optimizers/optimizerFactory.cpp

optimizer.o: lib/optimizers/optimizer.cpp
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $(BUILD)/$@ lib/optimizers/optimizer.cpp

reversalsOptimizer.o: lib/optimizers/reversalsOptimizer.cpp
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $(BUILD)/$@ lib/optimizers/reversalsOptimizer.cpp

oandaDataParser.o: lib/dataParsers/oandaDataParser.cpp
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $(BUILD)/$@ lib/dataParsers/oandaDataParser.cpp

study.o: lib/studies/study.cpp
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $(BUILD)/$@ lib/studies/study.cpp

smaStudy.o: lib/studies/smaStudy.cpp
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $(BUILD)/$@ lib/studies/smaStudy.cpp

emaStudy.o: lib/studies/emaStudy.cpp
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $(BUILD)/$@ lib/studies/emaStudy.cpp

rsiStudy.o: lib/studies/rsiStudy.cpp
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $(BUILD)/$@ lib/studies/rsiStudy.cpp

stochasticOscillatorStudy.o: lib/studies/stochasticOscillatorStudy.cpp
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $(BUILD)/$@ lib/studies/stochasticOscillatorStudy.cpp

polynomialRegressionChannelStudy.o: lib/studies/polynomialRegressionChannelStudy.cpp
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $(BUILD)/$@ lib/studies/polynomialRegressionChannelStudy.cpp

clean:
	rm -f $(BIN)/* $(BUILD)/*.o
