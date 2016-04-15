CC = g++
NVCC = nvcc
CFLAGS = -std=c++11 -Wall -pedantic
LFLAGS = -L/usr/local/lib -Llib
INCLUDES = -I/usr/local/include -Iinclude
LIBS = -lgsl -lcblas
BIN = ./bin
BUILD = ./build

all: prepareData

prepareData: src/prepareData.cpp oandaDataParser.o study.o smaStudy.o emaStudy.o rsiStudy.o stochasticOscillator.o polynomialRegressionChannelStudy.o
	$(CC) $(CFLAGS) $(LFLAGS) $(INCLUDES) $(LIBS) -o $(BIN)/$@ src/prepareData.cpp $(BUILD)/oandaDataParser.o $(BUILD)/study.o $(BUILD)/smaStudy.o $(BUILD)/emaStudy.o $(BUILD)/rsiStudy.o $(BUILD)/stochasticOscillator.o $(BUILD)/polynomialRegressionChannelStudy.o

optimize:
	$(NVCC) $(LIBS) $(INCLUDES) -o $(BIN)/optimize

oandaDataParser.o: ./lib/dataParsers/oandaDataParser.cpp
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $(BUILD)/$@ ./lib/dataParsers/oandaDataParser.cpp

study.o: ./lib/studies/study.cpp
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $(BUILD)/$@ ./lib/studies/study.cpp

smaStudy.o: ./lib/studies/smaStudy.cpp
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $(BUILD)/$@ ./lib/studies/smaStudy.cpp

emaStudy.o: ./lib/studies/emaStudy.cpp
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $(BUILD)/$@ ./lib/studies/emaStudy.cpp

rsiStudy.o: ./lib/studies/rsiStudy.cpp
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $(BUILD)/$@ ./lib/studies/rsiStudy.cpp

stochasticOscillator.o: ./lib/studies/stochasticOscillator.cpp
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $(BUILD)/$@ ./lib/studies/stochasticOscillator.cpp

polynomialRegressionChannelStudy.o: ./lib/studies/polynomialRegressionChannelStudy.cpp
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $(BUILD)/$@ ./lib/studies/polynomialRegressionChannelStudy.cpp

clean:
	rm -f $(BIN)/* $(BUILD)/*.o
