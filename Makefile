CC = g++
NVCC = nvcc
CFLAGS = -std=c++11 -Wall -pedantic
LFLAGS = -L/usr/local/lib -Llib $(shell pkg-config --libs libmongoc-1.0 libbson-1.0)
INCLUDES = -I/usr/include -I/usr/local/include -Iinclude $(shell pkg-config --cflags libmongoc-1.0 libbson-1.0)
LIBS = -lgsl -lcblas
BIN = ./bin
OBJDIR = ./obj
OBJ = factories/optimizationStrategyFactory.o positions/callPosition.o positions/putPosition.o \
      positions/position.o strategies/reversalsOptimizationStrategy.o strategies/optimizationStrategy.o \
      strategies/strategy.o factories/optimizerFactory.o optimizers/reversalsOptimizer.o optimizers/optimizer.o \
      factories/dataParserFactory.o dataParsers/oandaDataParser.o dataParsers/dataParser.o \
      studies/study.o studies/smaStudy.o studies/emaStudy.o studies/rsiStudy.o \
      studies/stochasticOscillatorStudy.o studies/polynomialRegressionChannelStudy.o

all: prepareData optimize

prepareData: src/prepareData.cpp $(addprefix lib/,$(OBJ))
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) -o $(BIN)/$@ src/prepareData.cpp $(addprefix $(OBJDIR)/,$(addprefix lib/,$(OBJ))) $(LFLAGS) $(LIBS)

optimize: src/optimize.cpp $(addprefix lib/,$(OBJ))
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) -o $(BIN)/$@ src/optimize.cpp $(addprefix $(OBJDIR)/,$(addprefix lib/,$(OBJ))) $(LFLAGS) $(LIBS)

%.o: %.cpp
	@mkdir -p $(OBJDIR)/lib/strategies $(OBJDIR)/lib/positions $(OBJDIR)/lib/factories $(OBJDIR)/lib/optimizers $(OBJDIR)/lib/dataParsers $(OBJDIR)/lib/studies
	$(CC) $(CFLAGS) $(INCLUDES) -o $(OBJDIR)/$@ -c $<
clean:
	rm -f $(BIN)/* $(OBJDIR)/*.o
