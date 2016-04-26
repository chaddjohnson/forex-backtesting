CC = g++
NVCC = nvcc
CFLAGS = -O3 -std=c++11 -Wall -pedantic
LFLAGS = -L/usr/local/lib -Llib
INCLUDES = -I/usr/include -I/usr/local/include -Iinclude
LIBS = -lgsl -lcblas
BIN = ./bin
OBJDIR = ./obj
OBJ = lib/factories/optimizerFactory.o lib/optimizers/reversalsOptimizer.o lib/optimizers/optimizer.o \
      lib/factories/dataParserFactory.o lib/dataParsers/oandaDataParser.o lib/dataParsers/dataParser.o \
      lib/studies/study.o lib/studies/smaStudy.o lib/studies/emaStudy.o lib/studies/rsiStudy.o \
      lib/studies/stochasticOscillatorStudy.o lib/studies/polynomialRegressionChannelStudy.o

all: prepareData optimize

prepareData: src/prepareData.cpp $(OBJ)
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(LFLAGS) $(INCLUDES) -o $(BIN)/$@ src/prepareData.cpp $(addprefix $(OBJDIR)/,$(OBJ)) $(LIBS)

optimize: src/optimize.cpp $(OBJ)
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(LFLAGS) $(INCLUDES) -o $(BIN)/$@ src/optimize.cpp $(addprefix $(OBJDIR)/,$(OBJ)) $(LIBS)

%.o: %.cpp
	@mkdir -p $(OBJDIR)/lib/factories $(OBJDIR)/lib/optimizers $(OBJDIR)/lib/dataParsers $(OBJDIR)/lib/studies
	$(CC) $(CFLAGS) $(INCLUDES) -o $(OBJDIR)/$@ -c $<
clean:
	rm -f $(BIN)/* $(OBJDIR)/*.o
