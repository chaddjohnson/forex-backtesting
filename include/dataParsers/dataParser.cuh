#ifndef DATAPARSER_H
#define DATAPARSER_H

#include <vector>
#include <map>
#include <string>
#include <sstream>
#include <iterator>
#include <cmath>
#include <cstdlib>
#include "types/tick.cuh"

class DataParser {
    private:
        std::string filePath;
        int type;

    protected:
        std::string getFilePath();
        int getType();
        float parseGroups(std::string groupString);

    public:
        DataParser(std::string filePath, int type);
        virtual ~DataParser() {}
        virtual std::vector<Tick*> parse() {
            return std::vector<Tick*>();
        }
        enum types { BACKTEST, FORWARDTEST };
};

#endif
