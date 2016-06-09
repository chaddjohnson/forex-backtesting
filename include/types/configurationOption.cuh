#ifndef CONFIGURATIONOPTION_H
#define CONFIGURATIONOPTION_H

#include "types/real.cuh"

typedef std::vector<std::map<std::string, boost::variant<std::string, Real>>> ConfigurationOption;

#endif

