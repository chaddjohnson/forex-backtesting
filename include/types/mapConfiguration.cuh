#ifndef MAPCONFIGURATION_H
#define MAPCONFIGURATION_H

#include "types/real.cuh"

typedef std::map<std::string, boost::variant<int, Real>> MapConfiguration;

#endif

