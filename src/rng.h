#ifndef CDTAU_RNG_H
#define CDTAU_RNG_H

#ifdef __AVX2__

#include "philox/PhiloxSIMD.h"
typedef at::philox_simd_engine RNGEngine;


#else


#include <random>
typedef std::mt19937 RNGEngine;


#endif

#endif  // CDTAU_RNG_H
