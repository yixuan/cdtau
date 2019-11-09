#pragma once

// define constants like M_PI and C keywords for MSVC
#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#include <math.h>
#endif

#include <stdint.h>
#include <x86intrin.h>
#include "splitmix64.h"

#include "Array.h"
#include <cmath>
#include <iostream>

#define UNSHUFFLE 0  // set to 1 to match non-SIMD ordering


namespace at {

// typedefs for holding vector data
namespace {

typedef at::detail::Array<uint32_t, 4> UINT4;
typedef at::detail::Array<uint32_t, 2> UINT2;
typedef at::detail::Array<double, 2> DOUBLE2;
typedef at::detail::Array<float, 2> FLOAT2;

// cache size for holding 128 bits philox randoms
// constexpr int philox_random_cache_size = 128;

} // anonymous namespace

static inline void print_m128i(__m128i v) {
    unsigned dst[4];
    _mm_storeu_si128((__m128i*)&dst, v);
    printf("%08x %08x %08x %08x\n", dst[0], dst[1], dst[2], dst[3]);
}

static inline void print_m256i(__m256i v) {
    unsigned dst[8];
    _mm256_storeu_si256((__m256i*)&dst, v);
    printf("%08x %08x %08x %08x %08x %08x %08x %08x\n", dst[0], dst[1], dst[2], dst[3], dst[4], dst[5], dst[6], dst[7]);
}

/**
 * Note [Philox Engine implementation]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Originally implemented in PyTorch's fusion compiler
 * Refer to: http://www.thesalmons.org/john/random123/papers/random123sc11.pdf
 * for details regarding the engine.
 *
 * The Philox engine is currently used in CUDA distributions
 * kernels as its random engine. 
 * 
 * It takes a seed value, a subsequeunce
 * for starting the generation and an offset for the sequence.
 *
 * Think of this engine as an algorithm producing a huge array. We are 
 * parallelizing this array by partitioning the huge array and assigning 
 * a thread index to each partition. In other words, each seed value 
 * (there are 2^64 possible seed values) gives a sub array of size 
 * 2^128 (each element in that array is a 128 bit number). Reasoning
 * behind the array being of size 2^128 is, there are 2^64 possible
 * thread index value and there is an array of size 2^64 for each of
 * those thread index. Hence 2^64 * 2^64 = 2^128 for each seed value.
 *
 * In short, this generator can produce 2^64 (seed values) * 2^128 (number
 * of elements in an array given by a seed value) = 2^192 values.
 *
 * Arguments:
 * seed:        Seed values could be any number from 0 to 2^64-1.
 * subsequence: Subsequence is just the cuda thread indexing with:
 *              - blockIdx.x * blockDim.x + threadIdx.x
 * offset:      The offset variable in PhiloxEngine  decides how many 128-bit 
 *              random numbers to skip (i.e. how many groups of 4, 32-bit numbers to skip)
 *              and hence really decides the total number of randoms that can be achieved 
 *              for the given subsequence.
 */
class philox_simd_engine {
public:

  inline explicit philox_simd_engine(uint64_t seed = 67280421310721,
                                     uint64_t subsequence = 0,
                                     uint64_t offset = 0) {
    key[0] = static_cast<uint32_t>(seed);
    key[1] = static_cast<uint32_t>(seed >> 32);
    counter = UINT4(0);
    counter[2] = static_cast<uint32_t>(subsequence);
    counter[3] = static_cast<uint32_t>(subsequence >> 32);
    STATE = 0;
    incr_n(offset);
  }

  inline void next32(__m256i& out0, __m256i& out1, __m256i& out2, __m256i& out3) {
    __m256i counter0 = _mm256_set1_epi32(counter[0]);
    __m256i counter1 = _mm256_set1_epi32(counter[1]);
    __m256i counter2 = _mm256_set1_epi32(counter[2]);
    __m256i counter3 = _mm256_set1_epi32(counter[3]);

    if (__builtin_expect(counter[0] < 4294967288, 1)) {
      counter0 = _mm256_add_epi32(counter0, _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));
      counter[0] += 8;
    } else {
      uint32_t tmp0[8];
      uint32_t tmp1[8];
      uint32_t tmp2[8];
      uint32_t tmp3[8];
      for (int j = 0; j < 8; j++) {
        tmp0[j] = counter[0];
        tmp1[j] = counter[1];
        tmp2[j] = counter[2];
        tmp3[j] = counter[3];
        incr();
      }
      counter0 = _mm256_loadu_si256((__m256i*)tmp0);
      counter1 = _mm256_loadu_si256((__m256i*)tmp1);
      counter2 = _mm256_loadu_si256((__m256i*)tmp2);
      counter3 = _mm256_loadu_si256((__m256i*)tmp3);
    }

    __m256i key0 = _mm256_set1_epi32(key[0]);
    __m256i key1 = _mm256_set1_epi32(key[1]);
    for (int j = 0; j < 10; j++) {
      single_round(counter0, counter1, counter2, counter3, key0, key1);
    }

#if UNSHUFFLE
    transpose(counter0, counter1, counter2, counter3);
#endif

    out0 = counter0;
    out1 = counter1;
    out2 = counter2;
    out3 = counter3;
  }

  inline uint32_t operator()() {
    if(STATE == 0) {
      __m256i a, b, c, d;
      next32(a, b, c, d);
      _mm256_storeu_si256((__m256i*)&output[0], a);
      _mm256_storeu_si256((__m256i*)&output[8], b);
      _mm256_storeu_si256((__m256i*)&output[16], c);
      _mm256_storeu_si256((__m256i*)&output[24], d);
    }
    uint32_t ret = output[STATE];
    STATE = (STATE + 1) & 31;
    return ret;
  }

  /**
   * Function that Skips N 128 bit numbers in a subsequence
   */
  inline void incr_n(uint64_t n) {
    uint32_t nlo = static_cast<uint32_t>(n);
    uint32_t nhi = static_cast<uint32_t>(n >> 32);
    counter[0] += nlo;
    // if overflow in x has occured, carry over to nhi
    if (counter[0] < nlo) {
      nhi++;
      // if overflow in nhi has occured during carry over,
      // propagate that overflow to y and exit to increment z
      // otherwise return
      counter[1] += nhi;
      if(nhi != 0) {
        if (nhi <= counter[1]) {
          return;
        }
      }
    } else {
      // if overflow in y has occured during addition,
      // exit to increment z
      // otherwise return
      counter[1] += nhi;
      if (nhi <= counter[1]) {
        return;
      }
    }
    if (++counter[2])
      return;
    ++counter[3];
  }

  /**
   * Function that Skips one 128 bit number in a subsequence
   */
  inline void incr() {
    if (++counter[0] == 0) {
      if (++counter[1] == 0) {
        if (++counter[2] == 0) {
          ++counter[3];
        }
      }
    }
  }

private:
  UINT4 counter;
  uint32_t output[32];
  UINT2 key;
  uint32_t STATE;

  void single_round(__m256i& ctr0, __m256i& ctr1, __m256i& ctr2, __m256i& ctr3,
                    __m256i& key0, __m256i& key1) {
    __m256i lohi0a = _mm256_mul_epu32(ctr0, _mm256_set1_epi32(kPhiloxSA));
    __m256i lohi0b = _mm256_mul_epu32(_mm256_srli_epi64(ctr0, 32), _mm256_set1_epi32(kPhiloxSA));
    __m256i lohi1a = _mm256_mul_epu32(ctr2, _mm256_set1_epi32(kPhiloxSB));
    __m256i lohi1b = _mm256_mul_epu32(_mm256_srli_epi64(ctr2, 32), _mm256_set1_epi32(kPhiloxSB));

    lohi0a = _mm256_shuffle_epi32(lohi0a, 0xD8);
    lohi0b = _mm256_shuffle_epi32(lohi0b, 0xD8);
    lohi1a = _mm256_shuffle_epi32(lohi1a, 0xD8);
    lohi1b = _mm256_shuffle_epi32(lohi1b, 0xD8);

    __m256i lo0 = _mm256_unpacklo_epi32(lohi0a, lohi0b);
    __m256i hi0 = _mm256_unpackhi_epi32(lohi0a, lohi0b);
    __m256i lo1 = _mm256_unpacklo_epi32(lohi1a, lohi1b);
    __m256i hi1 = _mm256_unpackhi_epi32(lohi1a, lohi1b);

    // ctr0 = hi1 ^ ctr[1] ^ key[0]
    ctr0 = _mm256_xor_si256(ctr1, key0);
    ctr0 = _mm256_xor_si256(ctr0, hi1);

    // ctr1 = lo1
    ctr1 = lo1;

    // ctr2 = hi0 ^ ctr[3] ^ key[1];
    ctr2 = _mm256_xor_si256(ctr3, key1);
    ctr2 = _mm256_xor_si256(ctr2, hi0);

    // ctr3 = lo0
    ctr3 = lo0;

    key0 = _mm256_add_epi32(key0, _mm256_set1_epi32(kPhilox10A));
    key1 = _mm256_add_epi32(key1, _mm256_set1_epi32(kPhilox10B));
  }

  inline void transpose(__m256i& ctr0, __m256i& ctr1, __m256i& ctr2, __m256i& ctr3) {
    __m256i a0, a1, a2, a3;
    a0 = _mm256_unpacklo_epi32(ctr0, ctr1);
    a2 = _mm256_unpacklo_epi32(ctr2, ctr3);
    a1 = _mm256_unpackhi_epi32(ctr0, ctr1);
    a3 = _mm256_unpackhi_epi32(ctr2, ctr3);

    __m256i b0, b1, b2, b3;
    b0 = _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(a0), _mm256_castsi256_ps(a2), 0x44));
    b1 = _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(a0), _mm256_castsi256_ps(a2), 0xEE));
    b2 = _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(a1), _mm256_castsi256_ps(a3), 0x44));
    b3 = _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(a1), _mm256_castsi256_ps(a3), 0xEE));

    ctr0 = _mm256_permute2f128_si256(b0, b1, 0x20);
    ctr2 = _mm256_permute2f128_si256(b0, b1, 0x31);
    ctr1 = _mm256_permute2f128_si256(b2, b3, 0x20);
    ctr3 = _mm256_permute2f128_si256(b2, b3, 0x31);
  }

  static const uint32_t kPhilox10A = 0x9E3779B9;
  static const uint32_t kPhilox10B = 0xBB67AE85;
  static const uint32_t kPhiloxSA = 0xD2511F53;
  static const uint32_t kPhiloxSB = 0xCD9E8D57;
};


} // namespace at
