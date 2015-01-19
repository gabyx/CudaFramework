// -*- c++ -*-
//
// Copyright 2005, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Authors: Zhanyong Wan, Sean Mcafee
//
// Taken from The Google C++ Testing Framework (Google Test).
// Modified for this discussion by Fred Richards.
// ========================================================================================
//  CudaFramework
//  Copyright (C) 2014 by Gabriel Nützi <nuetzig (at) imes (d0t) mavt (d0t) ethz (d0t) ch>
//
//  This Source Code Form is subject to the terms of the GNU GPL 3.0 licence.
//  If a copy of the GNU GPL 3.0 was not distributed with this
//  file, you can obtain one at http://opensource.org/licenses/GPL-3.0.
// ========================================================================================

#ifndef CudaFramework_General_FloatingPointType_hpp
#define CudaFramework_General_FloatingPointType_hpp

#include <cstddef>
#if defined _MSC_VER  // may need to check your VC++ version
  typedef unsigned __int32 uint32_t;
  typedef unsigned __int64 uint64_t;
  typedef __int32 int32_t;
  typedef __int64 int64_t;

#else
  #include <stdint.h>
  #include <cstdint>
#endif

#include <cmath>
#include <limits>

#define USE_GOGGLES_IMPLEMENTATION 1



#if USE_GOGGLES_IMPLEMENTATION == 1
namespace
{
    template <size_t bytes>
    struct TypeWithSize
    {
        typedef void UInt;
    };

    template <>
    struct TypeWithSize<4>
    {
        typedef uint32_t UInt;
    };

    template <>
    struct TypeWithSize<8>
    {
        typedef uint64_t UInt;
    };
}

template <typename RawType>
class FloatingPoint
{
public:
    typedef typename TypeWithSize<sizeof(RawType)>::UInt Bits;

    static const size_t kBitCount = 8*sizeof(RawType);
    static const size_t kFracBitCount = std::numeric_limits<RawType>::digits - 1;
    static const size_t kExpBitCount = kBitCount - 1 - kFracBitCount;

    static const Bits kSignBitMask = static_cast<Bits>(1) << (kBitCount - 1);
    static const Bits kFracBitMask = ~static_cast<Bits>(0) >> (kExpBitCount + 1);
    static const Bits kExpBitMask = ~(kSignBitMask | kFracBitMask);

    static const size_t kMaxUlps = 4;


    explicit FloatingPoint(const RawType& x) : value_(x) {}

    //
    // Now checking for NaN to match == behavior.
    //
    bool AlmostEquals(const FloatingPoint& rhs) const {
        if (is_nan() || rhs.is_nan()) return false;
        return ULP_diff(bits_, rhs.bits_) <= kMaxUlps;
    }

     //
    // Now checking for NaN to match == behavior.
    //
    bool AlmostEquals(const FloatingPoint& rhs, const size_t maxUlps, Bits & computedUlpDiff) const {
        if (is_nan() || rhs.is_nan()) return false;
        computedUlpDiff = ULP_diff(bits_, rhs.bits_);
        return computedUlpDiff <= maxUlps;
    }

private:
    bool is_nan() const {
        return ((kExpBitMask & bits_) == kExpBitMask) &&
	        ((kFracBitMask & bits_) != 0);
    }

    Bits SignAndMagnitudeToBiased(const Bits& sam) const {
        if (kSignBitMask & sam) {
            return ~sam + 1;  // two's complement
        } else {
            return kSignBitMask | sam;  // * 2
        }
    }

    Bits ULP_diff(const Bits& sam1, const Bits& sam2) const
    {
        const Bits biased1 = SignAndMagnitudeToBiased(sam1);
        const Bits biased2 = SignAndMagnitudeToBiased(sam2);

        return (biased1 >= biased2) ? (biased1 - biased2) : (biased2 - biased1);
    }

    union {
        RawType value_;
        Bits bits_;
    };
};

#else

using std::uint64_t;
using std::int64_t;
using std::uint32_t;
using std::int32_t;

bool is_nan(float f) {
  uint32_t const & b = reinterpret_cast<uint32_t const &>(f);
  return (b & 0x7f800000) == 0x7f800000 && (b & 0x007fffff);
}

bool is_nan(double f) {
  uint64_t const & b = reinterpret_cast<uint64_t const &>(f);
  return (b & 0x7ff0000000000000l) == 0x7ff0000000000000l && (b & 0x000fffffffffffffl);
}

bool is_snan(double f) {
  uint64_t const & b = reinterpret_cast<uint64_t const &>(f);
  return (b & 0x7ff0000000000000l) == 0x7ff0000000000000l && (b & 0x000fffffffffffffl) && !(b & 0x0008000000000000l);
}

bool is_qnan(double f) {
  uint64_t const & b = reinterpret_cast<uint64_t const &>(f);
  return (b & 0x7ff0000000000000l) == 0x7ff0000000000000l && (b & 0x000fffffffffffffl) && (b & 0x0008000000000000l);
}

bool is_snan(float f) {
  uint32_t const & b = reinterpret_cast<uint32_t const &>(f);
  return (b & 0x7f800000) == 0x7f800000 && (b & 0x007fffff) && !(b & 0x00400000);
}

bool is_qnan(float f) {
  uint32_t const & b = reinterpret_cast<uint32_t const &>(f);
  return (b & 0x7f800000) == 0x7f800000 && (b & 0x007fffff) && (b & 0x00400000);
}

bool is_inf(double f) {
  uint64_t const & b = reinterpret_cast<uint64_t const &>(f);
  return (b & 0x7fffffffffffffffl) == 0x7ff0000000000000l;
}

bool is_inf(float f) {
  uint32_t const & b = reinterpret_cast<uint32_t const &>(f);
  return (b & 0x7fffffff) == 0x7f800000;
}

uint32_t payload(float f) {
  uint32_t const & b = reinterpret_cast<uint32_t const &>(f);
  return (b & 0x003fffff);
}

uint64_t payload(double f) {
  uint64_t const & b = reinterpret_cast<uint64_t const &>(f);
  return (b & 0x0007ffffffffffffl);
}

uint64_t mantissa(double f) {
  uint64_t const & b = reinterpret_cast<uint64_t const &>(f);
  return (b & 0x000fffffffffffffl);
}

uint64_t exponent(double f) {
  uint64_t const & b = reinterpret_cast<uint64_t const &>(f);
  return ((b & 0x7ff0000000000000l) >> 52) - 1023;
}

uint32_t mantissa(float v) {
  uint32_t bits = reinterpret_cast<uint32_t const &>(v);
  return bits & 0x007fffff;
}

uint32_t exponent(float v) {
  uint32_t bits = reinterpret_cast<uint32_t const &>(v);
  return ((bits & 0x7f800000) >> 23) - 127;
}

uint64_t sign(double v) {
  uint64_t bits = reinterpret_cast<uint64_t const &>(v);
  return bits & 0x8000000000000000l;
}

uint32_t sign(float v) {
  uint32_t bits = reinterpret_cast<uint32_t const &>(v);
  return bits & 0x80000000;
}


double next(double prev) {
  int64_t bits = reinterpret_cast<int64_t const &>(prev);

  if(bits < 0) {
    --bits;
    if(bits > 0)
      bits = 0;
  } else
    ++bits;
  return reinterpret_cast<double const &>(bits);
}

double prev(double v) {
  int64_t bits = reinterpret_cast<int64_t const &>(v);

  if(bits < 0)
    ++bits;
  else {
    --bits;
    if(bits < 0)
      bits = 0x8000000000000000l;
  }
  return reinterpret_cast<double const &>(bits);
}

float next(float prev) {
  int32_t bits = reinterpret_cast<int32_t const &>(prev);

  if(bits < 0) {
    --bits;
    if(bits > 0)
      bits = 0;
  } else
    ++bits;

  return reinterpret_cast<float const &>(bits);
}

float prev(float v) {
  int32_t bits = reinterpret_cast<int32_t const &>(v);

  if(bits < 0)
    ++bits;
  else {
    --bits;
    if(bits < 0)
      bits = 0x80000000;
  }
  return reinterpret_cast<float const &>(bits);
}

uint32_t ulp_distance(float a, float b) {
  int32_t bits1 = reinterpret_cast<int32_t const &>(a);
  int32_t bits2 = reinterpret_cast<int32_t const &>(b);

  //if float a is negative build two's complement
  if(bits1 < 0)
    bits1 = 0x80000000 - bits1;

  //if float b is negative build two's complement
  if(bits2 < 0)
    bits2 = 0x80000000 - bits2;

  return std::abs(bits1 - bits2);
}

uint64_t ulp_distance(double a, double b) {
  int64_t bits1 = reinterpret_cast<int64_t const &>(a);
  int64_t bits2 = reinterpret_cast<int64_t const &>(b);

  if(bits1 < 0)
    bits1 = 0x8000000000000000l - bits1;

  if(bits2 < 0)
    bits2 = 0x8000000000000000l - bits2;

  return std::abs(bits1 - bits2);
}

#endif


#endif // FLOATINGPOINT_H
