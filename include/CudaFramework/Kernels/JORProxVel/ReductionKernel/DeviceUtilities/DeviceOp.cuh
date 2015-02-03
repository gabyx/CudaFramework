// ========================================================================================
//  CudaFramework
//  Copyright (C) 2014 by Gabriel Nützi <nuetzig (at) imes (d0t) mavt (d0t) ethz (d0t) ch>
//
//  This Source Code Form is subject to the terms of the GNU GPL 3.0 licence. 
//  If a copy of the GNU GPL 3.0 was not distributed with this
//  file, you can obtain one at http://opensource.org/licenses/GPL-3.0.
// ========================================================================================

/******************************************************************************
 * Copyright (c) 2013, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/******************************************************************************
 *
 * Code and text by Sean Baxter, NVIDIA Research
 * See http://nvlabs.github.io/moderngpu for repository and documentation.
 *
 ******************************************************************************/
#ifndef CudaFramework_Kernels_JORProxVel_ReductionKernel_DeviceUtilities_DeviceOp_cuh
#define CudaFramework_Kernels_JORProxVel_ReductionKernel_DeviceUtilities_DeviceOp_cuh

#pragma once
#include <cuda_runtime.h>
#if __CUDA_ARCH__ == 100
	#error "COMPUTE CAPABILITY 1.0 NOT SUPPORTED BY MPGU. TRY 2.0!"
#endif

#include <climits>
#include "CudaFramework/Kernels/JORProxVel/ReductionKernel/HostUtilities/Static.hpp"

#ifdef _MSC_VER
#define INLINESYMBOL __forceinline__
#else
#define INLINESYMBOL inline
#endif

namespace ReductionGPU {

#define MGPU_HOST __host__ INLINESYMBOL
#define MGPU_DEVICE __device__ INLINESYMBOL
#define MGPU_HOST_DEVICE __host__ __device__ INLINESYMBOL

const int WARP_SIZE = 32;
const int LOG_WARP_SIZE = 5;

////////////////////////////////////////////////////////////////////////////////
// Device-side comparison operators

template<typename T>
struct less : public std::binary_function<T, T, bool> {
	MGPU_HOST_DEVICE bool operator()(T a, T b) { return a < b; }
};
template<typename T>
struct less_equal : public std::binary_function<T, T, bool> {
	MGPU_HOST_DEVICE bool operator()(T a, T b) { return a <= b; }
};
template<typename T>
struct greater : public std::binary_function<T, T, bool> {
	MGPU_HOST_DEVICE bool operator()(T a, T b) { return a > b; }
};
template<typename T>
struct greater_equal : public std::binary_function<T, T, bool> {
	MGPU_HOST_DEVICE bool operator()(T a, T b) { return a >= b; }
};
template<typename T>
struct equal_to : public std::binary_function<T, T, bool> {
	MGPU_HOST_DEVICE bool operator()(T a, T b) { return a == b; }
};
template<typename T>
struct not_equal_to : public std::binary_function<T, T, bool> {
	MGPU_HOST_DEVICE bool operator()(T a, T b) { return a != b; }
};

////////////////////////////////////////////////////////////////////////////////
// Device-side arithmetic operators

template<typename T>
struct plus : public std::binary_function<T, T, T> {
	MGPU_HOST_DEVICE T operator()(T a, T b) { return a + b; }
};

template<typename T>
struct minus : public std::binary_function<T, T, T> {
	MGPU_HOST_DEVICE T operator()(T a, T b) { return a - b; }
};

template<typename T>
struct multiplies : public std::binary_function<T, T, T> {
	MGPU_HOST_DEVICE T operator()(T a, T b) { return a * b; }
};

template<typename T>
struct modulus : public std::binary_function<T, T, T> {
	MGPU_HOST_DEVICE T operator()(T a, T b) { return a % b; }
};

template<typename T>
struct bit_or : public std::binary_function<T, T, T> {
	MGPU_HOST_DEVICE T operator()(T a, T b) { return a | b; }
};

template<typename T>
struct bit_and : public std::binary_function<T, T, T> {
	MGPU_HOST_DEVICE T operator()(T a, T b) { return a & b; }
};

template<typename T>
struct bit_xor : public std::binary_function<T, T, T> {
	MGPU_HOST_DEVICE T operator()(T a, T b) { return a ^ b; }
};

template<typename T>
struct maximum : public std::binary_function<T, T, T> {
	MGPU_HOST_DEVICE T operator()(T a, T b) { return max(a, b); }
};

template<typename T>
struct minimum : public std::binary_function<T, T, T> {
	MGPU_HOST_DEVICE T operator()(T a, T b) { return min(a, b); }
};


////////////////////////////////////////////////////////////////////////////////

template<typename T>
MGPU_HOST_DEVICE void swap(T& a, T& b) {
	T c = a;
	a = b;
	b = c;
};


template<typename T>
struct DevicePair {
	T x, y;
};


template<typename T>
MGPU_HOST_DEVICE DevicePair<T> MakeDevicePair(T x, T y) {
	DevicePair<T> p = { x, y };
	return p;
}
}


# endif
