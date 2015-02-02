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

# ifndef loadstore_cuh
# define loadstore_cuh

#include <cuda_runtime.h>
#include "CudaFramework/Kernels/JORProxVel/ReductionKernel/DeviceUtilities/Instantiations.cuh"

namespace ReductionGPU {

////////////////////////////////////////////////////////////////////////////////
// Cooperative load functions.

template<int NT, int VT, typename InputIt, typename T>
__device__ void deviceSharedToReg(InputIt data, int tid, T* reg,
	bool sync) {

	#pragma unroll
	for(int i = 0; i < VT; ++i)
		reg[i] = data[NT * i + tid];

	if(sync) __syncthreads();
}

template<int NT, int VT, typename InputIt, typename T>
__device__ void DeviceGlobalToRegPred(int count, InputIt data, int tid,
	T* reg, bool sync) {

	// TODO: Attempt to issue 4 loads at a time.
	#pragma unroll
	for(int i = 0; i < VT; ++i) {
		int index = NT * i + tid;
		if(index < count) reg[i] = data[index];
	}
	if(sync) __syncthreads();
}

template<int NT, int VT, typename InputIt, typename T>
__device__ void DeviceGlobalToReg(int count, InputIt data, int tid,
	T* reg, bool sync) {

	if(count >= NT * VT) {
		#pragma unroll
		for(int i = 0; i < VT; ++i)
			reg[i] = data[NT * i + tid];
	} else
		DeviceGlobalToRegPred<NT, VT>(count, data, tid, reg, false);
	if(sync) __syncthreads();
}



template<int NT, int VT, typename InputIt, typename T>
__device__ void DeviceGlobalToRegDefault(int count, InputIt data, int tid,
	T* reg, T init, bool sync) {

	if(count >= NT * VT) {
		#pragma unroll
		for(int i = 0; i < VT; ++i)
			reg[i] = data[NT * i + tid];
	} else {
		#pragma unroll
		for(int i = 0; i < VT; ++i) {
			int index = NT * i + tid;
			reg[i] = init;
			if(index < count) reg[i] = data[index];
		}
	}
	if(sync) __syncthreads();
}



template<int NT, int VT, typename InputIt, typename T>
__device__ void deviceGlobalToThreadDefault(int count, InputIt data, int tid,
	T* reg, T init) {

	data += VT * tid;
	if(count >= NT * VT) {
		#pragma unroll
		for(int i = 0; i < VT; ++i)
			reg[i] = ldg(data + i);
	} else {
		count -= VT * tid;
		#pragma unroll
		for(int i = 0; i < VT; ++i)
			reg[i] = (i < count) ? ldg(data + i) : init;
	}
}


////////////////////////////////////////////////////////////////////////////////
// Cooperative store functions.

template<int NT, int VT, typename OutputIt, typename T>
__device__ void DeviceRegToShared(const T* reg, int tid,
	OutputIt dest, bool sync) {

	typedef typename std::iterator_traits<OutputIt>::value_type T2;
	#pragma unroll
	for(int i = 0; i < VT; ++i)
		dest[NT * i + tid] = (T2)reg[i];

	if(sync) __syncthreads();
}

template<int NT, int VT, typename InputIt, typename T>
__device__ void DeviceGlobalToShared(int count, InputIt source, int tid,
	T* dest, bool sync) {

	T reg[VT];
	DeviceGlobalToReg<NT, VT>(count, source, tid, reg, false);
	DeviceRegToShared<NT, VT>(reg, tid, dest, sync);
}

////////////////////////////////////////////////////////////////////////////////

template<int NT, int VT, typename InputIt, typename T>
__device__ void DeviceGlobalToSharedLoop(int count, InputIt source, int tid,
	T* dest, bool sync) {

	const int Granularity = MGPU_MIN(VT, 3);
	DeviceGlobalToShared<NT, Granularity>(count, source, tid, dest, false);

	int offset = Granularity * NT;
	if(count > offset)
		DeviceGlobalToShared<NT, VT - Granularity>(count - offset,
			source + offset, tid, dest + offset, false);

	if(sync) __syncthreads();


}



////////////////////////////////////////////////////////////////////////////////
// Transponse VT elements in NT threads (x) into thread-order registers (y)
// using only NT * VT / 2 elements of shared memory.

template<int NT, int VT, typename T>
__device__ void HalfSmemTranspose(const T* x, int tid, T* shared, T* y) {

	// Transpose the first half values (tid < NT / 2)
	#pragma unroll
	for(int i = 0; i <= VT / 2; ++i)
		if(i < VT / 2 || tid < NT / 2)
			shared[NT * i + tid] = x[i];
	__syncthreads();

	if(tid < NT / 2) {
		#pragma unroll
		for(int i = 0; i < VT; ++i)
			y[i] = shared[VT * tid + i];
	}
	__syncthreads();

	// Transpose the second half values (tid >= NT / 2)
	#pragma unroll
	for(int i = VT / 2; i < VT; ++i)
		if(i > VT / 2 || tid >= NT / 2)
			shared[NT * i - NT * VT / 2 + tid] = x[i];
	__syncthreads();

	if(tid >= NT / 2) {
		#pragma unroll
		for(int i = 0; i < VT; ++i)
			y[i] = shared[VT * tid + i - NT * VT / 2];
	}
	__syncthreads();
}

////////////////////////////////////////////////////////////////////////////////
// Gather/scatter functions

////////////////////////////////////////////////////////////////////////////////
// Cooperative transpose functions (strided to thread order)

template<int VT, typename T>
__device__ void DeviceThreadToShared(const T* threadReg, int tid, T* shared,
	bool sync) {

	if(1 & VT) {
		// Odd grain size. Store as type T.
		#pragma unroll
		for(int i = 0; i < VT; ++i)
			shared[VT * tid + i] = threadReg[i];
	} else {
		// Even grain size. Store as DevicePair<T>. This lets us exploit the
		// 8-byte shared memory mode on Kepler.
		DevicePair<T>* dest = (DevicePair<T>*)(shared + VT * tid);
		#pragma unroll
		for(int i = 0; i < VT / 2; ++i)
			dest[i] = MakeDevicePair(threadReg[2 * i], threadReg[2 * i + 1]);
	}
	if(sync) __syncthreads();
}

template<int VT, typename T>
__device__ void DeviceSharedToThread(const T* shared, int tid, T* threadReg,
	bool sync) {

	if(1 & VT) {
		#pragma unroll
		for(int i = 0; i < VT; ++i)
			threadReg[i] = shared[VT * tid + i];
	} else {
		const DevicePair<T>* source = (const DevicePair<T>*)(shared + VT * tid);
		#pragma unroll
		for(int i = 0; i < VT / 2; ++i) {
			DevicePair<T> p = source[i];
			threadReg[2 * i] = p.x;
			threadReg[2 * i + 1] = p.y;
		}
	}
	if(sync) __syncthreads();
}

///
///   Instantiations
///
/*
////////////////////////////////////////////////////////////////////////////////
// DeviceLoad2 - load from pointers of the same type. Optimize for a single LD
// statement.
template<int NT, int VT, typename InputIt, typename T>
__device__ void DeviceGlobalToRegDefault(int count, InputIt data, int tid,
	T* reg, T init, bool sync = false);

template<int NT, int VT, typename OutputIt, typename T>
__device__ void DeviceRegToShared(const T* reg, int tid, OutputIt dest,
	bool sync = true);

template<int NT, int VT, typename InputIt, typename T>
__device__ void DeviceGlobalToShared(int count, InputIt source, int tid,
	T* dest, bool sync = true);

template<int NT, int VT, typename InputIt, typename T>
__device__ void DeviceGlobalToSharedLoop(int count, InputIt source, int tid,
	T* dest, bool sync = true);

template<int VT, typename T>
__device__ void DeviceSharedToThread(const T* shared, int tid, T* threadReg,
	bool sync = true);
*/


} // namespace ReductionGPU


# endif
