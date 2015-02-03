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

#ifndef CudaFramework_Kernels_JORProxVel_ReductionKernel_DeviceUtilities_Instantiations_cuh
#define CudaFramework_Kernels_JORProxVel_ReductionKernel_DeviceUtilities_Instantiations_cuh
#include <cuda_runtime.h>
#include "CudaFramework/Kernels/JORProxVel/ReductionKernel/DeviceUtilities/EnumsDevice.cuh"
// #include "device/deviceutil.cuh"  THIERRY

namespace ReductionGPU {


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

} // namespace ReductionGPU

//#include "loadstore.cuh"

# endif

