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

#ifndef CudaFramework_Kernels_JORProxVel_ReductionKernel_DeviceFunctions_BinarySearch_cuh
#define CudaFramework_Kernels_JORProxVel_ReductionKernel_DeviceFunctions_BinarySearch_cuh

#include <cuda_runtime.h>

#include "CudaFramework/Kernels/JORProxVel/ReductionKernel/DeviceUtilities/EnumsDevice.cuh"

namespace ReductionGPU {

template<MgpuBounds Bounds, typename IntT, typename It, typename T,
         typename Comp>
__device__ void binarySearchIt(It data, int& begin, int& end, T key,
                               int shift, Comp comp) {

    IntT scale = (1<< shift) - 1;
    int mid = (int)((begin + scale * end)>> shift);

    T key2 = data[mid];
    bool pred = (MgpuBoundsUpper == Bounds) ?
                !comp(key, key2) :
                comp(key2, key);
    if(pred) begin = mid + 1;
    else end = mid;
}


template<MgpuBounds Bounds, typename T, typename It, typename Comp>
__device__ int binarySearch(It data, int count, T key, Comp comp) {
    int begin = 0;
    int end = count;
    while(begin < end)
        binarySearchIt<Bounds, int>(data, begin, end, key, 1, comp);
    return begin;
}

} // namespace ReductionGPU
# endif
