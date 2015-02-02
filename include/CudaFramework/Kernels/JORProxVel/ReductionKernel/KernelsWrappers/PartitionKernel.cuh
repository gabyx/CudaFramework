
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
# ifndef PartitionKernel_cuh
# define PartitionKernel_cuh


#include <cuda_runtime.h>

#include "CudaError.hpp"

#include "GPUDefines.cuh"
#include "EnumsDevice.cuh"
#include "BinarySearch.cuh"
#include "DeviceOp.cuh"
#include "Tuning.hpp"
#include "Loadstore.cuh"
#include "CTAReduce.cuh"
#include "Enum.h"


namespace ReductionGPU {




template<int NT, typename CsrIt>
__global__ void partitionCsrSegRedKernel(int nz, int nv, CsrIt csrGlobal,
        int numRows, const int* numRows2, int numPartitions, int* limitsGlobal) {

    if(numRows2) numRows = *numRows2;

    int gid = NT * blockIdx.x + threadIdx.x;
    if(gid < numPartitions) {
        int key = min(nv * gid, nz);

        int ub;
        if(key == nz) ub = numRows;
        else {

            // Upper-bound search for this partition.
            //  MgpuBounds Bds=MgpuBoundsUpper;
            ub = binarySearch<MgpuBoundsUpper>(csrGlobal, numRows, key,ReductionGPU::less<int>()) - 1;

            // Check if limit points to matching value.
            if(key != csrGlobal[ub]) ub |= 0x80000000;

        }
        limitsGlobal[gid] = ub;
    }
}



template<typename CsrIt>
__host__ void partitionCsrSegRedWrap(   int count,
                                        int nv,
                                        CsrIt csrGlobal,
                                        int numRows,
                                        int numPartitions,
                                        cudaStream_t stream,
                                        int * limitsDevice,
                                        float* elapsedTime) {

    int numBlocks2 = MGPU_DIV_UP(numPartitions, 64);
    float timebuffer[1];
    cudaEvent_t start,stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start,0));

    partitionCsrSegRedKernel<64><<<numBlocks2, 64, 0, stream>>>(count,
                                                                nv,
                                                                csrGlobal,
                                                                numRows,
                                                                (const int*)0,
                                                                numPartitions,
                                                                limitsDevice);

    MGPU_SYNC_CHECK("partitionCsrSegRedKernel");

    CHECK_CUDA(cudaEventRecord(stop,0));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(timebuffer,start,stop));
    elapsedTime[0]+=timebuffer[0];

}


}

#endif // KernelMethods_cuh

