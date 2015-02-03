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

#ifndef CudaFramework_Kernels_JORProxVel_ReductionKernel_KernelsWrappers_ReduceSpineKernels_cuh
#define CudaFramework_Kernels_JORProxVel_ReductionKernel_KernelsWrappers_ReduceSpineKernels_cuh

#include "DeviceUtilities/GPUDefines.cuh"
#include "DeviceUtilities/DeviceOp.cuh"

#include "CudaFramework/Kernels/JORProxVel/ReductionKernel/Enum.h"
namespace ReductionGPU {

////////////////////////////////////////////////////////////////////////////////
// segRedSpineWrap
// Compute the carry-in in-place. Return the carry-out for the entire tile.
// A final spine-reducer scans the tile carry-outs and adds into individual
// results.

template<int NT, typename T, typename DestIt, typename Op>
__global__ void segRedSpineKernel1(   const int* limitsGlobal,
                                      int count,
                                      DestIt destGlobal,
                                      const T* carryInGlobal,
                                      T identity,
                                      Op op,
                                      T* carryOut_global) {

    typedef CTASegScan<NT, Op> SegScan;

    union Shared {
        typename SegScan::Storage segScanStorage;
    };

    __shared__ Shared shared;

    int tid = threadIdx.x;
    int block = blockIdx.x;
    int gid = NT * block + tid;

    // Load the current carry-in and the current and next row indices.
    int row = (gid < count) ?
              (0x7fffffff & limitsGlobal[gid]) :
              INT_MAX;
    int row2 = (gid + 1 < count) ?
               (0x7fffffff & limitsGlobal[gid + 1]) :
               INT_MAX;

    T carryIn2 = (gid < count) ? carryInGlobal[gid] : identity;
    T dest = (gid < count) ? destGlobal[row] : identity;

    // Run a segmented scan of the carry-in values.
    bool endFlag = row != row2;

    T carryOut;
    T x = SegScan::SegScan(tid,
                           carryIn2,
                           endFlag,
                           shared.segScanStorage,
                           &carryOut,
                           identity,
                           op);

    // Store the reduction at the end of a segment to destGlobal.
    if(endFlag)
        destGlobal[row] = op(x, dest);

    // Store the CTA carry-out.
    if(!tid) carryOut_global[block] = carryOut;

}

template<int NT, typename T, typename DestIt, typename Op>
__global__ void segRedSpineKernel2(const int* limitsGlobal,
                                   int numBlocks,
                                   int count,
                                   int nv,
                                   DestIt destGlobal,
                                   const T* carryInGlobal,
                                   T identity,
                                   Op op) {
    //dest_globaldest_global[0]=100;
    typedef CTASegScan<NT, Op> SegScan;
    struct Shared {
        typename SegScan::Storage segScanStorage;
        int carryInRow;
        T carryIn;
    };
    __shared__ Shared shared;

    int tid = threadIdx.x;

    for(int i = 0; i < numBlocks; i += NT) {
        int gid = (i + tid) * nv;

        // Load the current carry-in and the current and next row indices.
        int row = (gid < count) ?
                  (0x7fffffff & limitsGlobal[gid]) : INT_MAX;
        int row2 = (gid + nv < count) ?
                   (0x7fffffff & limitsGlobal[gid + nv]) : INT_MAX;
        T carryIn2 = (i + tid < numBlocks) ? carryInGlobal[i + tid] : identity;
        T dest = (gid < count) ? destGlobal[row] : identity;

        // Run a segmented scan of the carry-in values.
        bool endFlag = row != row2;

        T carryOut;
        T x = SegScan::SegScan(tid,
                               carryIn2,
                               endFlag,
                               shared.segScanStorage,
                               &carryOut,
                               identity,
                               op);

        // Add the carry-in to the reductions when we get to the end of a segment.
        if(endFlag) {
            // Add the carry-in from the last loop iteration to the carry-in
            // from this loop iteration.
            if(i && row == shared.carryInRow)
                x = op(shared.carryIn, x);
            destGlobal[row] = op(x, dest);
        }

        // Set the carry-in for the next loop iteration.
        if(i + NT < numBlocks) {
            __syncthreads();
            if(i > 0) {
                // Add in the previous carry-in.
                if(NT - 1 == tid) {
                    shared.carryIn = (shared.carryInRow == row2) ?
                                     op(shared.carryIn, carryOut) : carryOut;
                    shared.carryInRow = row2;
                }
            } else {
                if(NT - 1 == tid) {
                    shared.carryIn = carryOut;
                    shared.carryInRow = row2;
                }
            }
            __syncthreads();
        }
    }
}
/**

template<int NT,typename T, typename DestIt>
__host__ void segRedSpineWrap1Kernel1( const int* limitsGlobal,
                                       int count,
                                       DestIt destGlobal,
                                       const T* carryInGlobal,
                                       T identity,
                                       T* carryOut_global,
                                       cudaStream_t stream,
                                       int numBlocks)  {

    segRedSpineKernel2<NT><<<numBlocks,NT,0,stream>>>(limitsGlobal,
                                                      count,
                                                      destGlobal,
                                                      carryInGlobal,
                                                      identity,
                                                      ReductionGPU::plus<int>(),
                                                      carryOut_global);

    MGPU_SYNC_CHECK("segRedSpineKernel2");



}

template<int NT,typename T, typename DestIt>
__host__ void segRedSpineWrap2( const int* limitsGlobal,
                                int count,
                                DestIt destGlobal,
                                const T* carryInGlobal,
                                T identity,
                                cudaStream_t stream,
                                int numBlocks)  {

    segRedSpineKernel2<NT><<<1, NT, 0, stream>>>(limitsGlobal,
                                                 numBlocks,
                                                 count,
                                                 NT,
                                                 destGlobal,
                                                 carryInGlobal,
                                                 identity,
                                                 ReductionGPU::plus<int>());

    MGPU_SYNC_CHECK("segRedSpineKernel2");



}
**/

template<typename PREC, ReductionGPU::Operation::Type Op>
__host__ void segRedSpineWrap( const int* limitsGlobal,
                               int count,
                               PREC* destGlobal,
                               const PREC* carryInGlobal,
                               PREC identity,
                               cudaStream_t stream,
                               PREC* carry,
                               float* elapsedTime) {

    const int NT = 128;

    int numBlocks = MGPU_DIV_UP(count, NT);
    if(Op==ReductionGPU::Operation::PLUS){
    // Fix-up the segment outputs between the original tiles.



    segRedSpineKernel1<NT><<<numBlocks, NT, 0,stream>>>(limitsGlobal,
                                                        count,
                                                        destGlobal,
                                                        carryInGlobal,
                                                        identity,
                                                        ReductionGPU::plus<PREC>(),
                                                        carry);
    MGPU_SYNC_CHECK("segRedSpineKernel1");

    // Loop over the segments that span the tiles of
    // segRedSpineKernel2 and fix those.
    if(numBlocks > 1) {
        segRedSpineKernel2<NT><<<1, NT, 0, stream>>>( limitsGlobal,
                                                      numBlocks,
                                                      count,
                                                      NT,
                                                      destGlobal,
                                                      carry,
                                                      identity,
                                                      ReductionGPU::plus<PREC>());
        MGPU_SYNC_CHECK("segRedSpineKernel2");
    }

    }
   CHECK_CUDA_LAST
}

} // namespace ReductionGPU

#endif

