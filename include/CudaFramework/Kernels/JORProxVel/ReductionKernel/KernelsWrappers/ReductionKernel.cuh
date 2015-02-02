
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
# ifndef ReductionKernel_cuh
# define ReductionKernel_cuh


#include <cuda_runtime.h>

#include "CudaError.hpp"
#include "EnumsDevice.cuh"
#include "DeviceOp.cuh"
#include "Loadstore.cuh"
#include "CTAReduce.cuh"
#include "Enum.h"


namespace ReductionGPU {

template<int NT, int VT, bool HalfCapacity, bool LdgTranspose, typename T>
struct CTASegReduceLoad {
    enum {
        NV = NT * VT,
        Capacity = HalfCapacity ? (NV / 2) : NV
    };

    union Storage {
        int sources[NV];
        T data[Capacity];
    };

    // Load elements from multiple segments and store in thread order.
    template<typename InputIt>
    __device__ static void loadDirect(int count2,
                                      int tid,
                                      int gid,
                                      InputIt dataGlobal,
                                      T identity,
                                      T data[VT],
                                      Storage& storage) {

        if(LdgTranspose) {
            // Load data in thread order from dataGlobal + gid.
            deviceGlobalToThreadDefault<NT, VT>(count2,
                                                dataGlobal + gid,
                                                tid,
                                                data,
                                                identity);
        } else {
            // Load data in strided order from dataGlobal + gid.
            T stridedData[VT];
            DeviceGlobalToRegDefault<NT, VT>(count2,
                                             dataGlobal + gid,
                                             tid,
                                             stridedData,
                                             identity);

            if(HalfCapacity)
                HalfSmemTranspose<NT, VT>(stridedData,
                                          tid,
                                          storage.data,
                                          data);
            else {
                DeviceRegToShared<NT, VT>(stridedData,
                                          tid,
                                          storage.data);

                DeviceSharedToThread<VT>(storage.data,
                                         tid,
                                         data);
            }
        }
    }

};



template<typename Tuning, bool Indirect, typename CsrIt,
         typename InputIt, typename DestIt, typename T, typename Op>
MGPU_LAUNCH_BOUNDS void segRedCsrKernel(    CsrIt csrGlobal,
                                            int count,
                                            const int* limitsGlobal,
                                            InputIt dataGlobal,
                                            T identity,
                                            Op op,
                                            DestIt destGlobal,
                                            T* carryOut_global)
{




    typedef MGPU_LAUNCH_PARAMS Params;

    const int NT = Params::NT;
    const int VT = Params::VT;
    const int NV = NT * VT;
    const bool HalfCapacity = (sizeof(T) > sizeof(int)) && Params::HalfCapacity;
    const bool LdgTranspose = Params::LdgTranspose;

    typedef CTAReduce<NT, Op> FastReduce;
    typedef CTASegReduce<NT, VT, HalfCapacity, T, Op> SegReduce;

    typedef CTASegReduceLoad<NT, VT, HalfCapacity, LdgTranspose, T> SegReduceLoad;


    union Shared {
        int csr[NV + 1];
        typename FastReduce::Storage reduceStorage;
        typename SegReduce::Storage segReduceStorage;
        typename SegReduceLoad::Storage loadStorage;
    };


    __shared__ Shared shared;

    int tid = threadIdx.x;
    int block = blockIdx.x;
    int gid = NV * block;
    int count2 = min(NV, count - gid);

    int limit0 = limitsGlobal[block];
    int limit1 = limitsGlobal[block + 1];

    SegReduceRange range;
    SegReduceTerms terms;

    int segs[VT + 1], segStarts[VT];
    T data[VT];

    ///                          ///
    ///   Deleted Indirect Load  ///
    ///                          ///

    {
        // Direct load. It is more efficient to load the full tile before
        // dealing with data dependencies.


        SegReduceLoad::loadDirect(count2,
                                  tid,
                                  gid,
                                  dataGlobal,
                                  identity,
                                  data,
                                  shared.
                                  loadStorage);

        range = deviceShiftRange(limit0, limit1);
        int numSegments = range.end - range.begin;

        if(range.total) {
            // Load the CSR interval.
            DeviceGlobalToSharedLoop<NT, VT>(numSegments,
                                             csrGlobal + range.begin,
                                             tid,
                                             shared.csr);
            // Compute the segmented scan terms.
            terms = DeviceSegReducePrepare<NT, VT>(shared.csr,
                                                   numSegments,
                                                   tid,
                                                   gid,
                                                   range.
                                                   flushLast,
                                                   segs,
                                                   segStarts);
        }
    }

    if(range.total) {
        // Reduce tile data and store to destGlobal. Write tile's carry-out
        // term to carryOut_global.segReduceInnerSecond
        SegReduce::reduceToGlobal(segs,
                                  range.total,
                                  terms.tidDelta,
                                  range.begin,
                                  block,
                                  tid,
                                  data,
                                  destGlobal,
                                  carryOut_global,
                                  identity,
                                  op,
                                  shared.segReduceStorage);
    } else {
        T x;
#pragma unroll
        for(int i = 0; i < VT; ++i)
            x = i ? op(x, data[i]) : data[i];
        x = FastReduce::Reduce(tid, x, shared.reduceStorage, op);
        if(!tid)
            carryOut_global[block] = x;
    }
}


template<bool indirect,typename Tuning, ReductionGPU::Operation::Type Op,typename PREC>
__host__ void segRedCsrWrap ( int launch_x,
                              int numBlocks,
                              int count,
                              int* csrGlobal,
                              cudaStream_t stream,
                              int* limitsDevice,
                              PREC* carryOutDevice,
                              PREC* destGlobal,
                              PREC* dataGlobal,
                              PREC identity,
                              float* elapsedTime) {


if (Op==ReductionGPU::Operation::PLUS){
    segRedCsrKernel<Tuning, indirect>
    <<<numBlocks, launch_x, 0, stream>>>(  csrGlobal,
                                           count,
                                           limitsDevice,
                                           dataGlobal,
                                           identity,
                                           ReductionGPU::plus<PREC>(),
                                           destGlobal,
                                           carryOutDevice);

    //CHECK_CUDA_LAST
    MGPU_SYNC_CHECK("segRedCsrKernel");
}
  CHECK_CUDA_LAST

}

}

#endif // KernelMethods_cuh

