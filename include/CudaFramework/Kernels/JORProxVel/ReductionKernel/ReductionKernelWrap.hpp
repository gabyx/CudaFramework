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

//#define STATIC_ASSERT(COND,MSG) typedef char static_assertion_##MSG[(COND)?1:-1]
#ifndef CudaFramework_Kernels_JORProxVel_ReductionKernel_ReductionKernelWrap_hpp
#define CudaFramework_Kernels_JORProxVel_ReductionKernel_ReductionKernelWrap_hpp

#include "CudaFramework/CudaModern/CudaContext.hpp"
#include "CudaFramework/CudaModern/CudaTypeDefs.hpp"
#include "HostUtilities/Tuning.hpp"
#include "HostUtilities/TuningFunctions.hpp"
#include "HostUtilities/Static.hpp"
#include "CudaFramework/Kernels/JORProxVel/ReductionKernel/Enum.h"


namespace ReductionGPU {



template<typename CsrIt>
void partitionCsrSegRedWrap( int count,
                             int nv,
                             CsrIt csrGlobal,
                             int numRows,
                             int numPartitions,
                             cudaStream_t stream,
                             int * limitsDevice,
                             float* elapsedTime) ;


template<bool Indirect, typename CsrIt,typename PREC>
utilCuda::DeviceMemPtr<int> segReduceInnerFirst(CsrIt csrGlobal,
                                                int count,
                                                int numSegments,
                                                utilCuda::CudaContext& context,
                                                float* elapsedTime) {

    typedef typename SegReduceNormalTuning<sizeof(PREC)>::Tuning Tuning;
    //ReductionGPU::plus<int>() op;

    int2 launch = GetLaunchParams<Tuning>(context);
    int NV = launch.x * launch.y;
    /*
    printf("DEBUG: NT: launch.x: %i\n", launch.x);
    printf("DEBUG: VT: launch.y: %i\n", launch.y);

    //unsigned const int NV=21;

    printf("DEBUG: NV: %i \n",NV);
    //printf("DEBUG: sources_global: %p \n",sources_global);
*/
    int numBlocks = MGPU_DIV_UP(count, NV);
    /// get numblocks => (number of values -1)/ values per thread +1  (divide and round up)

/*
    printf("DEBUG: numBlocks: %i \n",numBlocks);
    */
    // Use upper-bound binary search to partition the CSR structure into tiles.

    ///
    ///     Partition Csr Seg Reduce   START
    ///

    // Allocate one int per partition.
    utilCuda::DeviceMemPtr<int> limitsDevice = context.malloc<int>(numBlocks + 1);

    int numBlocks2 = MGPU_DIV_UP(numBlocks + 1, 64);



    partitionCsrSegRedWrap(count,
                           NV,
                           csrGlobal,
                           numSegments,
                           numBlocks + 1,
                           context.stream(),
                           limitsDevice->get(),
                           elapsedTime);

    ASSERTCHECK_CUDA(cudaThreadSynchronize());
    ASSERTCHECK_CUDA_LAST

    ///
    ///     Partition Csr Seg Reduce   END
    ///
/*
    printf("DEBUG1: Limits: ");
    */
    //utilCuda::printArray(*limitsDevice,"%i",numBlocks + 1);

    return limitsDevice;

}



/**


template<int NT, typename T, typename DestIt>
__host__ void segRedSpineWrap1Kernel1(const int* limitsGlobal,
                                     int count,
                                     DestIt destGlobal,
                                     const T* carryInGlobal,
                                     T identity,
                                     T* carryOut_global,
                                     cudaStream_t stream,
                                     int numBlocks) ;


template<int NT, typename T, typename DestIt>
__host__ void segRedSpineWrap2( const int* limitsGlobal,
                                      int count,
                                      DestIt destGlobal,
                                      const T* carryInGlobal,
                                      T identity,
                                      cudaStream_t stream,
                                      int numBlocks) ;

 **/

template<bool indirect,typename Tuning,ReductionGPU::Operation::Type Op,typename PREC>
void segRedCsrWrap(    int launch_x,
                       int numBlocks,
                       int count,
                       int* csrGlobal,
                       cudaStream_t stream,
                       int * limitsDevice,
                       PREC* carryOutDevice,
                       PREC* destGlobal,
                       PREC* dataGlobal,
                       PREC identity,
                       float* elapsedTime);

template<typename PREC,ReductionGPU::Operation::Type Op>
void segRedSpineWrap(   const int* limitsGlobal,
                        int count,
                        PREC* destGlobal,
                        const PREC* carryInGlobal,
                        PREC identity,
                        cudaStream_t stream,
                        PREC* carry,
                        float* elapsedTime) ;

template<bool Indirect,ReductionGPU::Operation::Type Op,typename CsrIt, typename PREC>
void segReduceInnerSecond(  PREC* dataGlobal,
                            CsrIt csrGlobal,
                            int count,
                            PREC* destGlobal,
                            PREC identity,
                            utilCuda::CudaContext& context,
                            utilCuda::DeviceMemPtr<int>  limitsDevice,
                            float* elapsedTime) {


    typedef typename SegReduceNormalTuning<sizeof(PREC)>::Tuning Tuning;
    int2 launch = GetLaunchParams<Tuning>(context);
    int NV = launch.x * launch.y;
    int numBlocks = MGPU_DIV_UP(count, NV);

    //utilCuda::DeviceMemPtr<PREC> carryOutDevice = context.genRandom<PREC>(numBlocks,0,5);
    utilCuda::DeviceMemPtr<PREC> carryOutDevice = context.malloc<PREC>(numBlocks);
    segRedCsrWrap<false,Tuning,Op>(    launch.x,
                                       numBlocks,
                                       count,
                                       csrGlobal,
                                       context.stream(),
                                       limitsDevice->get(),
                                       carryOutDevice->get(),
                                       destGlobal,
                                       dataGlobal,
                                       identity,
                                       elapsedTime);

    ///
    ///    Kernel Seg Reduce SPine
    ///

    int numBlocks2 = MGPU_DIV_UP(count, 128);

    // Fix-up the segment outputs between the original tiles.
    utilCuda::DeviceMemPtr<PREC> carryOutDevice2 = context.malloc<PREC>(numBlocks2);

    segRedSpineWrap  <PREC,Op>( limitsDevice->get(),
                     numBlocks,
                     destGlobal,
                     carryOutDevice->get(),
                     identity,
                     context.stream(),
                     carryOutDevice2->get(),
                     elapsedTime);

    ///
    ///    Kernel Seg Reduce SPine  End
    ///
                            }


} // namespace ReductionGPU


#endif // reductionkernel
