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



#ifndef Reduction_hpp
#define Reduction_hpp


#include "ReductionKernelWrap.hpp"
#include "Enum.h"



template<ReductionGPU::Operation::Type Op,typename CsrIt, typename PREC>
void segReduceCsr( PREC* dataGlobal,
                   CsrIt csrGlobal,
                   int count,
                   int numSegments,
                   bool supportEmpty,
                   PREC* destGlobal,
                   PREC identity,
                   utilCuda::CudaContext& context,
                   float* elapsedTime) {

    utilCuda::DeviceMemPtr<int> limitsDevice=ReductionGPU::segReduceInnerFirst<false,int*,PREC>(   csrGlobal,
                                                                                           count,
                                                                                           numSegments,
                                                                                           context,
                                                                                           elapsedTime);




    ReductionGPU::segReduceInnerSecond<false,Op>( dataGlobal,
                                       csrGlobal,
                                       count,
                                       destGlobal,
                                       identity,
                                       context,
                                       limitsDevice,
                                       elapsedTime);

}

// namespace ReductionGPU



#endif // Reduction_hpp
