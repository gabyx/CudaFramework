/******************************************************************************
* Copyright (c) 2013, NVIDIA CORPORATION. All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
* * Redistributions of source code must retain the above copyright
* notice, this list of conditions and the following disclaimer.
* * Redistributions in binary form must reproduce the above copyright
* notice, this list of conditions and the following disclaimer in the
* documentation and/or other materials provided with the distribution.
* * Neither the name of the NVIDIA CORPORATION nor the
* names of its contributors may be used to endorse or promote products
* derived from this software without specific prior written permission.
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
* 
*  Source code modified and extended from moderngpu.com
******************************************************************************/

#ifndef CudaFramework_CudaModern_CudaUtilities_hpp
#define CudaFramework_CudaModern_CudaUtilities_hpp

#include <stdio.h>
#include <iostream>
#include <fstream>

#include <cuda_runtime.h>


namespace utilCuda{

   void printAllCudaDeviceSpecs();
   void writeCudaDeviceProbs(std::ostream & file, const cudaDeviceProp & probs, int devicenumber);
   void printCudaDeviceProbs(const cudaDeviceProp & probs, int devicenumber);


   namespace memoryTransferTest {
       typedef struct{
          static const int default_size = ( 32 * ( 1 << 20 ) ); //32 M
          static const int memcpyIterations = 100;
          static const int cacheClearSize = 1 << 24;
          static const bool bDontUseGPUTiming = true;
          enum memoryMode { PINNED, PAGEABLE };
       } bandwithData;

       void measureCopySpeeds();
       double testHostToDeviceTransfer(unsigned int memSize, bandwithData::memoryMode memMode, bool wc);
       double testDeviceToHostTransfer(unsigned int memSize, bandwithData::memoryMode memMode, bool wc);
       double testMalloc(unsigned int memSize);
   }

}


#endif
