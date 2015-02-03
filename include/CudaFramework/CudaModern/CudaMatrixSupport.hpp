#ifndef CudaFramework_CudaModern_CudaMatrixSupport_hpp
#define CudaFramework_CudaModern_CudaMatrixSupport_hpp

#include <cassert>
#include <algorithm>
#include <random>

#include "CudaFramework/General/AssertionDebug.hpp"

#include "CudaFramework/CudaModern/CudaTypeDefs.hpp"
#include "CudaFramework/CudaModern/CudaError.hpp"
#include "CudaFramework/CudaModern/CudaDeviceMatrix.hpp"
#include "CudaFramework/CudaModern/CudaDeviceMemory.hpp"
#include "CudaFramework/CudaModern/CudaDevice.hpp"
#include "CudaFramework/CudaModern/CudaMemSupport.hpp"


namespace utilCuda{

    class CudaContext;/******************************************************************************
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



    class CudaMatrixSupport : public CudaMemSupport {
        friend class CudaDevice;
        friend class CudaContext;
    public:


    template<typename PREC, bool AlignMatrix, bool RowMajor=false>
    DeviceMatrixPtr<PREC,RowMajor > mallocMatrix(size_t M, size_t N){

        ASSERTMSG( M > 0 && N > 0, "CudaMatrix<PREC> has wrong dimensions:" << M << ","<<N);

        typedef CudaDeviceMatrix<PREC,RowMajor> MatrixType;
        typedef DeviceMatrixPtr<PREC,RowMajor > DevicePtr;

        MatrixType * A_dev = new MatrixType(m_alloc.get(),M,N,0);

         A_dev->m_matrix.m_N = N;
         A_dev->m_matrix.m_M = M;

          size_t pitch;
          cudaError_t err;
          if(AlignMatrix){
             if(RowMajor){
                // If rowmajor!
                //err = cudaMallocPitch((void**) &A_dev.m_pDevice, &pitch,A_dev->N *sizeof(PREC), A_dev->M);
                err = m_alloc->mallocPitch( (void**)&A_dev->m_matrix.m_pDevice, &pitch, A_dev->m_matrix.m_N *sizeof(PREC), A_dev->m_matrix.m_M);
             }else{
                // If colmajor
                //err = cudaMallocPitch((void**) &A_dev.m_pDevice, &pitch,A_dev->M *sizeof(PREC), A_dev->N);
                err = m_alloc->mallocPitch( (void**)&A_dev->m_matrix.m_pDevice, &pitch, A_dev->m_matrix.m_M *sizeof(PREC), A_dev->m_matrix.m_N);
             }
          }else{
             // Do not align, just malloc!
             err = m_alloc->malloc( sizeof(PREC)*M*N, (void**)&A_dev->m_matrix.m_pDevice);
             if(RowMajor){
                pitch =A_dev->m_matrix.m_N*sizeof(PREC);
             }else{
                pitch =A_dev->m_matrix.m_M*sizeof(PREC);
             }
          }

      A_dev->m_matrix.m_outerStrideBytes = (unsigned int)pitch;

      ASSERTCHECK_CUDA(err);

      return DevicePtr(A_dev);
   }


    template<typename PREC, bool AlignMatrix, bool RowMajor=false, typename EigenMatrixType >
    DeviceMatrixPtr<PREC,RowMajor > mallocMatrix(EigenMatrixType & m){
        return mallocMatrix<PREC,AlignMatrix>(m.rows(),m.cols());
    }


    protected:
        CudaMatrixSupport() :  CudaMemSupport() { }
    };


};

#endif
