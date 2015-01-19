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

#ifndef CudaFramework_CudaModern_CudaDeviceMatrix_hpp
#define CudaFramework_CudaModern_CudaDeviceMatrix_hpp

#include <cuda_runtime.h>
#include <driver_types.h>

#include <Eigen/Dense>
#include "CudaFramework/General/StaticAssert.hpp"
#include "CudaFramework/CudaModern/CudaTypeDefs.hpp"
#include "CudaFramework/CudaModern/CudaRefcounting.hpp"
#include "CudaFramework/CudaModern/CudaAlloc.hpp"
#include "CudaFramework/CudaModern/CudaDeviceMemory.hpp"
#include "CudaFramework/CudaModern/CudaMatrix.hpp"

namespace utilCuda {


template<typename PREC, bool RowMajor>
class CudaDeviceMatrix : public ReferenceCounting<void,CudaDeviceMatrix<PREC,RowMajor> >  {

    friend class CudaMatrixSupport;
    enum {
        Flags = (RowMajor)? CudaMatrixFlags::RowMajorBit : 0
    };

public:

    /** Get the underlying struct which can be passed to a kernel function */
    CudaMatrix<PREC,Flags> & get(){ return  m_matrix;}
    const CudaMatrix<PREC,Flags> & get() const{ return  m_matrix;}

    // Size is in units of bytes.
    size_t size() const { return RowMajor? ( m_matrix.m_outerStrideBytes*m_matrix.m_M ) : (m_matrix.m_outerStrideBytes*m_matrix.m_N); }


    // Overloads (from CudaDeviceMem)
    // Copy from this to the argument array.
    /**
    * For outerStrideBytes see http://eigen.tuxfamily.org/dox/classEigen_1_1Stride.html
    */
    // Standard for ColMajor host matrix
    cudaError_t toHost(PREC * target, size_t M, size_t N) const {
        return toHost(target,M,N,sizeof(PREC)*M);
    }

    cudaError_t toHost(PREC * target, size_t M, size_t N, size_t outerStrideBytes) const {

        ASSERTMSG(m_matrix.m_N*m_matrix.m_M == M*N, "Bytes to copy to to host has wrong dimensions (" << m_matrix.m_M << "," << m_matrix.m_N << " != " << M<<","<<N <<")");
        ASSERTMSG(m_matrix.m_pDevice != nullptr ,"Device memory not allocated!");

        cudaError_t err;
        if(RowMajor) {
            // If rowmajor!
            err = cudaMemcpy2D(target, outerStrideBytes, m_matrix.m_pDevice, m_matrix.m_outerStrideBytes,  N*sizeof(PREC), M, cudaMemcpyDeviceToHost);
        } else {
            // If colmajor!
            err = cudaMemcpy2D(target, outerStrideBytes, m_matrix.m_pDevice, m_matrix.m_outerStrideBytes,  M*sizeof(PREC), N, cudaMemcpyDeviceToHost);
        }
        return err;
    }

    cudaError_t toHost(Eigen::Ref< Eigen::Matrix<PREC,Eigen::Dynamic,Eigen::Dynamic> > target) const {

        typedef Eigen::Matrix<PREC,Eigen::Dynamic,Eigen::Dynamic> Matrix;

        ASSERTMSG(target.innerStride() == 1, "InnerStride not 1, should be to avoid strange data alignment");
        STATIC_ASSERTM( ((Matrix::Flags & Eigen::RowMajorBit) == (RowMajor)), ROWMAJOR_COLMAJOR_INCONSITENCY );
        ASSERTMSG( m_matrix.m_N == target.cols() && m_matrix.m_M == target.rows() , "Matrix to copy to device has wrong dimensions (" << m_matrix.m_M << "," << m_matrix.m_N << ") to (" << target.rows()<<","<<target.cols()<<")");

        cudaError_t err;
        if(RowMajor) {
            // If rowmajor!
            err = cudaMemcpy2D(target.data(), (size_t)target.outerStride()*sizeof(PREC), m_matrix.m_pDevice , m_matrix.m_outerStrideBytes,  (size_t)target.cols()*sizeof(PREC), (size_t)target.rows(), cudaMemcpyDeviceToHost);
        } else {
            // If colmajor!
            err = cudaMemcpy2D(target.data(), (size_t)target.outerStride()*sizeof(PREC), m_matrix.m_pDevice , m_matrix.m_outerStrideBytes,  (size_t)target.rows()*sizeof(PREC), (size_t)target.cols(), cudaMemcpyDeviceToHost);
        }
        return err;
    }


    cudaError_t fromHost(const PREC* data, size_t M, size_t N, size_t outerStrideBytes) {

        ASSERTMSG( m_matrix.m_N == N && m_matrix.m_M == M , "Copy to device has wrong dimensions (" << m_matrix.m_M << "," << m_matrix.m_N << " to (" << M<<","<<N<<")");
        ASSERTMSG( m_matrix.m_pDevice != nullptr && data != nullptr, "Device memory or Host memory not allocated!");

        cudaError_t err;
        if(RowMajor) {
            // If rowmajor!
            err = cudaMemcpy2D(m_matrix.m_pDevice, m_matrix.m_outerStrideBytes, data, outerStrideBytes,  m_matrix.m_N*sizeof(PREC), m_matrix.m_M, cudaMemcpyHostToDevice);
        } else {
            // If colmajor!
            err = cudaMemcpy2D(m_matrix.m_pDevice, m_matrix.m_outerStrideBytes, data, outerStrideBytes,  m_matrix.m_M*sizeof(PREC), m_matrix.m_N, cudaMemcpyHostToDevice);
        }

        return err;
    }

    cudaError_t fromHost(const Eigen::Ref<const Eigen::Matrix<PREC,Eigen::Dynamic,Eigen::Dynamic> > src) {

        typedef Eigen::Matrix<PREC,Eigen::Dynamic,Eigen::Dynamic> Matrix;

        ASSERTMSG(src.innerStride() == 1, "InnerStride not 1, should be to avoid strange data alignment");
        STATIC_ASSERTM( (Matrix::Flags & Eigen::RowMajorBit) == (RowMajor),ROWMAJOR_COLMAJOR_INCONSITENCY);
        ASSERTMSG( m_matrix.m_pDevice != nullptr,"Device memory not allocated!");
        ASSERTMSG( m_matrix.m_N == src.cols() && m_matrix.m_M == src.rows() , "Matrix to copy to device has wrong dimensions (" << m_matrix.m_M << "," << m_matrix.m_N << ") to (" << src.rows()<<","<<src.cols()<<")");

        cudaError_t err;
        if(RowMajor) {
            // If rowmajor!
            err = cudaMemcpy2D(m_matrix.m_pDevice, m_matrix.m_outerStrideBytes, src.data(), (size_t)src.outerStride()*sizeof(PREC),  m_matrix.m_N*sizeof(PREC), m_matrix.m_M, cudaMemcpyHostToDevice);
        } else {
            // If colmajor!
            err = cudaMemcpy2D(m_matrix.m_pDevice, m_matrix.m_outerStrideBytes, src.data(), (size_t)src.outerStride()*sizeof(PREC),  m_matrix.m_M*sizeof(PREC), m_matrix.m_N,cudaMemcpyHostToDevice);
        }
        return err;
    }


    ~CudaDeviceMatrix(){
        //CUDA_DESTRUCTOR_MESSAGE(this);
        m_alloc->free(m_matrix.m_pDevice);
    }

    private:

    CudaDeviceMatrix(CudaAlloc* alloc, unsigned int M, unsigned int N, unsigned int outerStrideBytes)
        : m_alloc(alloc), m_matrix(M,N,outerStrideBytes)
    {};

    AllocPtrType m_alloc;

    CudaMatrix<PREC,Flags> m_matrix;

};



}



#endif
