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

#ifndef CudaFramework_CudaModern_CudaMatrixUtilities_hpp
#define CudaFramework_CudaModern_CudaMatrixUtilities_hpp

#include <cuda_runtime.h>

#include <type_traits>

#include <Eigen/Dense>

#include "CudaFramework/General/StaticAssert.hpp"
#include "CudaFramework/General/AssertionDebug.hpp"
#include "CudaFramework/CudaModern/CudaMatrix.hpp"
#include "CudaFramework/General/Utilities.hpp"
#include "CudaFramework/CudaModern/CudaError.hpp"


namespace utilCuda {


template<bool AlignMatrix, typename TCudaMatrix>
cudaError_t mallocMatrixDevice(TCudaMatrix & A_dev, int M,int N) {

    typedef typename TCudaMatrix::PREC PREC;

    ASSERTMSG( M > 0 && N > 0, "CudaMatrix<PREC> has wrong dimensions:" << M << ","<<N);
    ASSERTMSG( A_dev.m_pDevice == nullptr , "A_dev.m_pDevice is NOT NULL, we loose this pointer if we malloc!");

    A_dev.m_N = N;
    A_dev.m_M = M;

    size_t pitch;
    cudaError_t err;
    if(AlignMatrix) {
        if(CudaMatrixFlags::isRowMajor<TCudaMatrix::Flags>::value) {
            // If rowmajor!
            err = cudaMallocPitch((void**) &A_dev.m_pDevice, &pitch, A_dev.m_N *sizeof(PREC),  A_dev.m_M);
        } else {
            // If colmajor!
            err = cudaMallocPitch((void**) &A_dev.m_pDevice, &pitch, A_dev.m_M *sizeof(PREC),  A_dev.m_N);
        }
    } else {
        // Do not align, just malloc!
        err = cudaMalloc((void**)&A_dev.m_pDevice, (A_dev.m_M * A_dev.m_N)*sizeof(PREC));
        if(CudaMatrixFlags::isRowMajor<TCudaMatrix::Flags>::value) {
            pitch = A_dev.m_N*sizeof(PREC);
        } else {
            pitch = A_dev.m_M*sizeof(PREC);
        }
    }

    A_dev.m_outerStrideBytes = (unsigned int)pitch;
    return err;
}

template<bool AlignMatrix, typename TCudaMatrix>
cudaError_t releaseAndMallocMatrixDevice(TCudaMatrix & A_dev, int M, int N) {
    freeMatrixDevice(A_dev);
    return mallocMatrixDevice<AlignMatrix>(A_dev,M,N);
}

template<bool AlignMatrix, typename TCudaMatrix, typename Derived>
cudaError_t mallocMatrixDevice(TCudaMatrix & A_dev, const Eigen::MatrixBase<Derived> & A) {
    typedef typename TCudaMatrix::PREC PREC;
    STATIC_ASSERTM( (std::is_same<typename TCudaMatrix::PREC, typename Derived::Scalar>::value), YOU_MIXED_DIFFERENT_PRECISIONS)
    STATIC_ASSERTM( (CudaMatrixFlags::isRowMajor<TCudaMatrix::Flags>::value == (Derived::Flags & Eigen::RowMajorBit)) , EIGEN_AND_CUDAMATRIX_HAVE_DIFFERENT_LAYOUTS)

    ASSERTMSG( A_dev.m_pDevice == nullptr , "A_dev.m_pDevice is NOT NULL, we loose this pointer if we malloc!");

    A_dev.m_N = (unsigned int)A.cols();
    A_dev.m_M = (unsigned int)A.rows();

    size_t pitch;
    cudaError_t err;
    if(AlignMatrix && (A_dev.m_N != 1 && A_dev.m_M !=1) ) {
        if(CudaMatrixFlags::isRowMajor<TCudaMatrix::Flags>::value) {
            // If rowmajor!
            err = cudaMallocPitch((void**) &A_dev.m_pDevice, &pitch, A_dev.m_N *sizeof(PREC),  A_dev.m_M);
        } else {
            // If colmajor!
            err = cudaMallocPitch((void**) &A_dev.m_pDevice, &pitch, A_dev.m_M *sizeof(PREC),  A_dev.m_N);
        }
    } else {
        // Do not align, just malloc!
        err = cudaMalloc((void**)&A_dev.m_pDevice, (A_dev.m_M * A_dev.m_N)*sizeof(PREC));
        if(CudaMatrixFlags::isRowMajor<TCudaMatrix::Flags>::value) {
            pitch = A_dev.m_N*sizeof(PREC);
        } else {
            pitch = A_dev.m_M*sizeof(PREC);
        }
    }

    A_dev.m_outerStrideBytes = (unsigned int)pitch;
    return err;
}

template<bool AlignMatrix, typename TCudaMatrix, typename Derived>
cudaError_t releaseAndMallocMatrixDevice(TCudaMatrix & A_dev, const Eigen::MatrixBase<Derived> & A) {
    typedef typename TCudaMatrix::PREC PREC;
    STATIC_ASSERTM( (std::is_same<typename TCudaMatrix::PREC, typename Derived::Scalar>::value), YOU_MIXED_DIFFERENT_PRECISIONS)
    STATIC_ASSERTM( (CudaMatrixFlags::isRowMajor<TCudaMatrix::Flags>::value == (Derived::Flags & Eigen::RowMajorBit)) , EIGEN_AND_CUDAMATRIX_HAVE_DIFFERENT_LAYOUTS)

    // Only tries to free matrix!
    freeMatrixDevice(A_dev);
    return mallocMatrixDevice<AlignMatrix>(A_dev,A);
}



template<typename TCudaMatrix>
cudaError_t freeMatrixDevice(TCudaMatrix & A_dev) {

    cudaError_t e = cudaSuccess;
    if(A_dev.m_pDevice) {
        e = cudaFree(A_dev.m_pDevice);
        A_dev.m_pDevice = nullptr;
    }
    return e;
}
template<typename TCudaMatrix>
cudaError_t copyMatrixToDevice(TCudaMatrix & target_dev, const TCudaMatrix & src) {
    typedef typename TCudaMatrix::PREC PREC;
    ASSERTMSG( target_dev.m_N == src.m_N && target_dev.m_M == src.m_M , "CudaMatrix to copy to device has wrong dimensions (" << target_dev.m_M << "," << target_dev.m_N << " to (" << src.m_M<<","<<src.m_N<<")");
    ASSERTMSG( target_dev.m_pDevice != nullptr && src.m_pDevice != nullptr,"target_dev or src not allocated!");

    cudaError_t err;
    if(CudaMatrixFlags::isRowMajor<TCudaMatrix::Flags>::value) {
        // If rowmajor!
        err = cudaMemcpy2D(target_dev.m_pDevice, target_dev.m_outerStrideBytes, src.m_pDevice, (size_t)src.m_outerStrideBytes,  (size_t)target_dev.m_N*sizeof(PREC), (size_t)target_dev.m_M, cudaMemcpyHostToDevice);
    } else {
        // If colmajor!
        err = cudaMemcpy2D(target_dev.m_pDevice, target_dev.m_outerStrideBytes, src.m_pDevice, (size_t)src.m_outerStrideBytes,  (size_t)target_dev.m_M*sizeof(PREC), (size_t)target_dev.m_N, cudaMemcpyHostToDevice);
    }

    return err;
}

template<typename TCudaMatrix, typename Derived>
cudaError_t copyMatrixToDevice(TCudaMatrix & target_dev, const Eigen::MatrixBase<Derived> & src) {

    typedef typename TCudaMatrix::PREC PREC;
    STATIC_ASSERTM( (std::is_same<typename TCudaMatrix::PREC, typename Derived::Scalar>::value), YOU_MIXED_DIFFERENT_PRECISIONS)
    STATIC_ASSERTM( (CudaMatrixFlags::isRowMajor<TCudaMatrix::Flags>::value == (Derived::Flags & Eigen::RowMajorBit)) , EIGEN_AND_CUDAMATRIX_HAVE_DIFFERENT_LAYOUTS)
    ASSERTMSG( target_dev.m_pDevice != NULL,"target_dev not allocated!");
    ASSERTMSG( target_dev.m_N == src.cols() && target_dev.m_M == src.rows() , "Matrix to copy to device has wrong dimensions (" << target_dev.m_M << "," << target_dev.m_N << ") to (" << src.rows()<<","<<src.cols()<<")");

    cudaError_t err;
    if(CudaMatrixFlags::isRowMajor<TCudaMatrix::Flags>::value) {
        // If rowmajor!
        err = cudaMemcpy2D(target_dev.m_pDevice, target_dev.m_outerStrideBytes, &src.operator()(0,0), (size_t)src.outerStride()*sizeof(PREC),  (size_t)target_dev.m_N*sizeof(PREC), (size_t)target_dev.m_M, cudaMemcpyHostToDevice);
    } else {
        // If colmajor!
        err = cudaMemcpy2D(target_dev.m_pDevice, target_dev.m_outerStrideBytes, &src.operator()(0,0), (size_t)src.outerStride()*sizeof(PREC),  (size_t)target_dev.m_M*sizeof(PREC), (size_t)target_dev.m_N,cudaMemcpyHostToDevice);
    }

    return err;
}

template<typename TCudaMatrix>
cudaError_t copyMatrixDeviceToDevice(TCudaMatrix & target_dev, const TCudaMatrix & src_dev) {

    typedef typename TCudaMatrix::PREC PREC;
    ASSERTMSG( target_dev.m_N == src_dev.m_N && target_dev.m_M == src_dev.m_M , "CudaMatrix<PREC> to copy to device has wrong dimensions (" << target_dev.m_M << "," << target_dev.m_N << " to (" << src_dev.m_M<<","<<src_dev.m_N<<")");
    ASSERTMSG(target_dev.m_pDevice != nullptr && src_dev.m_pDevice != nullptr,"target_dev or src_dev not allocated!");

    cudaError_t err;
    if(CudaMatrixFlags::isRowMajor<TCudaMatrix::Flags>::value) {
        // If rowmajor!
        err = cudaMemcpy2D(target_dev.m_pDevice, target_dev.m_outerStrideBytes, &src_dev.operator()(0,0), (size_t)src_dev.m_outerStrideBytes,  (size_t)target_dev.m_N*sizeof(PREC), (size_t)target_dev.m_M, cudaMemcpyDeviceToDevice);
    } else {
        // If colmajor!
        err = cudaMemcpy2D(target_dev.m_pDevice, target_dev.m_outerStrideBytes, &src_dev.operator()(0,0), (size_t)src_dev.m_outerStrideBytes,  (size_t)target_dev.m_M*sizeof(PREC), (size_t)target_dev.m_N, cudaMemcpyDeviceToDevice);
    }

    return err;
}

template<bool AlignMatrix, typename TCudaMatrix, typename Derived>
cudaError_t mallocAndCopyMatrixToDevice(TCudaMatrix & target_dev, const Eigen::MatrixBase<Derived> & src) {

    STATIC_ASSERTM( (std::is_same<typename TCudaMatrix::PREC, typename Derived::Scalar>::value), YOU_MIXED_DIFFERENT_PRECISIONS)
    STATIC_ASSERTM( (CudaMatrixFlags::isRowMajor<TCudaMatrix::Flags>::value == (Derived::Flags & Eigen::RowMajorBit)) , EIGEN_AND_CUDAMATRIX_HAVE_DIFFERENT_LAYOUTS)

    CHECK_CUDA(mallocMatrixDevice<AlignMatrix>(target_dev,src));

    return copyMatrixToDevice(target_dev,src);
}

template<typename TCudaMatrix, typename Derived>
cudaError_t copyMatrixToHost(Eigen::MatrixBase<Derived> & target, const TCudaMatrix & src_dev) {

    typedef typename TCudaMatrix::PREC PREC;
    STATIC_ASSERTM( (std::is_same<typename TCudaMatrix::PREC, typename Derived::Scalar>::value), YOU_MIXED_DIFFERENT_PRECISIONS)
    STATIC_ASSERTM( (CudaMatrixFlags::isRowMajor<TCudaMatrix::Flags>::value == (Derived::Flags & Eigen::RowMajorBit)) , EIGEN_AND_CUDAMATRIX_HAVE_DIFFERENT_LAYOUTS)

    ASSERTMSG( src_dev.m_N == target.cols() && src_dev.m_M == target.rows() , "Matrix to copy to device has wrong dimensions (" << src_dev.m_M << "," << src_dev.m_N << ") to (" << target.rows()<<","<<target.cols()<<")");

    cudaError_t err;
    if(CudaMatrixFlags::isRowMajor<TCudaMatrix::Flags>::value) {
        // If rowmajor!
        err = cudaMemcpy2D(&(target.operator()(0,0)), (size_t)target.outerStride()*sizeof(PREC), src_dev.m_pDevice, (size_t)src_dev.m_outerStrideBytes,  (size_t)target.cols()*sizeof(PREC), (size_t)target.rows(), cudaMemcpyDeviceToHost);
    } else {
        // If colmajor!
        err = cudaMemcpy2D(&(target.operator()(0,0)), (size_t)target.outerStride()*sizeof(PREC), src_dev.m_pDevice, (size_t)src_dev.m_outerStrideBytes,  (size_t)target.rows()*sizeof(PREC), (size_t)target.cols(), cudaMemcpyDeviceToHost);
    }

    return err;
}

template<typename TCudaMatrix>
void referenceFrom( TCudaMatrix & ref, const TCudaMatrix & src) {
    ASSERTMSG( ref.m_pDevice == nullptr , "ref.m_pDevice is NOT NULL, we loose this pointer if we referece to it!");
    ref = src;
}

};



#endif
