// ========================================================================================
//  CudaFramework
//  Copyright (C) 2014 by Gabriel Nützi <nuetzig (at) imes (d0t) mavt (d0t) ethz (d0t) ch>
//
//  This Source Code Form is subject to the terms of the GNU GPL 3.0 licence.
//  If a copy of the GNU GPL 3.0 was not distributed with this
//  file, you can obtain one at http://opensource.org/licenses/GPL-3.0.
// ========================================================================================

#ifndef CudaFramework_Kernels_MatrixMultGPU_MatrixMultGPU_hpp
#define CudaFramework_Kernels_MatrixMultGPU_MatrixMultGPU_hpp

#include "cublas_v2.h"

#include "CudaFramework/CudaModern/CudaMatrix.hpp"

#include "CudaFramework/CudaModern/CudaError.hpp"

namespace matrixMultGPU{

   template<typename TCudaMatrix>
   void matrixMultiply_kernelWrap(TCudaMatrix & C,TCudaMatrix & A, TCudaMatrix & B,  const dim3 &threads, const dim3 &blocks);

   template<typename TCudaMatrix>
   void matrixMultiplyShared_kernelWrap(TCudaMatrix & C,TCudaMatrix & A, TCudaMatrix & B,  const dim3 &threads, const dim3 &blocks);

   template<typename TCudaMatrix>
   void matrixMultiplySharedFixed_kernelWrap(TCudaMatrix & C,TCudaMatrix & A, TCudaMatrix & B);

   template<typename TCudaMatrix>
   void matrixMultiplySharedFixedLargeRow_kernelWrap(TCudaMatrix & C,TCudaMatrix & A, TCudaMatrix & B);

   template<typename TCudaMatrix>
   void matrixMultiplySharedFixedLargeCol_kernelWrap(TCudaMatrix & C,TCudaMatrix & A, TCudaMatrix & B);

   template<typename TCudaMatrix>
   void matrixMultiplySharedFixedLargeColOptimized_kernelWrap(TCudaMatrix & C,TCudaMatrix & A, TCudaMatrix & B);

   template<typename TCudaMatrix>
   void matrixMultiplySharedFixedLargeBase_kernelWrap(TCudaMatrix & C,TCudaMatrix & A, TCudaMatrix & B);


   /**
   * @brief A random matrix multiplication to test several kernels.
   */
   void randomMatrixMult(int kernel);

   /**
   * @brief A random matrix multiplication performance test over several block and grid dimensions in CUDA.
   */
   void performanceTestMatrixMult(std::ostream & data, std::ostream & log, int kernel);


   /**
   * @brief cuBlas implementation of a Matrix Multiplication, cuBlas needs matrices in column major format.
   */

template<typename TCudaMatrix>
void matrixMultiplyCUBLAS(cublasHandle_t handle, TCudaMatrix Cdev, TCudaMatrix Adev,TCudaMatrix Bdev) {

    if( std::is_same<typename TCudaMatrix::PREC,float>::value ) {
        float a = 1.0;
        float b = 0.0;
        CHECK_CUBLAS(cublasSgemm(
                         handle,
                         CUBLAS_OP_N ,
                         CUBLAS_OP_N,
                         Adev.m_M,
                         Bdev.m_N,
                         Adev.m_N,
                         (float*)&a,
                         (float*)Adev.m_pDevice,
                         (int)(Adev.m_outerStrideBytes / sizeof(double)),
                         (float*)Bdev.m_pDevice,
                         (int)(Bdev.m_outerStrideBytes / sizeof(double)),
                         (float*)&b,
                         (float*)Cdev.m_pDevice,Cdev.m_M)
                    );
    } else {
        double a = 1.0;
        double b = 0.0;
        CHECK_CUBLAS(cublasDgemm(
                         handle,
                         CUBLAS_OP_N ,
                         CUBLAS_OP_N,
                         Adev.m_M,
                         Bdev.m_N,
                         Adev.m_N,
                         (double*)&a,
                         (double*)Adev.m_pDevice,
                         (int)(Adev.m_outerStrideBytes / sizeof(double)),
                         (double*)Bdev.m_pDevice,
                         (int)(Bdev.m_outerStrideBytes / sizeof(double)),
                         (double*)&b,
                         (double*)Cdev.m_pDevice,Cdev.m_M)
                    );
    }
}

}

#endif
