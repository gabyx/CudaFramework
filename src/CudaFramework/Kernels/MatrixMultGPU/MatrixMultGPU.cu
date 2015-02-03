// ========================================================================================
//  CudaFramework
//  Copyright (C) 2014 by Gabriel Nützi <nuetzig (at) imes (d0t) mavt (d0t) ethz (d0t) ch>
//
//  This Source Code Form is subject to the terms of the GNU GPL 3.0 licence. 
//  If a copy of the GNU GPL 3.0 was not distributed with this
//  file, you can obtain one at http://opensource.org/licenses/GPL-3.0.
// ========================================================================================

#include "CudaFramework/Kernels/MatrixMultGPU/KernelsMatrixMult.cuh"


#include <cuda_runtime.h>
#include "CudaFramework/CudaModern/CudaMatrix.hpp"
#include "CudaFramework/General/AssertionDebugC.hpp"

namespace matrixMultGPU {
using namespace utilCuda;
using namespace matrixMultKernels;

template<typename TCudaMatrix>
__host__ void matrixMultiply_kernelWrap(TCudaMatrix &C,TCudaMatrix &A, TCudaMatrix &B, const dim3 &threads, const dim3 &blocks) {
    ASSERTMSG_C(A.m_N == B.m_M && C.m_M == A.m_M && C.m_N == B.m_N, "Matrices have wrong dimensions for multiplication");
    // Launch kernel ===========================================================
    //printf("Launch kernel, B: %i,%i, T: %i,%i \n",blocks.x,blocks.y, threads.x,threads.y);
    matrixMultiply_kernel<<< blocks, threads >>>( C, A, B);
}

template<typename TCudaMatrix>
__host__ void matrixMultiplyShared_kernelWrap(TCudaMatrix &C,TCudaMatrix &A, TCudaMatrix &B,  const dim3 &threads, const dim3 &blocks) {
    ASSERTMSG_C( A.m_N == B.m_M && C.m_M == A.m_M && C.m_N == B.m_N, "Matrices have wrong dimensions for multiplication");
    typedef typename TCudaMatrix::PREC PREC;
    // Launch kernel ===========================================================
    //printf("Launch kernel, B: %i,%i, T: %i,%i \n",blocks.x,blocks.y, threads.x,threads.y);
    // Calcualte the amount of shared memory for A_sh and B_sh
    size_t size_Ash_and_Bsh = 2 * (threads.x*threads.y) * sizeof(PREC);
    matrixMultiplyShared_kernel<<< blocks, threads , size_Ash_and_Bsh >>>(C, A, B);
}

template<typename TCudaMatrix>
__host__ void matrixMultiplySharedFixed_kernelWrap(TCudaMatrix &C,TCudaMatrix &A, TCudaMatrix &B) {
    ASSERTMSG_C( A.m_N == B.m_M && C.m_M == A.m_M && C.m_N == B.m_N, "Matrices have wrong dimensions for multiplication");
    // Compute appropriate Grid size and Blocksize ==============================
    dim3 threads(mMSF_BLOCKSIZE,mMSF_BLOCKSIZE);
    dim3 blocks(C.m_M/mMSF_BLOCKSIZE,C.m_N/mMSF_BLOCKSIZE);
    // Launch kernel ===========================================================
    //printf("Launch kernel, B: %i,%i, T: %i,%i \n",blocks.x,blocks.y, threads.x,threads.y);;
    matrixMultiplySharedFixed_kernel<<< blocks, threads >>>(C, A, B);
}

template<typename TCudaMatrix>
__host__ void matrixMultiplySharedFixedLargeRow_kernelWrap(TCudaMatrix &C,TCudaMatrix &A, TCudaMatrix &B) {
    ASSERTMSG_C( A.m_N == B.m_M && C.m_M == A.m_M && C.m_N == B.m_N, "Matrices have wrong dimensions for multiplication");
    // Compute appropriate Grid size and Blocksize ==============================
    dim3 threads(16,16);
    dim3 blocks(C.m_N/16,C.m_M/256);
    // Launch kernel ===========================================================
    //printf("Launch kernel, B: %i,%i, T: %i,%i \n",blocks.x,blocks.y, threads.x,threads.y);
    matrixMultiplySharedFixedLargeRow_kernel<<< blocks, threads >>>(C,A,B);
}

template<typename TCudaMatrix>
__host__ void matrixMultiplySharedFixedLargeCol_kernelWrap(TCudaMatrix &C,TCudaMatrix &A, TCudaMatrix &B) {
    ASSERTMSG_C( A.m_N == B.m_M && C.m_M == A.m_M && C.m_N == B.m_N, "Matrices have wrong dimensions for multiplication");

    //Compute appropriate Grid size and Blocksize ==============================
    dim3 threads(16,16);
    dim3 blocks(C.m_N / 256 , C.m_M / 16 );
    // Launch kernel ===========================================================
    //printf("Launch kernel, B: %i,%i, T: %i,%i \n",blocks.x,blocks.y, threads.x,threads.y);
    matrixMultiplySharedFixedLargeCol_kernel<<< blocks, threads >>>(C,A,B);

}

template<typename TCudaMatrix>
__host__ void matrixMultiplySharedFixedLargeColOptimized_kernelWrap(TCudaMatrix &C,TCudaMatrix &A, TCudaMatrix &B) {
    ASSERTMSG_C( A.m_N == B.m_M && C.m_M == A.m_M && C.m_N == B.m_N, "Matrices have wrong dimensions for multiplication");
    ASSERTMSG_C( A.m_N % 16 == 0 && B.m_N % 256 == 0 && A.m_M % 16 == 0, "Matrices must be a multiple of (16 x 256)!");
    //Compute appropriate Grid size and Blocksize ==============================
    dim3 threads(16,16);
    dim3 blocks(C.m_N/256,C.m_M/16);
    // Launch kernel ===========================================================
    //printf("Launch kernel, B: %i,%i, T: %i,%i \n",blocks.x,blocks.y, threads.x,threads.y);
    matrixMultiplySharedFixedLargeColOptimized_kernel<<< blocks, threads >>>(C,A,B);

}

template<typename TCudaMatrix>
__host__ void matrixMultiplySharedFixedLargeBase_kernelWrap(TCudaMatrix &C,TCudaMatrix &A, TCudaMatrix &B) {
    ASSERTMSG_C( A.m_N == B.m_M && C.m_M == A.m_M && C.m_N == B.m_N, "Matrices have wrong dimensions for multiplication");
    ASSERTMSG_C( A.m_N == A.m_M && B.m_N == B.m_M , "Matrices must be square!");
    ASSERTMSG_C( A.m_N % 256 == 0 && B.m_N % 256 == 0, "Matrices must be a multiple of 256!");

    dim3 threads(16,16);
    dim3 blocks(C.m_M/256,C.m_N/16);
    matrix_large_tile_base<<< blocks, threads >>>(A.m_pDevice, B.m_pDevice, C.m_pDevice);
}

// Explicit instantiate the types which are need in C++, other wise the code is not available for linking

#define EXPLICIT_INSTANTIATION \
template __host__ void matrixMultiply_kernelWrap(TCudaMatrix &C,TCudaMatrix &A, TCudaMatrix &B,const dim3 &threads, const dim3 &blocks);\
template __host__ void matrixMultiplyShared_kernelWrap(TCudaMatrix &C,TCudaMatrix &A, TCudaMatrix &B,const dim3 &threads, const dim3 &blocks);\
template __host__ void matrixMultiplySharedFixed_kernelWrap(TCudaMatrix &C,TCudaMatrix &A, TCudaMatrix &B);\
template __host__ void matrixMultiplySharedFixedLargeRow_kernelWrap(TCudaMatrix &C,TCudaMatrix &A, TCudaMatrix &B);\
template __host__ void matrixMultiplySharedFixedLargeCol_kernelWrap(TCudaMatrix &C,TCudaMatrix &A, TCudaMatrix &B);\
template __host__ void matrixMultiplySharedFixedLargeColOptimized_kernelWrap(TCudaMatrix &C,TCudaMatrix &A, TCudaMatrix &B); \
template __host__ void matrixMultiplySharedFixedLargeBase_kernelWrap(TCudaMatrix &C,TCudaMatrix &A, TCudaMatrix &B);\


#define TCudaMatrix CudaMatrix<double,CudaMatrixFlags::RowMajor>
EXPLICIT_INSTANTIATION
#undef TCudaMatrix
#define TCudaMatrix CudaMatrix<float,CudaMatrixFlags::RowMajor>
EXPLICIT_INSTANTIATION
#undef TCudaMatrix

#define TCudaMatrix CudaMatrix<double,CudaMatrixFlags::ColMajor>
EXPLICIT_INSTANTIATION
#undef TCudaMatrix
#define TCudaMatrix CudaMatrix<float,CudaMatrixFlags::ColMajor>
EXPLICIT_INSTANTIATION
#undef TCudaMatrix

#undef TCudaMatrix
#undef EXPLICIT_INSTANTIATION
}
