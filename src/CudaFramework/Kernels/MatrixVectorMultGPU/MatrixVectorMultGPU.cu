// ========================================================================================
//  CudaFramework
//  Copyright (C) 2014 by Gabriel Nützi <nuetzig (at) imes (d0t) mavt (d0t) ethz (d0t) ch>
//
//  This Source Code Form is subject to the terms of the GNU GPL 3.0 licence. 
//  If a copy of the GNU GPL 3.0 was not distributed with this
//  file, you can obtain one at http://opensource.org/licenses/GPL-3.0.
// ========================================================================================


#include <cuda_runtime.h>
#include "CudaFramework/CudaModern/CudaMatrix.hpp"
#include "CudaFramework/General/GPUMutex.hpp"
#include "CudaFramework/General/AssertionDebugC.hpp"

#include "CudaFramework/Kernels/MatrixVectorMultGPU/KernelsMatrixVectorMult.cuh"

#include "CudaFramework/General/StaticAssert.hpp"
#include "CudaFramework/Kernels/ProxGPU/ProxKernelSettings.hpp"

namespace matrixVectorMultGPU{
   using namespace utilCuda;
   using namespace matrixVectorMultKernels;


    template<typename PREC, typename T>
    __host__ int foo1(PREC a){
      int g = T::ThreadsPerBlockKernelA + 1;
      return g;
    };

   template<> __host__ int foo1<float,   SorProxKernelSettings<1,ConvexSets::RPlusAndDisk> >(float a);
   template<> __host__ int foo1<double, RelaxedSorProxKernelSettings<1,ConvexSets::RPlusAndDisk> >(double a);

   template<typename TCudaMatrix>
   __host__ void matrixVectorMultiply_kernelWrap(  TCudaMatrix & y_dev,
                                                   int incr_y,
                                                   typename TCudaMatrix::PREC alpha,
                                                   TCudaMatrix & A_dev,
                                                   TCudaMatrix & x_dev,
                                                   int incr_x,
                                                   typename TCudaMatrix::PREC beta,
                                                   TCudaMatrix & b_dev,
                                                   int incr_b){
      typedef typename TCudaMatrix::PREC PREC;
      ASSERTMSG_C(A_dev.m_N == x_dev.m_M && y_dev.m_M == A_dev.m_M && b_dev.m_M == A_dev.m_M, "Matrices have wrong dimensions for multiplication");
      // Launch kernel ===========================================================
      //printf("Launch kernel, B: %i,%i, T: %i,%i \n",blocks.x,blocks.y, threads.x,threads.y);
      dim3 blocks;


      if(y_dev.m_pDevice == x_dev.m_pDevice){
         blocks.x = (A_dev.m_M + (BLOCK_DIM-1)) / BLOCK_DIM; // only this allowed!!
         // we have inplace
         GPUAtomicCounter global_sync;
         global_sync.init();
         matrixVectorMultiply_inPlace_kernel<<< blocks, THREADS_PER_BLOCK >>>( y_dev, incr_y, alpha, A_dev, x_dev, incr_x, beta, b_dev, incr_b, global_sync);
         global_sync.free();
      }else{
         blocks.x = (A_dev.m_M + (BLOCK_DIM-1)) / BLOCK_DIM; // or another blockDim
         // no in place
         matrixVectorMultiply_kernel<<< blocks, THREADS_PER_BLOCK >>>( y_dev, incr_y, alpha, A_dev, x_dev, incr_x, beta, b_dev, incr_b);
      }
   }



   // Explicit instantiate the types which are need in C++, other wise the code is not available for linking
   #define TCudaMatrix CudaMatrix<double,CudaMatrixFlags::ColMajor>
   template __host__ void matrixVectorMultiply_kernelWrap(TCudaMatrix & y_dev,
                                                          int incr_y,
                                                          typename TCudaMatrix::PREC
                                                          alpha, TCudaMatrix & A_dev,
                                                          TCudaMatrix & x_dev,
                                                          int incr_x,
                                                          typename TCudaMatrix::PREC  beta,
                                                          TCudaMatrix & b_dev,int incr_b);
   #undef TCudaMatrix

    #define TCudaMatrix CudaMatrix<float,CudaMatrixFlags::ColMajor>
   template __host__ void matrixVectorMultiply_kernelWrap(TCudaMatrix & y_dev,
                                                          int incr_y,
                                                          typename TCudaMatrix::PREC
                                                          alpha, TCudaMatrix & A_dev,
                                                          TCudaMatrix & x_dev,
                                                          int incr_x,
                                                          typename TCudaMatrix::PREC  beta,
                                                          TCudaMatrix & b_dev,int incr_b);
   #undef TCudaMatrix
}
