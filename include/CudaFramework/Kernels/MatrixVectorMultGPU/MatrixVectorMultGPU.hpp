// ========================================================================================
//  CudaFramework
//  Copyright (C) 2014 by Gabriel Nützi <nuetzig (at) imes (d0t) mavt (d0t) ethz (d0t) ch>
//
//  This Source Code Form is subject to the terms of the GNU GPL 3.0 licence.
//  If a copy of the GNU GPL 3.0 was not distributed with this
//  file, you can obtain one at http://opensource.org/licenses/GPL-3.0.
// ========================================================================================

#ifndef CudaFramework_Kernels_MatrixVectorMultGPU_MatrixVectorMultGPU_hpp
#define CudaFramework_Kernels_MatrixVectorMultGPU_MatrixVectorMultGPU_hpp


#include <cuda_runtime.h>
#include "cublas_v2.h"

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include "CudaFramework/General/ConfigureFile.hpp"
#include "CudaFramework/CudaModern/CudaError.hpp"
#include "CudaFramework/CudaModern/CudaUtilities.hpp"
#include "CudaFramework/CudaModern/CudaMatrix.hpp"
#include "CudaFramework/CudaModern/CudaMatrixUtilities.hpp"

#include "CudaFramework/General/GPUDefines.hpp"



#if USE_INTEL_BLAS == 1
#include "omp.h"
#include "mkl.h"
#include "mkl_types.h"
#include "mkl_cblas.h"
#else
#if USE_GOTO_BLAS == 1
#include "myblas.h"
#endif
#if USE_OPEN_BLAS == 1
#include "cblas.h"
#endif
#endif

#include "CudaFramework/CudaModern/CudaError.hpp"
#include "CudaFramework/CudaModern/CudaMatrix.hpp"
#include "CudaFramework/General/TypeTraitsHelper.hpp"
#include "CudaFramework/General/StaticAssert.hpp"


namespace matrixVectorMultGPU {

using namespace utilCuda;

template<typename TCudaMatrix>
void cublasGemv(cublasHandle_t handle,
                typename TCudaMatrix::PREC  a,
                const TCudaMatrix  & A_dev,
                const TCudaMatrix  &x_old_dev,
                typename TCudaMatrix::PREC  b,
                TCudaMatrix & b_dev) {

    ASSERTMSG(A_dev.m_outerStrideBytes % sizeof(typename TCudaMatrix::PREC) == 0,"ERROR: The stride of your matrix is not a multiple of float size");
    ASSERTMSG(A_dev.m_N == x_old_dev.m_M && A_dev.m_M == b_dev.m_M ,"ERROR: Vector/Matrix wrong dimension");

    // b_dev = alpha * A_dev * x_old_dev + beta *b_dev
    if(CudaMatrixFlags::isRowMajor<TCudaMatrix::Flags>::value ) {
        if( std::is_same<typename TCudaMatrix::PREC,float>::value ){
            CHECK_CUBLAS((cublasSgemv(
                             handle,
                             CUBLAS_OP_T ,
                             A_dev.m_M,
                             A_dev.m_N,
                             (float*)&a,
                             (float*)A_dev.m_pDevice,
                             (int)(A_dev.m_outerStrideBytes / sizeof(typename TCudaMatrix::PREC)),
                             (float*)x_old_dev.m_pDevice,
                             1,
                             (float*)&b,
                             (float*)b_dev.m_pDevice,
                             1)
                        ));
        }else{
            CHECK_CUBLAS(cublasDgemv(
                             handle,
                             CUBLAS_OP_T ,
                             A_dev.m_M,
                             A_dev.m_N,
                             (double*)&a,
                             (double*)A_dev.m_pDevice,
                             (int)(A_dev.m_outerStrideBytes / sizeof(typename TCudaMatrix::PREC)),
                             (double*)x_old_dev.m_pDevice,
                             1,
                             (double*)&b,
                             (double*)b_dev.m_pDevice,
                             1)
                        );
        }
    } else {
        if( std::is_same<typename TCudaMatrix::PREC,float>::value ){
            CHECK_CUBLAS(cublasSgemv(
                             handle,
                             CUBLAS_OP_N ,
                             A_dev.m_M,
                             A_dev.m_N,
                             (float*)&a,
                             (float*)A_dev.m_pDevice,
                             (int)(A_dev.m_outerStrideBytes / sizeof(typename TCudaMatrix::PREC)),
                             (float*)x_old_dev.m_pDevice,
                             1,
                             (float*)&b,
                             (float*)b_dev.m_pDevice,
                             1)
                        );
        }else{

                    CHECK_CUBLAS(cublasDgemv(
                             handle,
                             CUBLAS_OP_N ,
                             A_dev.m_M,
                             A_dev.m_N,
                             (double*)&a,
                             (double*)A_dev.m_pDevice,
                             (int)(A_dev.m_outerStrideBytes / sizeof(typename TCudaMatrix::PREC)),
                             (double*)x_old_dev.m_pDevice,
                             1,
                             (double*)&b,
                             (double*)b_dev.m_pDevice,
                             1)
                        );

        }
    }
}

template<typename TCudaMatrix>
void cublasSymv(cublasHandle_t handle,
                typename TCudaMatrix::PREC  a,
                const TCudaMatrix  & A_dev,
                const TCudaMatrix  &x_old_dev,
                typename TCudaMatrix::PREC  b,
                TCudaMatrix & b_dev)
{

   ASSERTMSG(A_dev.m_outerStrideBytes % sizeof(typename TCudaMatrix::PREC) == 0,"ERROR: The stride of your matrix is not a multiple of float size!");
   ASSERTMSG(A_dev.m_M == A_dev.m_N ,"ERROR: Matrix not symetric!");
   ASSERTMSG(A_dev.m_N == x_old_dev.m_M && A_dev.m_M == b_dev.m_M ,"ERROR: Vector/Matrix wrong dimension");

   if( std::is_same<typename TCudaMatrix::PREC,float>::value ){
      CHECK_CUBLAS((cublasSsymv(
         handle,
         CUBLAS_FILL_MODE_LOWER ,
         A_dev.m_N,
         (float*)&a,
         (float*)A_dev.m_pDevice,
         (int)(A_dev.m_outerStrideBytes / sizeof(typename TCudaMatrix::PREC)),
         (float*)x_old_dev.m_pDevice,
         1,
         (float*)&b,
         (float*)b_dev.m_pDevice,
         1))
         );
   }else{
      CHECK_CUBLAS( (cublasDsymv(
         handle,
         CUBLAS_FILL_MODE_LOWER ,
         A_dev.m_N,
         (double*)&a,
         (double*)A_dev.m_pDevice,
         (int)(A_dev.m_outerStrideBytes / sizeof(typename TCudaMatrix::PREC)),
         (double*)x_old_dev.m_pDevice,
         1,
         (double*)&b,
         (double*)b_dev.m_pDevice,
         1))
         );
   }
}



template<typename TCudaMatrix>
void matrixVectorMultiply_kernelWrap(  TCudaMatrix & y_dev,
                                       int incr_y,
                                       typename TCudaMatrix::PREC alpha,
                                       TCudaMatrix & A_dev,
                                       TCudaMatrix & x_dev,
                                       int incr_x,
                                       typename TCudaMatrix::PREC beta,
                                       TCudaMatrix & b_dev,
                                       int incr_b);



template<typename PREC> struct blasGemv;

template<>
struct blasGemv<double> {

    typedef double PREC;

    template<typename Derived1, typename Derived2>
    static void run( double a, const Eigen::MatrixBase<Derived2> & A, const Eigen::MatrixBase<Derived1> & x, double  b, Eigen::MatrixBase<Derived1>  &y) {

        EIGEN_STATIC_ASSERT(sizeof(PREC) == sizeof(typename Derived1::Scalar), YOU_MIXED_DIFFERENT_NUMERIC_TYPES__YOU_NEED_TO_USE_THE_CAST_METHOD_OF_MATRIXBASE_TO_CAST_NUMERIC_TYPES_EXPLICITLY)

        ASSERTMSG(A.cols() == x.rows() && A.rows() == y.rows() ,"ERROR: Vector/Matrix wrong dimension");

        // b_dev = alpha * A_dev * x_old_dev + beta *b_dev

        //Derived2 C = A;

#if USE_INTEL_BLAS == 1

        CBLAS_ORDER     order;
        CBLAS_TRANSPOSE trans;

        if(Derived1::Flags & Eigen::RowMajorBit) {
            order = CblasRowMajor;
        } else {
            order = CblasColMajor;
        }

        trans = CblasNoTrans;

        mkl_set_dynamic(false);
        mkl_set_num_threads(BLAS_NUM_THREADS);
        //cout << "Threads:" << mkl_get_max_threads();
        cblas_dgemv(order, trans, A.rows(), A.cols(), a, const_cast<double*>(&(A.operator()(0,0))), A.outerStride(), const_cast<double*>(&(x.operator()(0,0))), 1, b, &(y.operator()(0,0)), 1);
        //cblas_dgemm(order,trans,trans, A.rows(), A.cols(), A.cols(), 1.0,  const_cast<double*>(&(A.operator()(0,0))), A.rows(), const_cast<double*>(&(A.operator()(0,0))), A.rows(), 1.0 , &(C.operator()(0,0)), C.rows() );
#else

#if USE_GOTO_BLAS == 1
        /* static DGEMVFunc DGEMV = NULL;
        if (DGEMV == NULL) {
          HINSTANCE hInstLibrary = LoadLibrary("libopenblasp-r0.1alpha2.2.dll");
          DGEMV = (DGEMVFunc)GetProcAddress(hInstLibrary, "DGEMV");
        }*/

        char trans = 'N';
        BLAS_INT idx = 1;
        BLAS_INT m = A.rows();
        BLAS_INT n = A.cols();

        dgemv(&trans, &m, &n, &a, &(A.operator()(0,0)), &m, &(x.operator()(0,0)), &idx, &b, &(y.operator()(0,0)), &idx);

        //   FreeLibrary(hInstLibrary);
#else
        ASSERTMSG(false,"No implementation for BLAS defined!");
#endif

#endif
    }

};

template<>
struct blasGemv<float> {
    typedef float PREC;
    template<typename Derived1, typename Derived2>
    static void run( float a, const Eigen::MatrixBase<Derived2> & A, const Eigen::MatrixBase<Derived1> & x, float  b,  Eigen::MatrixBase<Derived1>  &y) {

        EIGEN_STATIC_ASSERT(sizeof(PREC) == sizeof(typename Derived1::Scalar), YOU_MIXED_DIFFERENT_NUMERIC_TYPES__YOU_NEED_TO_USE_THE_CAST_METHOD_OF_MATRIXBASE_TO_CAST_NUMERIC_TYPES_EXPLICITLY)

        ASSERTMSG(A.cols() == x.rows() && A.rows() == y.rows() ,"ERROR: Vector/Matrix wrong dimension");

        // b_dev = alpha * A_dev * x_old_dev + beta *b_dev
        //Derived2 C = A;

#if USE_INTEL_BLAS == 1

        CBLAS_ORDER     order;
        CBLAS_TRANSPOSE trans;

        if(Derived1::Flags & Eigen::RowMajorBit) {
            order = CblasRowMajor;
        } else {
            order = CblasColMajor;
        }

        trans = CblasNoTrans;


        mkl_set_dynamic(false);
        mkl_set_num_threads(BLAS_NUM_THREADS);
        //cout << "Threads:" << mkl_get_max_threads();
        cblas_sgemv(order, trans, A.rows(), A.cols(), a, const_cast<double*>(&(A.operator()(0,0))), A.outerStride(), const_cast<double*>(&(x.operator()(0,0))), 1, b, &(y.operator()(0,0)), 1);
        //cblas_dgemm(order,trans,trans, A.rows(), A.cols(), A.cols(), 1.0,  const_cast<double*>(&(A.operator()(0,0))), A.rows(), const_cast<double*>(&(A.operator()(0,0))), A.rows(), 1.0 , &(C.operator()(0,0)), C.rows() );
#else

#if USE_GOTO_BLAS == 1
        /* static DGEMVFunc DGEMV = NULL;
        if (DGEMV == NULL) {
          HINSTANCE hInstLibrary = LoadLibrary("libopenblasp-r0.1alpha2.2.dll");
          DGEMV = (DGEMVFunc)GetProcAddress(hInstLibrary, "DGEMV");
        }*/

        char trans = 'N';
        BLAS_INT idx = 1;
        BLAS_INT m = A.rows();
        BLAS_INT n = A.cols();

        sgemv(&trans, &m, &n, &a, &(A.operator()(0,0)), &m, &(x.operator()(0,0)), &idx, &b, &(y.operator()(0,0)), &idx);

        //   FreeLibrary(hInstLibrary);
#else
        ASSERTMSG(false,"No implementation for BLAS defined!");
#endif

#endif

    }

};


};

#endif
