#include "CudaFramework/Kernels/GaussSeidelGPU/KernelsGaussSeidel.cuh"

#include <cuda_runtime.h>

#include "CudaFramework/CudaModern/CudaMatrix.hpp"

namespace gaussSeidelGPU {

using namespace utilCuda;
using namespace gaussSeidelKernels;

template<typename TCudaMatrix>
__host__ void blockGaussSeidelStepA_kernelWrap( TCudaMatrix &G_dev, TCudaMatrix &c_dev, TCudaMatrix &t_dev, TCudaMatrix &x_old_dev,  int j_g) {

    dim3 threads(BLOCK_DIM);
    dim3 blocks(1);
    gaussSeidelKernels::blockGaussSeidelStepA_kernel<<< blocks, threads >>>( G_dev, c_dev, t_dev, x_old_dev,  j_g);

}
template<typename TCudaMatrix>
__host__ void blockGaussSeidelStepANoDivision_kernelWrap( TCudaMatrix &T_dev, TCudaMatrix &d_dev, TCudaMatrix &t_dev, TCudaMatrix &x_old_dev,  int j_g, bool* convergedFlag_dev, typename TCudaMatrix::PREC absTOL, typename TCudaMatrix::PREC relTOL) {

    dim3 threads(BLOCK_DIM);
    dim3 blocks(1);

    gaussSeidelKernels::blockGaussSeidelStepANoDivision_kernel<<< blocks, threads >>>( T_dev, d_dev, t_dev, x_old_dev,  j_g, convergedFlag_dev, absTOL, relTOL);
}


template<typename TCudaMatrix>
__host__ void blockGaussSeidelStepACorrect_kernelWrap( TCudaMatrix &G_dev, TCudaMatrix &c_dev, TCudaMatrix &t_dev, TCudaMatrix &x_old_dev,  int j_g) {

    dim3 threads(BLOCK_DIM);
    dim3 blocks(1);

    gaussSeidelKernels::blockGaussSeidelStepACorrect_kernel<<< blocks, threads >>>( G_dev, c_dev, t_dev, x_old_dev,  j_g);
}

template<typename TCudaMatrix>
__host__ void blockGaussSeidelStepACorrectNoDivision_kernelWrap( TCudaMatrix &T_dev, TCudaMatrix &d_dev, TCudaMatrix &t_dev, TCudaMatrix &x_old_dev,  int j_g) {

    dim3 threads(BLOCK_DIM);
    dim3 blocks(1);

    gaussSeidelKernels::blockGaussSeidelStepACorrectNoDivision_kernel<<< blocks, threads >>>( T_dev, d_dev, t_dev, x_old_dev,  j_g);
}
template<typename TCudaMatrix>
__host__ void blockGaussSeidelStepACorrectNoDivision_kernelWrap( TCudaMatrix &T_dev, TCudaMatrix &d_dev, TCudaMatrix &t_dev, TCudaMatrix &x_old_dev,  int j_g, bool* convergedFlag_dev, typename TCudaMatrix::PREC absTOL, typename TCudaMatrix::PREC relTOL) {

    dim3 threads(BLOCK_DIM);
    dim3 blocks(1);

    gaussSeidelKernels::blockGaussSeidelStepACorrectNoDivision_kernel<<< blocks, threads >>>( T_dev, d_dev, t_dev, x_old_dev,  j_g, convergedFlag_dev, absTOL, relTOL);
}

template<typename TCudaMatrix>
__host__ void blockGaussSeidelStepB_kernelWrap( TCudaMatrix &G_dev, TCudaMatrix &c_dev, TCudaMatrix &t_dev, TCudaMatrix &x_old_dev,  int j_g) {

    dim3 threads(BLOCK_DIM);
    dim3 blocks(G_dev.m_M - BLOCK_DIM);
    gaussSeidelKernels::blockGaussSeidelStepB_kernel<<< blocks, threads >>>( G_dev, c_dev, t_dev, x_old_dev,  j_g);

}

// Explicit instantiate the types which are need in C++, other wise the code is not available for linking
#define TCudaMatrix CudaMatrix<float,CudaMatrixFlags::ColMajor>

template __host__ void blockGaussSeidelStepA_kernelWrap( TCudaMatrix &G_dev, TCudaMatrix &c_dev, TCudaMatrix &t_dev, TCudaMatrix &x_old_dev,  int j_g);
template __host__ void blockGaussSeidelStepANoDivision_kernelWrap( TCudaMatrix &T_dev, TCudaMatrix &d_dev, TCudaMatrix &t_dev, TCudaMatrix &x_old_dev,  int j_g, bool* convergedFlag_dev, typename TCudaMatrix::PREC absTOL, typename TCudaMatrix::PREC relTOL);

template __host__ void blockGaussSeidelStepACorrect_kernelWrap( TCudaMatrix &G_dev, TCudaMatrix &c_dev, TCudaMatrix &t_dev, TCudaMatrix &x_old_dev,  int j_g);
template __host__ void blockGaussSeidelStepACorrectNoDivision_kernelWrap( TCudaMatrix &T_dev, TCudaMatrix &d_dev, TCudaMatrix &t_dev, TCudaMatrix &x_old_dev,  int j_g);
template __host__ void blockGaussSeidelStepACorrectNoDivision_kernelWrap( TCudaMatrix &T_dev, TCudaMatrix &d_dev, TCudaMatrix &t_dev, TCudaMatrix &x_old_dev,  int j_g, bool* convergedFlag_dev, typename TCudaMatrix::PREC absTOL, typename TCudaMatrix::PREC relTOL);

template __host__ void blockGaussSeidelStepB_kernelWrap( TCudaMatrix &G_dev, TCudaMatrix &c_dev, TCudaMatrix &t_dev, TCudaMatrix &x_old_dev,  int j_g);
#undef TCudaMatrix


#define TCudaMatrix CudaMatrix<double,CudaMatrixFlags::ColMajor>
template __host__ void blockGaussSeidelStepA_kernelWrap( TCudaMatrix &G_dev, TCudaMatrix &c_dev, TCudaMatrix &t_dev, TCudaMatrix &x_old_dev,  int j_g);
template __host__ void blockGaussSeidelStepANoDivision_kernelWrap( TCudaMatrix &T_dev, TCudaMatrix &d_dev, TCudaMatrix &t_dev, TCudaMatrix &x_old_dev,  int j_g, bool* convergedFlag_dev, typename TCudaMatrix::PREC absTOL, typename TCudaMatrix::PREC relTOL);

template __host__ void blockGaussSeidelStepACorrect_kernelWrap( TCudaMatrix &G_dev, TCudaMatrix &c_dev, TCudaMatrix &t_dev, TCudaMatrix &x_old_dev,  int j_g);
template __host__ void blockGaussSeidelStepACorrectNoDivision_kernelWrap( TCudaMatrix &T_dev, TCudaMatrix &d_dev, TCudaMatrix &t_dev, TCudaMatrix &x_old_dev,  int j_g);
template __host__ void blockGaussSeidelStepACorrectNoDivision_kernelWrap( TCudaMatrix &T_dev, TCudaMatrix &d_dev, TCudaMatrix &t_dev, TCudaMatrix &x_old_dev,  int j_g, bool* convergedFlag_dev, typename TCudaMatrix::PREC absTOL, typename TCudaMatrix::PREC relTOL);

template __host__ void blockGaussSeidelStepB_kernelWrap( TCudaMatrix &G_dev, TCudaMatrix &c_dev, TCudaMatrix &t_dev, TCudaMatrix &x_old_dev,  int j_g);
#undef TCudaMatrix
}
