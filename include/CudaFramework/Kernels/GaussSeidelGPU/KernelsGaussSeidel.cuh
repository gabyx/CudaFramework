// ========================================================================================
//  CudaFramework
//  Copyright (C) 2014 by Gabriel Nützi <nuetzig (at) imes (d0t) mavt (d0t) ethz (d0t) ch>
//
//  This Source Code Form is subject to the terms of the GNU GPL 3.0 licence.
//  If a copy of the GNU GPL 3.0 was not distributed with this
//  file, you can obtain one at http://opensource.org/licenses/GPL-3.0.
// ========================================================================================

#ifndef CudaFramework_Kernels_GaussSeidelGPU_KernelsGaussSeidel_cuh
#define CudaFramework_Kernels_GaussSeidelGPU_KernelsGaussSeidel_cuh

#include "CudaFramework/CudaModern/CudaMatrix.hpp"

#include "CudaFramework/General/GPUMutex.hpp"

using namespace utilCuda;

namespace gaussSeidelKernels {

__forceinline__ __device__ bool checkConverged(float x_new ,float x_old , float absTOL, float relTOL) {
    if(fabsf(x_new - x_old) > absTOL  /*+relTOL * fabsf(x_old)*/ ) {
        return false;
    }
    return true;
}

__forceinline__ __device__ bool checkConverged(double x_new ,double x_old , double absTOL, double relTOL) {
    if(fabs(x_new - x_old) > absTOL  /*+relTOL * fabs(x_old)*/ ) {
        return false;
    }
    return true;
}


const unsigned int BLOCK_DIM = 64;

// Untypename TCudaMatrix::PRECise Versions
template<typename TCudaMatrix>
__global__ void blockGaussSeidelStepA_kernel( TCudaMatrix G_dev, TCudaMatrix c_dev,  TCudaMatrix t_dev,  TCudaMatrix x_old_dev,  int j_g) {


    typedef typename TCudaMatrix::PREC PREC;
    // Assumend 1 Block, with 64 Threads and Column Major Matrix G_dev

    // Block on the diagonal
    int i_g = j_g;


    /*PARFOR*/
    int i = i_g * BLOCK_DIM + threadIdx.x;
    int j;
    typename TCudaMatrix::PREC G_ij;

    for(int j_t = 0; j_t < BLOCK_DIM ; j_t++) {

        j =  j_g * BLOCK_DIM + j_t;
        G_ij = Elem_ColM(G_dev,i,j);

        if( i == j) { // If we are on diagonal finish the line of the gauss seidel algorithm.
            x_old_dev.m_pDevice[i] = -(c_dev.m_pDevice[i] + t_dev.m_pDevice[i]) /  G_ij;
            t_dev.m_pDevice[i] = 0;
        } else {
            t_dev.m_pDevice[i] += G_ij * x_old_dev.m_pDevice[j];
        }

        // If this is needed, not really sure;
        __syncthreads();
    }
}



template<typename TCudaMatrix>
__global__ void blockGaussSeidelStepANoDivision_kernel( TCudaMatrix T_dev, TCudaMatrix d_dev,  TCudaMatrix t_dev,  TCudaMatrix x_old_dev,  int j_g, bool * convergedFlag_dev, typename TCudaMatrix::PREC _absTOL, typename TCudaMatrix::PREC _relTOL) {
    typedef typename TCudaMatrix::PREC PREC;    // Assumend 1 Block, with 64 Threads and Column Major Matrix G_dev

    // Block on the diagonal
    int i_g = j_g;

    typename TCudaMatrix::PREC absTOL = _absTOL;
    typename TCudaMatrix::PREC relTOL = _relTOL;

    /*PARFOR*/
    int i = i_g * BLOCK_DIM + threadIdx.x;
    int j;
    typename TCudaMatrix::PREC T_ij;

    for(int j_t = 0; j_t < BLOCK_DIM ; j_t++) {

        j =  j_g * BLOCK_DIM + j_t;
        T_ij = Elem_ColM(T_dev,i,j);

        if( i == j) { // If we are on diagonal finish the line of the gauss seidel algorithm.
            typename TCudaMatrix::PREC x_old = x_old_dev.m_pDevice[i];
            typename TCudaMatrix::PREC x_new = -(d_dev.m_pDevice[i] + t_dev.m_pDevice[i]);
            if( !checkConverged(x_new,x_old,absTOL,relTOL)) {
                *convergedFlag_dev = 0;
            }

            x_old_dev.m_pDevice[i] = x_new;
            t_dev.m_pDevice[i] = 0;
        } else { /*if(i != j)*/
            t_dev.m_pDevice[i] += T_ij * x_old_dev.m_pDevice[j];
        }
        // If this is needed, not really sure;
        __syncthreads();
    }

}


// Correct Versions
template<typename TCudaMatrix>
__global__ void blockGaussSeidelStepACorrect_kernel( TCudaMatrix G_dev, TCudaMatrix c_dev,  TCudaMatrix t_dev,  TCudaMatrix x_old_dev,  int j_g) {
    typedef typename TCudaMatrix::PREC PREC;    // Assumend 1 Block, with 64 Threads and Column Major Matrix G_dev

    // Block on the diagonal
    int i_g = j_g;


    /*PARFOR*/
    int i = i_g * BLOCK_DIM + threadIdx.x;
    int j;
    typename TCudaMatrix::PREC G_ij;

    for(int j_t = 0; j_t < BLOCK_DIM ; j_t++) {

        j =  j_g * BLOCK_DIM + j_t;
        G_ij = Elem_ColM(G_dev,i,j);

        if( i == j) { // If we are on diagonal finish the line of the gauss seidel algorithm.
            x_old_dev.m_pDevice[i] = -(c_dev.m_pDevice[i] + t_dev.m_pDevice[i]) /  G_ij;
            t_dev.m_pDevice[i] = 0;
        }

        __syncthreads();

        if(i != j) {
            t_dev.m_pDevice[i] += G_ij * x_old_dev.m_pDevice[j];
        }
        // If this is needed, not really sure;
        __syncthreads();
    }

}

template<typename TCudaMatrix>
__global__ void blockGaussSeidelStepACorrectNoDivision_kernel( TCudaMatrix T_dev, TCudaMatrix d_dev,  TCudaMatrix t_dev,  TCudaMatrix x_old_dev,  int j_g) {
    typedef typename TCudaMatrix::PREC PREC;    // Assumend 1 Block, with 64 Threads and Column Major Matrix G_dev

    // Block on the diagonal
    int i_g = j_g;


    /*PARFOR*/
    int i = i_g * BLOCK_DIM + threadIdx.x;
    int j;
    typename TCudaMatrix::PREC T_ij;

    for(int j_t = 0; j_t < BLOCK_DIM ; j_t++) {

        j =  j_g * BLOCK_DIM + j_t;
        T_ij = Elem_ColM(T_dev,i,j);

        if( i == j) { // If we are on diagonal finish the line of the gauss seidel algorithm.
            x_old_dev.m_pDevice[i] = -(d_dev.m_pDevice[i] + t_dev.m_pDevice[i]);
            t_dev.m_pDevice[i] = 0;
        }

        __syncthreads();

        if(i != j) {
            t_dev.m_pDevice[i] += T_ij * x_old_dev.m_pDevice[j];
        }
        // If this is needed, not really sure;
        __syncthreads();
    }

}

template<typename TCudaMatrix>
__global__ void blockGaussSeidelStepACorrectNoDivision_kernel( TCudaMatrix T_dev, TCudaMatrix d_dev,  TCudaMatrix t_dev,  TCudaMatrix x_old_dev,  int j_g, bool * convergedFlag_dev, typename TCudaMatrix::PREC _absTOL, typename TCudaMatrix::PREC _relTOL) {
    typedef typename TCudaMatrix::PREC PREC;    // Assumend 1 Block, with 64 Threads and Column Major Matrix G_dev

    // Block on the diagonal
    int i_g = j_g;

    typename TCudaMatrix::PREC absTOL = _absTOL;
    typename TCudaMatrix::PREC relTOL = _relTOL;

    /*PARFOR*/
    int i = i_g * BLOCK_DIM + threadIdx.x;
    int j;
    typename TCudaMatrix::PREC T_ij;

    for(int j_t = 0; j_t < BLOCK_DIM ; j_t++) {

        j =  j_g * BLOCK_DIM + j_t;
        T_ij = Elem_ColM(T_dev,i,j);

        if( i == j) { // If we are on diagonal finish the line of the gauss seidel algorithm.
            typename TCudaMatrix::PREC x_old = x_old_dev.m_pDevice[i];
            typename TCudaMatrix::PREC x_new = -(d_dev.m_pDevice[i] + t_dev.m_pDevice[i]);
            if( !checkConverged(x_new,x_old,absTOL,relTOL)) {
                *convergedFlag_dev = 0;
            }

            x_old_dev.m_pDevice[i] = x_new;

            t_dev.m_pDevice[i] = 0;
        }

        __syncthreads();

        if(i != j) {
            t_dev.m_pDevice[i] += T_ij * x_old_dev.m_pDevice[j];
        }
        // If this is needed, not really sure;
        __syncthreads();
    }

}




template<typename TCudaMatrix>
__global__ void blockGaussSeidelStepB_kernel( TCudaMatrix G_dev, TCudaMatrix c_dev,  TCudaMatrix t_dev,  TCudaMatrix x_old_dev,  int j_g) {
    typedef typename TCudaMatrix::PREC PREC;    // Assumend G_dev.m_M - 64 Blocks, with each 64 Threads, and Column Major Matrix G_dev!

    // Shared cache
    __shared__ typename TCudaMatrix::PREC running_sum[BLOCK_DIM];

    // Calculate the indices i,j for the G_ij value which we need to multiply with the x_j value
    int i = blockIdx.x; //[0- G_dev.m_M-64]

    // Thread index [0-64]
    int j_t = threadIdx.x;

    if( i >= j_g * BLOCK_DIM ) {
        i = i + BLOCK_DIM; // skip diagonal block!
    }

    int j = j_g * BLOCK_DIM + j_t;
    // =================================

    // Save into sum
    running_sum[threadIdx.x] = Elem_ColM(G_dev,i,j) * x_old_dev.m_pDevice[j];

    // Sync all threads before we doo the reduction step to obtain the value t_dev[i];
    __syncthreads();

    //Reduction ( BLOCK_DIM needs to be a power of 2) =========
    for( int k = BLOCK_DIM/2; k>0; k>>=1) {
        if( j_t  < k) {
            running_sum[j_t] += running_sum[j_t + k];
        }
        __syncthreads();
    }

    // running_sum[0] contains the end result!
    // =========================================================

    // First thread writes the value
    if( j_t == 0) {
        t_dev.m_pDevice[i] +=  running_sum[0];
    }
}



#undef BLOCK_DIM



}




#endif
