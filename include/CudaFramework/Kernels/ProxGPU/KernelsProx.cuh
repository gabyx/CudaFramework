// ========================================================================================
//  CudaFramework
//  Copyright (C) 2014 by Gabriel Nützi <nuetzig (at) imes (d0t) mavt (d0t) ethz (d0t) ch>
//
//  This Source Code Form is subject to the terms of the GNU GPL 3.0 licence.
//  If a copy of the GNU GPL 3.0 was not distributed with this
//  file, you can obtain one at http://opensource.org/licenses/GPL-3.0.
// ========================================================================================

#ifndef CudaFramework_Kernels_ProxGPU_KernelsProx_cuh
#define CudaFramework_Kernels_ProxGPU_KernelsProx_cuh

#include "CudaFramework/CudaModern/CudaMatrix.hpp"

#include "CudaFramework/General/GPUMutex.hpp"
#include "CudaFramework/General/TypeTraitsHelper.hpp"
#include "CudaFramework/General/StaticAssert.hpp"
#include "ConvexSets.hpp"



namespace proxKernels {

using namespace utilCuda;

__forceinline__ __device__ bool checkConverged(float x_new ,float x_old , float absTOL, float relTOL) {
    if(fabsf(x_new - x_old) > (absTOL  +relTOL * fabsf(x_old)) ) {
        return false;
    }
    return true;
}

__forceinline__ __device__ bool checkConverged(double x_new ,double x_old , double absTOL, double relTOL) {
    if(fabs(x_new - x_old) > (absTOL  +relTOL * fabs(x_old)) ) {
        return false;
    }
    return true;
}

/**
* @brief  This Prox, assignes 2 threads for one triplet, one for normal and one for tangential direction.
* It uses the radius given in radius_dev. 2 threads for one triplet is useless, because if we want the normal direction first, we need to wait anyway!
* So the 1 thread implementation makes more sense!
*/
template<typename TCudaMatrix>
__global__ void proxContactOrdered_RPlusAndDisk_2threads_kernel(TCudaMatrix mu_dev, TCudaMatrix proxTerm_dev) {

    typedef typename TCudaMatrix::PREC PREC;
    int index_x = threadIdx.x + blockIdx.x * blockDim.x;
    int stride_x = blockDim.x * gridDim.x;

    // Calculate continious index of 2! e.g 7 / 2 = 3*2 + 1  , indexHigh = 3 and indexLow = 1
    int indexHigh = index_x >> 1; // index to which triplet this thread is assigned, 0= first triplet, 1 = second triplet etc.
    int indexLow = index_x & 1; // normal or tangential, 0 = normal , 1 = tangential

    int nTiles = (proxTerm_dev.m_M + (3-1)) / 3; // how many tiles we need to shift the whole grid!


    for(int i = 0; i<nTiles; i++) {


        // Do prox!==
        if(indexLow == 0 && (indexHigh+1)*3 <= proxTerm_dev.m_M ) {
            // do normal direction Prox
            if(proxTerm_dev.m_pDevice[indexHigh*3] < 0) {
                proxTerm_dev.m_pDevice[indexHigh*3] = 0;
            }
        }

        __syncthreads();

        if(indexLow == 1 && (indexHigh+1)*3 <= proxTerm_dev.m_M ) {
            // do tangential direction prox
            PREC lambda_T1 =  proxTerm_dev.m_pDevice[indexHigh*3+1];
            PREC lambda_T2 =  proxTerm_dev.m_pDevice[indexHigh*3+2];
            PREC radius = mu_dev.m_pDevice[indexHigh]* proxTerm_dev.m_pDevice[indexHigh*3];
            PREC absvalue = sqrt(lambda_T1*lambda_T1 + lambda_T2*lambda_T2);

            if(absvalue > radius) {
                proxTerm_dev.m_pDevice[indexHigh*3+1] =  lambda_T1 / absvalue * radius;
                proxTerm_dev.m_pDevice[indexHigh*3+2] =  lambda_T2 / absvalue * radius;
            }
        }

        // ==========
        // Shift index
        index_x += stride_x;
        // Calculate continious index of 2!
        indexHigh = index_x >> 1; // index to which triplet this thread is assigned, 0= first triplet, 1 = second triplet etc.
        indexLow = index_x & 1; // normal or tangential, 0 = normal , 1 = tangential

    }

}

/**
* @brief  This Prox, assignes 1 threads for one triplet
* It uses the radius scale factor given in mu_dev.
* Solves: proxTerm = prox_C( proxTerm ).
* Type: Contact Ordered.
* Approximate Flop Count: Flops = nContacs * ( 1 + 1 + 3 + 1*FLOPS_SQRT + 2 + 2*FLOPS_DIVISION)
* With assignment: Flops += 7
*/
template<typename TCudaMatrix, typename TConvexSet>
__global__ void proxContactOrdered_1threads_kernel(TCudaMatrix mu_dev, TCudaMatrix proxTerm_dev) {

    typedef typename TCudaMatrix::PREC PREC;
    STATIC_ASSERTM( (TypeTraitsHelper::IsSame<TConvexSet,ConvexSets::RPlusAndDisk>::result) , ONLY_RPLUS_AND_DISK_CONVEX_SET_IMPLEMENTED_SO_FAR )

    if(TypeTraitsHelper::IsSame<TConvexSet,ConvexSets::RPlusAndDisk>::result) {
        int index_x = threadIdx.x + blockIdx.x * blockDim.x;
        int stride_x = blockDim.x * gridDim.x;

        while((index_x+1)*3 <= proxTerm_dev.m_M) {

            // Do prox!==
            int idx_n = index_x*3;
            int idx_t1 = index_x*3 + 1;
            int idx_t2 = index_x*3 + 2;

            //Normal direction
            if(proxTerm_dev.m_pDevice[idx_n] < 0) {
                proxTerm_dev.m_pDevice[idx_n] = 0;
            }

            // Tangential direction
            PREC lambda_T1 =  proxTerm_dev.m_pDevice[idx_t1];
            PREC lambda_T2 =  proxTerm_dev.m_pDevice[idx_t2];
            PREC radius    =  mu_dev.m_pDevice[index_x]* proxTerm_dev.m_pDevice[idx_n];
            PREC absvalue  = lambda_T1*lambda_T1 + lambda_T2*lambda_T2;

            if(absvalue > radius*radius) {
                if(TypeTraitsHelper::IsSame<PREC,double>::result) {
                    absvalue = radius * rsqrt(absvalue);
                } else {
                    absvalue = radius * rsqrtf(absvalue);
                }
                proxTerm_dev.m_pDevice[idx_t1] = lambda_T1 *  absvalue;
                proxTerm_dev.m_pDevice[idx_t2] = lambda_T2 *  absvalue;
            }
            // ==========

            // Shift index
            index_x += stride_x;
        }
    }
}


/**
* @brief This is a stupid WORST CASE test for the prox function, this is ONLY for performance tests!
*/
template<typename TCudaMatrix, typename TConvexSet>
__global__ void proxContactOrderedWORSTCASE_1threads_kernel(TCudaMatrix mu_dev, TCudaMatrix proxTerm_dev) {

    typedef typename TCudaMatrix::PREC PREC;
    STATIC_ASSERTM( (TypeTraitsHelper::IsSame<TConvexSet,ConvexSets::RPlusAndDisk>::result) , ONLY_RPLUS_AND_DISK_CONVEX_SET_IMPLEMENTED_SO_FAR )

    if(TypeTraitsHelper::IsSame<TConvexSet,ConvexSets::RPlusAndDisk>::result) {
        int index_x = threadIdx.x + blockIdx.x * blockDim.x;
        int stride_x = blockDim.x * gridDim.x;

        while((index_x+1)*3 <= proxTerm_dev.m_M) {

            // Do prox!==
            int idx_n = index_x*3;
            int idx_t1 = index_x*3 + 1;
            int idx_t2 = index_x*3 + 2;

            //Normal direction
            //if(proxTerm_dev.m_pDevice[idx_n] < 0){
            proxTerm_dev.m_pDevice[idx_n] = 0;
            //}

            // Tangential direction
            PREC lambda_T1 =  proxTerm_dev.m_pDevice[idx_t1];
            PREC lambda_T2 =  proxTerm_dev.m_pDevice[idx_t2];
            PREC radius    =  mu_dev.m_pDevice[index_x]* proxTerm_dev.m_pDevice[idx_n];
            PREC absvalue  = lambda_T1*lambda_T1 + lambda_T2*lambda_T2;

            //if(absvalue > radius*radius){
            if(TypeTraitsHelper::IsSame<PREC,double>::result) {
                absvalue = radius * rsqrt(absvalue);
            } else {
                absvalue = radius * rsqrtf(absvalue);
            }
            proxTerm_dev.m_pDevice[idx_t1] = lambda_T1 *  absvalue;
            proxTerm_dev.m_pDevice[idx_t2] = lambda_T2 *  absvalue;
            //}
            // ==========

            // Shift index
            index_x += stride_x;
        }
    }
}



/**
* Solves: proxTerm = prox_C( proxTerm + b).
* Type: Contact Ordered. (CO)
* Approximate Flop Count: Flops = nContacs * ( 1 + 1 + 3 + 1*FLOPS_SQRT + 2 + 2*FLOPS_DIVISION)
* With assignment: Flops += 7
*/
template<typename TCudaMatrix, typename TConvexSet>
__global__ void proxContactOrdered_1threads_kernel(TCudaMatrix mu_dev, TCudaMatrix proxTerm_dev, TCudaMatrix b_dev) {

    typedef typename TCudaMatrix::PREC PREC;
    STATIC_ASSERT( (TypeTraitsHelper::IsSame<TConvexSet,ConvexSets::RPlusAndDisk>::result))

    if(TypeTraitsHelper::IsSame<TConvexSet,ConvexSets::RPlusAndDisk>::result) {
        int index_x = threadIdx.x + blockIdx.x * blockDim.x;
        int stride_x = blockDim.x * gridDim.x;

        while((index_x+1)*3 <= proxTerm_dev.m_M) {

            // Do prox!==
            int idx_n = index_x*3;
            int idx_t1 = index_x*3 + 1;
            int idx_t2 = index_x*3 + 2;

            // Add b_dev
            proxTerm_dev.m_pDevice[idx_n]  += b_dev.m_pDevice[idx_n];
            proxTerm_dev.m_pDevice[idx_t1] += b_dev.m_pDevice[idx_t1];
            proxTerm_dev.m_pDevice[idx_t2] += b_dev.m_pDevice[idx_t2];

            //Normal direction
            if(proxTerm_dev.m_pDevice[idx_n] < 0) {
                proxTerm_dev.m_pDevice[idx_n] = 0;
            }

            // Tangential direction
            PREC lambda_T1 =  proxTerm_dev.m_pDevice[idx_t1];
            PREC lambda_T2 =  proxTerm_dev.m_pDevice[idx_t2];
            PREC radius    =  mu_dev.m_pDevice[index_x]* proxTerm_dev.m_pDevice[idx_n];
            PREC absvalue  = lambda_T1*lambda_T1 + lambda_T2*lambda_T2;

            if(absvalue > radius*radius) {
                if(TypeTraitsHelper::IsSame<PREC,double>::result) {
                    absvalue = radius * rsqrt(absvalue);
                } else {
                    absvalue = radius * rsqrtf(absvalue);
                }
                proxTerm_dev.m_pDevice[idx_t1] = lambda_T1 *  absvalue;
                proxTerm_dev.m_pDevice[idx_t2] = lambda_T2 *  absvalue;
            }
            // ==========

            // Shift index
            index_x += stride_x;
        }
    }
}


/*
* @brief This prox does everything together.
* Solves: y = prox_C( alpha*A*x + beta*b).
* It performs an efficient Matrix-Vector multiplication to evaluate the inner term of the Prox, and afterwards it applies the prox to the result!
*/

#define XXINCR (THREADS_PER_BLOCK)
#define JINCR (X_ELEMS_PER_THREAD * THREADS_PER_BLOCK)
#define IDXA(row,col) (A_dev.outerStride*(col) + (row))
#define IDXX(row) ((row)*incr_x)
#define IDXB(row) ((row)*incr_b)
#define IDXY(row) ((row)*incr_y)
#define DOT_PROD_SEGMENT (UNROLL_BLOCK_DOTPROD)
#define PROX_PACKAGE_SIZE (TConvexSet::Dimension)

template<typename TCudaMatrix, int THREADS_PER_BLOCK, int BLOCK_DIM, int X_ELEMS_PER_THREAD, int PROX_PACKAGES, int UNROLL_BLOCK_DOTPROD, typename TConvexSet>
__global__ void jorProxContactOrdered_1threads_kernel(
    TCudaMatrix mu_dev,
    TCudaMatrix y_dev,
    int incr_y,
    typename TCudaMatrix::PREC alpha,
    TCudaMatrix A_dev,
    TCudaMatrix x_dev,
    int incr_x,
    typename TCudaMatrix::PREC beta,
    TCudaMatrix b_dev,
    int incr_b) {

    typedef typename TCudaMatrix::PREC PREC;
    STATIC_ASSERT( ! CudaMatrixFlags::isRowMajor< TCudaMatrix::Flags >::value )
    STATIC_ASSERT( (TypeTraitsHelper::IsSame<TConvexSet,ConvexSets::RPlusAndDisk>::result))

    // MATRIX-VECTOR MULTPILICATION  (y_dev = alpha*T_dev * x_dev + beta *b_dev) ===========================================
    __shared__ PREC XX[JINCR]; // Shared values for the x_dev;

    int thid = threadIdx.x;

    int i,j, ii, jj, row,  jjLimit, idx, incr;
    // i: first level index in row direction to the start of the current grid (grid level),
    // ii: second level index in row direction to the start of the current block (block level),

    PREC dotp; // Dot product which is accumulated by 1 thread.

    // Shift the whole cuda grid over the rows!

    for(i = 0 ; i < A_dev.m_M ; i += gridDim.x * BLOCK_DIM ) {

        ii = i + blockIdx.x * BLOCK_DIM;   // Represents the row index to the current block start
        if( ii >= A_dev.m_M) break;  // The blocks which lie outside the matrix can be rejected already here
        // Set the dot product to zero!
        dotp = 0;

        // Shift the block with dim[ BLOCK_DIM , (BLOCK_DIM*X_ELEMS_PER_THREAD)]
        // horizontally over A_dev, each thread calculates the sub dot product of
        // blockDim.x*X_ELEMS_PER_THREAD elements.
        // j points to the beginning of the block and is shifted with JINCR = BLOCK_DIM
        for(j = 0 ; j < A_dev.m_N; j += JINCR) {

            jj = j + thid;                            // Represents here the start index into x_dev for each thread!
            jjLimit = min (j + JINCR, A_dev.m_N);       // Represents here the maximal index for jj for this shifted block!

            // Syncronize before loading the (blockDim.x*X_ELEMS_PER_THREAD) shared values of x_dev
            // Each thread loads X_ELEMS_PER_THREAD elements.
            __syncthreads();

            // Load loop (all THREADS_PER_BLOCK participate!)
            // Each of this BLOCK_DIM threads goes into another if statement if the
            incr = incr_x * XXINCR;
            idx = IDXX(jj);
            STATIC_ASSERTM(X_ELEMS_PER_THREAD == 4 || X_ELEMS_PER_THREAD == 2 || X_ELEMS_PER_THREAD == 1, X_ELEMS_PER_THREAD_WRONG_VALUE)
            if (X_ELEMS_PER_THREAD == 4) {
                if ((jj + 3 * XXINCR)< jjLimit ) { // See if we can read one plus  3 shifted values?
                    XX[thid+ 0*XXINCR] = alpha * x_dev.m_pDevice[idx + 0 * incr];
                    XX[thid+ 1*XXINCR] = alpha * x_dev.m_pDevice[idx + 1 * incr];
                    XX[thid+ 2*XXINCR] = alpha * x_dev.m_pDevice[idx + 2 * incr];
                    XX[thid+ 3*XXINCR] = alpha * x_dev.m_pDevice[idx + 3 * incr];
                } else if ((jj + 2 * XXINCR) < jjLimit ) { // See if we can read one plus 2 shifted values?
                    XX[thid+ 0*XXINCR] = alpha * x_dev.m_pDevice[idx + 0 * incr];
                    XX[thid+ 1*XXINCR] = alpha * x_dev.m_pDevice[idx + 1 * incr];
                    XX[thid+ 2*XXINCR] = alpha * x_dev.m_pDevice[idx + 2 * incr];
                } else if ((jj + 1 * XXINCR) < jjLimit) { // See if we can read one plus 1 shifted values?
                    XX[thid+ 0*XXINCR] = alpha * x_dev.m_pDevice[idx + 0 * incr];
                    XX[thid+ 1*XXINCR] = alpha * x_dev.m_pDevice[idx + 1 * incr];
                } else if (jj < jjLimit) { // See if we can read one plus 0 shifted values?
                    XX[thid+0*XXINCR] = alpha * x_dev.m_pDevice[idx + 0 * incr];
                }
            } else if (X_ELEMS_PER_THREAD == 2) {
                if (jj + 1 * XXINCR < jjLimit) {// See if we can read one plus 1 shifted values?
                    XX[thid+ 0*XXINCR] = alpha * x_dev.m_pDevice[idx + 0 * incr];
                    XX[thid+ 1*XXINCR] = alpha * x_dev.m_pDevice[idx + 1 * incr];
                } else if (jj < jjLimit) { // See if we can read one plus 0 shifted values?
                    XX[thid+0*XXINCR] = alpha * x_dev.m_pDevice[idx + 0 * incr];
                }
            } else if (X_ELEMS_PER_THREAD == 1) {
                if (jj < jjLimit) {// See if we can read one plus 0 shifted values?
                    XX[thid+0*XXINCR] = alpha * x_dev.m_pDevice[idx + 0 * incr];
                }
            }

            // Syncronize again
            __syncthreads();

            row = ii + thid; // Global index into A_dev. The row which is processed by this thread!
            // Accumulate the dot product (all BLOCK_DIM Threads participate!)
            if(row < A_dev.m_M && thid < BLOCK_DIM) { // If this row is active
                PREC * A_ptr = PtrElem_ColM(A_dev,row,j); //  IDXA(row,j); // Start into A_dev for this block
                jjLimit = jjLimit - j; // represents the length of the dot product!
                jj=0;

                // Do block dot product in segments of DOT_PROD_SEGMENT
                // Register blocking, first load all stuff
                PREC regA[DOT_PROD_SEGMENT];
                PREC regx[DOT_PROD_SEGMENT];
                incr = A_dev.m_outerStrideBytes;
                while ( (jj + (DOT_PROD_SEGMENT-1)) < jjLimit) { // see if we can do one and DOT_PROD_SEGMENT-1 more values
                    // Dot product , 6er elements
#pragma unroll
                    for(int k = 0 ; k < DOT_PROD_SEGMENT ; k++) {
                        regA[k] = *(A_ptr);
                        regx[k] = XX[jj + k];
                        A_ptr = PtrColOffset_ColM(A_ptr,1,incr);
                    }
#pragma unroll
                    for(int k = 0 ; k < DOT_PROD_SEGMENT ; k++) {
                        dotp += regA[k] * regx[k] ;
                    }
                    // ==========================

                    jj     += DOT_PROD_SEGMENT;          // jump DOT_PROD_SEGMENT rows in x_dev
                    //A_ptr = PtrColOffset_ColM(A_ptr, DOT_PROD_SEGMENT, incr);   // jump DOT_PROD_SEGMENT cols in A_dev
                }
                // if no more DOT_PROD_SEGMENT segments are available do the rest
                while (jj < jjLimit) {
                    dotp += (*A_ptr) * XX[jj + 0];
                    jj   += 1;           // jump 1 rows in XX
                    A_ptr = PtrColOffset_ColM(A_ptr, 1, incr);   // jump 1 col in A_dev
                }

            } // Accumulate dot product

        } // shift blocks horizontally to obtain overall dot product!

        // Sync because we write in XX now , we may not write in XX is some threads are not finished with the dot product!
        __syncthreads();

        if( row < A_dev.m_M && thid < BLOCK_DIM) {
            idx = IDXB(row);
            if (beta != 0.0) {
                dotp += beta * b_dev.m_pDevice[idx];
            }
            // Write all values to shared memory back!
            XX[thid] = dotp; // Prox values are saved from 0 to BLOCK_DIM-1
        }

        // Sync threads in block because we need XX to be up to date for the new threads!
        __syncthreads();


        // DO PROX =================
        if(thid < PROX_PACKAGES) { // only PROX_PACKAGES do the prox!

            int package_start = i + blockIdx.x * BLOCK_DIM + thid * PROX_PACKAGE_SIZE; //start row for the prox package
            if(package_start < A_dev.m_M) { // See if start package index is not out of A_dev
                if ( TypeTraitsHelper::IsSame<TConvexSet, ConvexSets::RPlusAndDisk>::result ) {
                    // Do prox!==
                    int idx_n  = thid*PROX_PACKAGE_SIZE + 0;
                    int idx_t1 = thid*PROX_PACKAGE_SIZE + 1;
                    int idx_t2 = thid*PROX_PACKAGE_SIZE + 2;

                    //Normal direction
                    if(XX[idx_n] < 0) {
                        XX[idx_n] = 0;
                    }

                    // Tangential direction
                    PREC lambda_T1 =  XX[idx_t1];
                    PREC lambda_T2 =  XX[idx_t2];
                    PREC radius    =  mu_dev.m_pDevice[thid] * XX[idx_n];
                    PREC absvalue = (lambda_T1*lambda_T1 + lambda_T2*lambda_T2);

                    if(absvalue > radius * radius) {
                        if(TypeTraitsHelper::IsSame<PREC,double>::result) {
                            absvalue = radius * rsqrt(absvalue);
                        } else {
                            absvalue = radius * rsqrtf(absvalue);
                        }
                        XX[idx_t1] =  lambda_T1 * absvalue ;
                        XX[idx_t2] =  lambda_T2 * absvalue ;
                    }


                }
            }
        }
        // ========================

        // Collect all threads again and write the BLOCK_DIM values
        __syncthreads();

        if( row < A_dev.m_M && thid < BLOCK_DIM) {
            idx = IDXY(row);
            y_dev.m_pDevice[idx] = XX[thid];
        }
        //__syncthreads();

    } // Shift grid
    // ======================================================================================================================

    //

}


template<typename TCudaMatrix, int THREADS_PER_BLOCK, int BLOCK_DIM, int PROX_PACKAGES, typename TConvexSet>
__global__ void sorProxContactOrdered_1threads_StepA_kernel(
    TCudaMatrix mu_dev,
    TCudaMatrix x_new_dev,
    TCudaMatrix T_dev,
    TCudaMatrix d_dev,
    TCudaMatrix t_dev,
    int kernelAIdx,
    int maxNContacts,
    bool * convergedFlag_dev,
    typename TCudaMatrix::PREC _absTOL, typename TCudaMatrix::PREC _relTOL) {

    typedef typename TCudaMatrix::PREC PREC;
    STATIC_ASSERT( ! CudaMatrixFlags::isRowMajor< TCudaMatrix::Flags >::value )
    STATIC_ASSERT( (TypeTraitsHelper::IsSame<TConvexSet,ConvexSets::RPlusAndDisk>::result))

    __shared__ PREC xx[BLOCK_DIM]; // each thread writes one element, if its in the limit!!
    __shared__ PREC tt[BLOCK_DIM];


    // Assumend 1 Block, with THREADS_PER_BLOCK Threads and Column Major Matrix T_dev

    int thid = threadIdx.x;
    int m = min(maxNContacts*PROX_PACKAGE_SIZE, BLOCK_DIM); // this is the actual size of the diagonal block!
    int i = kernelAIdx * BLOCK_DIM;
    int ii = i + thid;

    // References to often used values in shared mem.
    PREC & xx_thid = xx[thid];
    PREC & tt_thid = tt[thid];

    //First copy t_dev in shared
    if(thid < m) {
        tt_thid = t_dev.m_pDevice[ii];
    }
    //__syncthreads();

    PREC d_value1, d_value2;
    if(thid<m) {
        d_value1 = d_dev.m_pDevice[ii];
    }
    if(thid<m-1) {
        d_value2 = d_dev.m_pDevice[ii+1];
    }


    int jj;
    //PREC T_iijj;
    //Offset the T_dev_ptr to the start of the Block
    PREC * T_dev_ptr  = PtrElem_ColM(T_dev,i,i);
    PREC * mu_dev_ptr = &mu_dev.m_pDevice[PROX_PACKAGES*kernelAIdx];

    for(int j_t = 0; j_t < m ; j_t+=PROX_PACKAGE_SIZE) {

        //Select the number of threads we need!

        // Here we process one [m x PROX_PACKAGE_SIZE] Block

        // First  Normal Direction ==========================================================
        jj =  i  +  j_t;


        if( ii == jj ) { // select thread on the diagonal ...

            PREC lambda_N = (d_value1 + tt_thid);

            //Prox Normal!
            if(lambda_N <= 0.0) {
                lambda_N = 0.0;
            }
            /* if( !checkConverged(x_new,xx_thid,absTOL,relTOL)){
            *convergedFlag_dev = 0;
            }*/

            xx_thid = lambda_N;
            tt_thid = 0.0;
        }
        // all threads not on the diagonal fall into this sync!
        __syncthreads();


        // Select only m threads!
        if(thid < m) {
            tt_thid += T_dev_ptr[thid] * xx[j_t];
        }
        // ====================================================================================
        // wee need to syncronize here because one threads finished lambda_t2 with shared mem tt, which is updated from another thread!
        __syncthreads();


        // Second  Tangential Direction ==========================================================
        jj++;
        if( ii == jj ) { // select thread on diagonal, one thread finishs T1 and T2 directions.

            // Prox tangential
            PREC lambda_T1 =  (d_value1 + tt_thid);
            PREC lambda_T2 =  (d_value2 + tt[thid+1]);

            PREC radius = (*mu_dev_ptr) * xx[thid-1];
            PREC absvalue = lambda_T1*lambda_T1 + lambda_T2*lambda_T2;

            if(absvalue > radius * radius) {
                if(TypeTraitsHelper::IsSame<PREC,double>::result) {
                    absvalue = radius * rsqrt(absvalue);
                } else {
                    absvalue = radius * rsqrtf(absvalue);
                }
                lambda_T1 *= absvalue;
                lambda_T2 *= absvalue;
            }

            /* if( !checkConverged(lambda_T1,xx_thid,absTOL,relTOL)){
            *convergedFlag_dev = 0;
            }

            if( !checkConverged(lambda_T2,x x[thid+1],absTOL,relTOL)){
            *convergedFlag_dev = 0;
            }*/

            //Write the two values back!
            xx_thid =  lambda_T1;
            tt_thid = 0.0;
            xx[thid+1] = lambda_T2;
            tt[thid+1] = 0.0;

        }
        // all threads not on the diagonal fall into this sync!
        __syncthreads();

        // Updating the pointers inside is better, then having two of this loops and updating the pointers between globally!
        if(thid < m) {
            T_dev_ptr = PtrColOffset_ColM(T_dev_ptr,1,T_dev.m_outerStrideBytes);
            tt_thid += T_dev_ptr[thid] * xx[j_t+1];
            T_dev_ptr = PtrColOffset_ColM(T_dev_ptr,1,T_dev.m_outerStrideBytes);
            tt_thid += T_dev_ptr[thid] * xx[j_t+2];
        }

        // ====================================================================================


        // move T_dev_ptr 1 column
        T_dev_ptr = PtrColOffset_ColM(T_dev_ptr,1,T_dev.m_outerStrideBytes);
        // move mu_ptr to nex contact
        mu_dev_ptr += 1;
    }

    // Write back the results, dont need to syncronize because
    // do it anyway to be safe for testing first!
    if(thid < m) {
        x_new_dev.m_pDevice[ii] = xx_thid;
        t_dev.m_pDevice[ii] = tt_thid;
    }


}


template<typename TCudaMatrix, int THREADS_PER_BLOCK, int BLOCK_DIM, int BLOCK_DIM_KERNEL_A, int X_ELEMS_PER_THREAD, int UNROLL_BLOCK_DOTPROD>
__global__ void sorProx_StepB_kernel( TCudaMatrix t_dev,  TCudaMatrix T_dev,  TCudaMatrix x_new_dev , int kernelAIdx) {

    typedef typename TCudaMatrix::PREC PREC;
    STATIC_ASSERT( ! CudaMatrixFlags::isRowMajor< TCudaMatrix::Flags >::value )

    __shared__ PREC XX[JINCR]; // Shared values for the x_new_dev;

    int thid = threadIdx.x;

    int /*i*/ j, ii, jj, row, jLimit, jjLimit, /*idx,*/ incr;
    // i: first level index in row direction to the start of the current grid (grid level),
    // ii: second level index in row direction to the start of the current block (block level),

    PREC dotp; // Dot product which is accumulated by 1 thread.

    // Shift the whole cuda grid over the rows!

    //for(i = 0 ; i < T_dev.m_M ; i += gridDim.x * BLOCK_DIM ){
    //i = 0;
    ii = blockIdx.x * BLOCK_DIM;   // Represents the row index to the current block start


    // Set the dot product to zero!
    dotp = 0;

    // Shift the block with dim[ BLOCK_DIM , (THREADS_PER_BLOCK*X_ELEMS_PER_THREAD)]
    // horizontally over T_dev, each thread calculates the sub dot product of
    // THREADS_PER_BLOCK*X_ELEMS_PER_THREAD elements.
    // j points to the beginning of the block and is shifted with JINCR = THREADS_PER_BLOCK*X_ELEMS_PER_THREAD
    jLimit = min( T_dev.m_N, BLOCK_DIM_KERNEL_A*(kernelAIdx+1));
    for(j = BLOCK_DIM_KERNEL_A*kernelAIdx ; j < jLimit; j += JINCR) {

        jj = j + thid;                                                        // Represents here the start index into x_new_dev for each thread!
        jjLimit = min (j + JINCR, jLimit);        // Represents here the maximal index for jj for this shifted block!

        // Syncronize before loading the (THREADS_PER_BLOCK*X_ELEMS_PER_THREAD) shared values of x_new_dev
        // Each thread loads X_ELEMS_PER_THREAD elements.
        __syncthreads();

        // Load loop (all THREADS_PER_BLOCK participate!)
        // Each of this BLOCK_DIM threads goes into another if statement if the
        //incr = incr_x * XXINCR;
        //idx = IDXX(jj);

        STATIC_ASSERTM(X_ELEMS_PER_THREAD == 4 || X_ELEMS_PER_THREAD == 3 || X_ELEMS_PER_THREAD == 2 || X_ELEMS_PER_THREAD == 1, X_ELEMS_PER_THREAD_WRONG_VALUE)
        if (X_ELEMS_PER_THREAD == 4) {
            if ((jj + 3 * XXINCR)< jjLimit ) { // See if we can read one plus  3 shifted values?
                XX[thid+ 0*XXINCR] = x_new_dev.m_pDevice[jj + 0 * XXINCR];
                XX[thid+ 1*XXINCR] = x_new_dev.m_pDevice[jj + 1 * XXINCR];
                XX[thid+ 2*XXINCR] = x_new_dev.m_pDevice[jj + 2 * XXINCR];
                XX[thid+ 3*XXINCR] = x_new_dev.m_pDevice[jj + 3 * XXINCR];
            } else if ((jj + 2 * XXINCR) < jjLimit ) { // See if we can read one plus 2 shifted values?
                XX[thid+ 0*XXINCR] = x_new_dev.m_pDevice[jj + 0 * XXINCR];
                XX[thid+ 1*XXINCR] = x_new_dev.m_pDevice[jj + 1 * XXINCR];
                XX[thid+ 2*XXINCR] = x_new_dev.m_pDevice[jj + 2 * XXINCR];
            } else if ((jj + 1 * XXINCR) < jjLimit) { // See if we can read one plus 1 shifted values?
                XX[thid+ 0*XXINCR] =  x_new_dev.m_pDevice[jj + 0 * XXINCR];
                XX[thid+ 1*XXINCR] =  x_new_dev.m_pDevice[jj + 1 * XXINCR];
            } else if (jj < jjLimit) { // See if we can read one plus 0 shifted values?
                XX[thid+0*XXINCR] = x_new_dev.m_pDevice[jj + 0 * XXINCR];
            }
        }
        if (X_ELEMS_PER_THREAD == 3) {
            if ((jj + 2 * XXINCR) < jjLimit ) {// See if we can read one plus 2 shifted values?
                XX[thid+ 0*XXINCR] = x_new_dev.m_pDevice[jj + 0 * XXINCR];
                XX[thid+ 1*XXINCR] = x_new_dev.m_pDevice[jj + 1 * XXINCR];
                XX[thid+ 2*XXINCR] = x_new_dev.m_pDevice[jj + 2 * XXINCR];
            } else if ((jj + 1 * XXINCR) < jjLimit) { // See if we can read one plus 1 shifted values?
                XX[thid+ 0*XXINCR] =  x_new_dev.m_pDevice[jj + 0 * XXINCR];
                XX[thid+ 1*XXINCR] =  x_new_dev.m_pDevice[jj + 1 * XXINCR];
            } else if (jj < jjLimit) { // See if we can read one plus 0 shifted values?
                XX[thid+0*XXINCR] = x_new_dev.m_pDevice[jj + 0 * XXINCR];
            }
        } else if (X_ELEMS_PER_THREAD == 2) {
            if (jj + 1 * XXINCR < jjLimit) {// See if we can read one plus 1 shifted values?
                XX[thid+ 0*XXINCR] =  x_new_dev.m_pDevice[jj + 0 * XXINCR];
                XX[thid+ 1*XXINCR] =  x_new_dev.m_pDevice[jj + 1 * XXINCR];
            } else if (jj < jjLimit) { // See if we can read one plus 0 shifted values?
                XX[thid+0*XXINCR] =  x_new_dev.m_pDevice[jj + 0 * XXINCR];
            }
        } else if  (X_ELEMS_PER_THREAD == 1) {
            if (jj < jjLimit) {     // See if we can read one plus 0 shifted values?
                XX[thid+0*XXINCR] = x_new_dev.m_pDevice[jj + 0 * XXINCR];
            }
        }


        // Syncronize again
        __syncthreads();

        row = ii + thid; // Global index into T_dev. The row which is processed by this thread!
        if( row >= kernelAIdx * BLOCK_DIM_KERNEL_A ) {
            row += BLOCK_DIM_KERNEL_A; // skip diagonal block!
        }

        // Accumulate the dot product (all BLOCK_DIM Threads participate!)
        if(row < T_dev.m_M && thid < BLOCK_DIM) { // If this row is active
            PREC * A_ptr = PtrElem_ColM(T_dev,row,j); //  IDXA(row,j); // Start into T_dev for this block
            jjLimit = jjLimit - j;                    // represents the length of the dot product!
            jj=0;

            // Do block dot product in segments of DOT_PROD_SEGMENT
            // Register blocking, first load all stuff
            PREC regA[DOT_PROD_SEGMENT];
            PREC regx[DOT_PROD_SEGMENT];
            incr = T_dev.m_outerStrideBytes;
            while ( (jj + (DOT_PROD_SEGMENT-1)) < jjLimit) { // see if we can do one and DOT_PROD_SEGMENT-1 more values
                // Dot product , 6er elements
#pragma unroll
                for(int k = 0 ; k < DOT_PROD_SEGMENT ; k++) {
                    regA[k] = *(A_ptr);
                    regx[k] = XX[jj + k];
                    A_ptr = PtrColOffset_ColM(A_ptr,1,incr);
                }
#pragma unroll
                for(int k = 0 ; k < DOT_PROD_SEGMENT ; k++) {
                    dotp += regA[k] * regx[k] ;
                }
                // ==========================

                jj     += DOT_PROD_SEGMENT;          // jump DOT_PROD_SEGMENT rows in x_new_dev
                //A_ptr = PtrColOffset_ColM(A_ptr, DOT_PROD_SEGMENT, incr);   // jump DOT_PROD_SEGMENT cols in T_dev
            }
            // if no more DOT_PROD_SEGMENT segments are available do the rest
            while (jj < jjLimit) {
                dotp += (*A_ptr) * XX[jj + 0];
                jj   += 1;           // jump 1 rows in XX
                A_ptr = PtrColOffset_ColM(A_ptr, 1, incr);   // jump 1 col in T_dev
            }

        } // Accumulate dot product
    } // shift blocks horizontally to obtain overall dot product!


    if( row < T_dev.m_M && thid < BLOCK_DIM) {
        //idx = IDXB(row);
        /*if (beta != 0.0) {
        dotp += beta * b_dev.m_pDevice[row];
        }*/
        //idx = IDXY(row);
        t_dev.m_pDevice[row] += dotp;
    }

    //} // Shift grid

}



/**
* Solves: proxTerm = prox_C( proxTerm + b).
* Type: Contact Ordered. (CO)
* Approximate Flop Count: Flops = nContacs * ( 1 + 1 + 3 + 1*FLOPS_SQRT + 2 + 2*FLOPS_DIVISION)
* With assignment: Flops += 7
*/
template<typename TCudaMatrix, int THREADS_PER_BLOCK, int BLOCK_DIM, int PROX_PACKAGES, typename TConvexSet>
__global__ void sorProxRelaxed_StepA_kernel(TCudaMatrix mu_dev, TCudaMatrix x_new_dev, TCudaMatrix t_dev, TCudaMatrix d_dev, int kernelAIdx) {
    typedef typename TCudaMatrix::PREC PREC;
    STATIC_ASSERT( (TypeTraitsHelper::IsSame<TConvexSet,ConvexSets::RPlusAndDisk>::result))
    STATIC_ASSERT( ! CudaMatrixFlags::isRowMajor< TCudaMatrix::Flags >::value )

    int thid  = threadIdx.x;
    int contactIdx = (blockIdx.x * THREADS_PER_BLOCK + thid); // each thread per block tries to prox one packages if its in the range
    int ii = TConvexSet::Dimension*(contactIdx) + BLOCK_DIM * kernelAIdx;
    contactIdx += kernelAIdx*PROX_PACKAGES;
    int iiLimit = min(t_dev.m_M,  BLOCK_DIM * (kernelAIdx+1));

    if ( ii < iiLimit ) { // only PROX_PACKAGES threads do the prox! THREADS_PER_BLOCK might be bigger to fit nicely into an SM
        // no Coalesced read, stupid, but lambda vector can not be ordered better, for example Type Ordered (with a certain pitch) becaus then the multiplication becomes bad
        PREC lambda_N =   (d_dev.m_pDevice[ii]   + t_dev.m_pDevice[ii]);
        PREC lambda_T1 =  (d_dev.m_pDevice[ii+1] + t_dev.m_pDevice[ii+1]);
        PREC lambda_T2 =  (d_dev.m_pDevice[ii+2] + t_dev.m_pDevice[ii+2]);

        //Prox Normal!
        if(lambda_N <= 0.0) {
            lambda_N = 0.0;
        }

        // Prox tangential
        PREC radius = (mu_dev.m_pDevice[contactIdx]) * lambda_N;
        PREC absvalue = lambda_T1*lambda_T1 + lambda_T2*lambda_T2;

        if(absvalue > radius*radius) {
            if(TypeTraitsHelper::IsSame<PREC,double>::result) {
                absvalue = radius * rsqrt(absvalue);
            } else {
                absvalue = radius * rsqrtf(absvalue);
            }
            lambda_T1   *= absvalue;
            lambda_T2   *= absvalue;
        }

        //Write the three values back! Not coalesced which is stupid
        x_new_dev.m_pDevice[ii+0] =  lambda_N;
        x_new_dev.m_pDevice[ii+1] =  lambda_T1;
        x_new_dev.m_pDevice[ii+2] =  lambda_T2;
        t_dev.m_pDevice[ii+0] =  0.0;
        t_dev.m_pDevice[ii+1] =  0.0;
        t_dev.m_pDevice[ii+2] =  0.0;
        // ==========
    }

}

template<typename TCudaMatrix, int THREADS_PER_BLOCK, int BLOCK_DIM, int BLOCK_DIM_KERNEL_A, int X_ELEMS_PER_THREAD, int UNROLL_BLOCK_DOTPROD>
__global__ void sorProxRelaxed_StepB_kernel( TCudaMatrix t_dev,  TCudaMatrix T_dev,  TCudaMatrix x_new_dev , int kernelAIdx) {

    typedef typename TCudaMatrix::PREC PREC;
    STATIC_ASSERT( ! CudaMatrixFlags::isRowMajor< TCudaMatrix::Flags >::value )

    __shared__ PREC XX[JINCR]; // Shared values for the x_new_dev;

    int thid = threadIdx.x;

    int /*i*/ j, ii, jj, row, jLimit, jjLimit, /*idx,*/ incr;
    // i: first level index in row direction to the start of the current grid (grid level),
    // ii: second level index in row direction to the start of the current block (block level),

    PREC dotp; // Dot product which is accumulated by 1 thread.

    // Shift the whole cuda grid over the rows!

    //for(i = 0 ; i < T_dev.m_M ; i += gridDim.x * BLOCK_DIM ){
    //i = 0;
    ii = blockIdx.x * BLOCK_DIM;   // Represents the row index to the current block start


    // Set the dot product to zero!
    dotp = 0;

    // Shift the block with dim[ BLOCK_DIM , (THREADS_PER_BLOCK*X_ELEMS_PER_THREAD)]
    // horizontally over T_dev, each thread calculates the sub dot product of
    // THREADS_PER_BLOCK*X_ELEMS_PER_THREAD elements.
    // j points to the beginning of the block and is shifted with JINCR = THREADS_PER_BLOCK*X_ELEMS_PER_THREAD
    jLimit = min( T_dev.m_N, BLOCK_DIM_KERNEL_A*(kernelAIdx+1));
    for(j = BLOCK_DIM_KERNEL_A*kernelAIdx ; j < jLimit; j += JINCR) {

        jj = j + thid;                                                          // Represents here the start index into x_new_dev for each thread!
        jjLimit = min (j + JINCR, jLimit);                                      // Represents here the maximal index for jj for this shifted block!

        // Syncronize before loading the (THREADS_PER_BLOCK*X_ELEMS_PER_THREAD) shared values of x_new_dev
        // Each thread loads X_ELEMS_PER_THREAD elements.
        __syncthreads();

        // Load loop (all THREADS_PER_BLOCK participate!)
        // Each of this BLOCK_DIM threads goes into another if statement if the
        //incr = incr_x * XXINCR;
        //idx = IDXX(jj);

        STATIC_ASSERTM(X_ELEMS_PER_THREAD == 4 || X_ELEMS_PER_THREAD == 3 || X_ELEMS_PER_THREAD == 2 || X_ELEMS_PER_THREAD == 1, X_ELEMS_PER_THREAD_WRONG_VALUE)
        if (X_ELEMS_PER_THREAD == 4) {
            if ((jj + 3 * XXINCR)< jjLimit ) { // See if we can read one plus  3 shifted values?
                XX[thid+ 0*XXINCR] = x_new_dev.m_pDevice[jj + 0 * XXINCR];
                XX[thid+ 1*XXINCR] = x_new_dev.m_pDevice[jj + 1 * XXINCR];
                XX[thid+ 2*XXINCR] = x_new_dev.m_pDevice[jj + 2 * XXINCR];
                XX[thid+ 3*XXINCR] = x_new_dev.m_pDevice[jj + 3 * XXINCR];
            } else if ((jj + 2 * XXINCR) < jjLimit ) { // See if we can read one plus 2 shifted values?
                XX[thid+ 0*XXINCR] = x_new_dev.m_pDevice[jj + 0 * XXINCR];
                XX[thid+ 1*XXINCR] = x_new_dev.m_pDevice[jj + 1 * XXINCR];
                XX[thid+ 2*XXINCR] = x_new_dev.m_pDevice[jj + 2 * XXINCR];
            } else if ((jj + 1 * XXINCR) < jjLimit) { // See if we can read one plus 1 shifted values?
                XX[thid+ 0*XXINCR] =  x_new_dev.m_pDevice[jj + 0 * XXINCR];
                XX[thid+ 1*XXINCR] =  x_new_dev.m_pDevice[jj + 1 * XXINCR];
            } else if (jj < jjLimit) { // See if we can read one plus 0 shifted values?
                XX[thid+0*XXINCR] = x_new_dev.m_pDevice[jj + 0 * XXINCR];
            }
        }
        if (X_ELEMS_PER_THREAD == 3) {
            if ((jj + 2 * XXINCR) < jjLimit ) {// See if we can read one plus 2 shifted values?
                XX[thid+ 0*XXINCR] = x_new_dev.m_pDevice[jj + 0 * XXINCR];
                XX[thid+ 1*XXINCR] = x_new_dev.m_pDevice[jj + 1 * XXINCR];
                XX[thid+ 2*XXINCR] = x_new_dev.m_pDevice[jj + 2 * XXINCR];
            } else if ((jj + 1 * XXINCR) < jjLimit) { // See if we can read one plus 1 shifted values?
                XX[thid+ 0*XXINCR] =  x_new_dev.m_pDevice[jj + 0 * XXINCR];
                XX[thid+ 1*XXINCR] =  x_new_dev.m_pDevice[jj + 1 * XXINCR];
            } else if (jj < jjLimit) { // See if we can read one plus 0 shifted values?
                XX[thid+0*XXINCR] = x_new_dev.m_pDevice[jj + 0 * XXINCR];
            }
        } else if (X_ELEMS_PER_THREAD == 2) {
            if (jj + 1 * XXINCR < jjLimit) {// See if we can read one plus 1 shifted values?
                XX[thid+ 0*XXINCR] =  x_new_dev.m_pDevice[jj + 0 * XXINCR];
                XX[thid+ 1*XXINCR] =  x_new_dev.m_pDevice[jj + 1 * XXINCR];
            } else if (jj < jjLimit) { // See if we can read one plus 0 shifted values?
                XX[thid+0*XXINCR] =  x_new_dev.m_pDevice[jj + 0 * XXINCR];
            }
        } else if  (X_ELEMS_PER_THREAD == 1) {
            if (jj < jjLimit) {     // See if we can read one plus 0 shifted values?
                XX[thid+0*XXINCR] = x_new_dev.m_pDevice[jj + 0 * XXINCR];
            }
        }


        // Syncronize again
        __syncthreads();

        row = ii + thid; // Global index into T_dev. The row which is processed by this thread!
        //if( row >= kernelAIdx * BLOCK_DIM_KERNEL_A ){
        //    row += BLOCK_DIM_KERNEL_A; // skip diagonal block!
        //}

        // Accumulate the dot product (all BLOCK_DIM Threads participate!)
        if(row < T_dev.m_M && thid < BLOCK_DIM) { // If this row is active
            PREC * A_ptr = PtrElem_ColM(T_dev,row,j); //  IDXA(row,j); // Start into T_dev for this block
            jjLimit = jjLimit - j;                    // represents the length of the dot product!
            jj=0;

            // Do block dot product in segments of DOT_PROD_SEGMENT
            // Register blocking, first load all stuff
            PREC regA[DOT_PROD_SEGMENT];
            PREC regx[DOT_PROD_SEGMENT];
            incr = T_dev.m_outerStrideBytes;
            while ( (jj + (DOT_PROD_SEGMENT-1)) < jjLimit) { // see if we can do one and DOT_PROD_SEGMENT-1 more values
                // Dot product , 6er elements
#pragma unroll
                for(int k = 0 ; k < DOT_PROD_SEGMENT ; k++) {
                    regA[k] = *(A_ptr);
                    regx[k] = XX[jj + k];
                    A_ptr = PtrColOffset_ColM(A_ptr,1,incr);
                }
#pragma unroll
                for(int k = 0 ; k < DOT_PROD_SEGMENT ; k++) {
                    dotp += regA[k] * regx[k] ;
                }
                // ==========================

                jj     += DOT_PROD_SEGMENT;          // jump DOT_PROD_SEGMENT rows in x_new_dev
                //A_ptr = PtrColOffset_ColM(A_ptr, DOT_PROD_SEGMENT, incr);   // jump DOT_PROD_SEGMENT cols in T_dev
            }
            // if no more DOT_PROD_SEGMENT segments are available do the rest
            while (jj < jjLimit) {
                dotp += (*A_ptr) * XX[jj + 0];
                jj   += 1;           // jump 1 rows in XX
                A_ptr = PtrColOffset_ColM(A_ptr, 1, incr);   // jump 1 col in T_dev
            }

        } // Accumulate dot product
    } // shift blocks horizontally to obtain overall dot product!


    if( row < T_dev.m_M && thid < BLOCK_DIM) {
        //idx = IDXB(row);
        /*if (beta != 0.0) {
        dotp += beta * b_dev.m_pDevice[row];
        }*/
        //idx = IDXY(row);
        t_dev.m_pDevice[row] += dotp;
    }

    //} // Shift grid



}




template<typename TCudaMatrix>
__global__ void convergedEach_kernel(TCudaMatrix x_new_dev,
                                     TCudaMatrix x_olb_dev,
                                     bool * convergedFlag_dev,
                                     typename TCudaMatrix::PREC absTOL, typename TCudaMatrix::PREC relTOL) {


    typedef typename TCudaMatrix::PREC PREC;
    // Calculate  indexes
    int index_x = threadIdx.x + blockIdx.x * blockDim.x;
    int stride_x = gridDim.x * blockDim.x;

    while(index_x < x_olb_dev.m_M) {

        if(!checkConverged(x_new_dev.m_pDevice[index_x], x_olb_dev.m_pDevice[index_x] , absTOL, relTOL)) {
            *convergedFlag_dev = 0;
        }

        index_x+= stride_x;
    }

}


template<typename TCudaMatrix, typename TConvexSet>
__global__ void csrBlockSparseFullSORProx(TCudaMatrix t_dev,
                                          TCudaMatrix csrT_dev,
                                          TCudaMatrix x_new_dev,
                                          TCudaMatrix d_dev,
                                          TCudaMatrix mu_dev,
                                          unsigned int *csrRowPtrT,
                                          unsigned int * csrColIndT,
                                          unsigned int* atomicCounter) {
    typedef typename TCudaMatrix::PREC PREC;

    STATIC_ASSERT( ! CudaMatrixFlags::isRowMajor< TCudaMatrix::Flags >::value )
    STATIC_ASSERT( (TypeTraitsHelper::IsSame<TConvexSet,ConvexSets::RPlusAndDisk>::result))
    // Each Cuda Block computes one block row!
    int thid = threadIdx.x;
    int row = blockIdx.x;

    int startBlock = csrRowPtrT[row];
    int endBlock = csrRowPtrT[row+1];

    const int dimBlock = ConvexSets::RPlusAndDisk::Dimension;

    for (int block = startBlock ; block < endBlock; block++) {

        int col =  csrColIndT[block];
        PREC * T_ij = csrT_dev.m_pDevice[block*dimBlock*dimBlock];
        int idx = row*3;

        // If diagonal block!
        if(col == row) {

            // only 1 thread
            if(thid==0) {
                PREC lambda_N = (d_dev.m_pDevice[idx] + t_dev.m_pDevice[idx]);
                //Prox Normal!
                if(lambda_N <= 0.0) {
                    lambda_N = 0.0;
                }
                x_new_dev.m_pDevice[idx] = lambda_N;
                t_dev.m_pDevice[idx] = 0.0;
            }
            // all threads not on the diagonal fall into this sync!
            __syncthreads();

            // Select only 3 threads! Update Normal!
            if(thid < 3) {
                t_dev.m_pDevice[idx+thid] += T_ij[0+thid] * x_new_dev.m_pDevice[idx];
            }

            // ====================================================================================
            // wee need to syncronize here because one threads finished lambda_t2 with shared mem tt, which is updated from another thread!
            __syncthreads();


            // Second  Tangential Direction ==========================================================
            if( thid == 0 ) { // select thread on diagonal, one thread finishs T1 and T2 directions.

                // Prox tangential
                PREC lambda_T1 =  (d_dev.m_pDevice[idx+1] + t_dev.m_pDevice[idx+1]);
                PREC lambda_T2 =  (d_dev.m_pDevice[idx+2] + t_dev.m_pDevice[idx+2]);
                PREC radius = (mu_dev.m_pDevice[row]) * x_new_dev.m_pDevice[idx];
                PREC absvalue = lambda_T1*lambda_T1 + lambda_T2*lambda_T2;

                if(absvalue > radius * radius) {
                    if(TypeTraitsHelper::IsSame<PREC,double>::result) {
                        absvalue = radius * rsqrt(absvalue);
                    } else {
                        absvalue = radius * rsqrtf(absvalue);
                    }
                    lambda_T1 *= absvalue;
                    lambda_T2 *= absvalue;
                }
                //Write the two values back!
                x_new_dev.m_pDevice[idx+1] =  lambda_T1;
                t_dev.m_pDevice[idx+1] = 0.0;
                x_new_dev.m_pDevice[idx+2]  = lambda_T2;
                t_dev.m_pDevice[thid+2] = 0.0;

            }
            __syncthreads();

            //Update with 3 threads
            // Select only 3 threads! Update Normal!
            if(thid < 3) {
                t_dev.m_pDevice[idx+thid] += T_ij[thid+dimBlock*1] * x_new_dev.m_pDevice[idx+1];
                t_dev.m_pDevice[idx+thid] += T_ij[thid+dimBlock*2] * x_new_dev.m_pDevice[idx+2];
            }

            if(thid==0) {
                //Update atomic counter! to signal that a new diagonal block has been computed
                atomicAdd(atomicCounter,1);
            }

        } else {
            // Off diagonal block , wait till atomic counter ==  col then multiply
            if(thid == 0) {
                while(atomicAdd(atomicCounter,0) != col+1); //Spinn loop till the the diagonal has been computed!
            }
            __syncthreads();

            // Multiply Block
            if(thid < 3) {
                t_dev.m_pDevice[idx+thid] += T_ij[thid] * x_new_dev.m_pDevice[idx];
                t_dev.m_pDevice[idx+thid] += T_ij[thid+dimBlock*1] * x_new_dev.m_pDevice[idx+1];
                t_dev.m_pDevice[idx+thid] += T_ij[thid+dimBlock*2] * x_new_dev.m_pDevice[idx+2];
            }

        }

    }
}


}




#endif
