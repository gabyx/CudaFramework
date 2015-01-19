// ========================================================================================
//  CudaFramework
//  Copyright (C) 2014 by Gabriel Nützi <nuetzig (at) imes (d0t) mavt (d0t) ethz (d0t) ch>
//
//  This Source Code Form is subject to the terms of the GNU GPL 3.0 licence.
//  If a copy of the GNU GPL 3.0 was not distributed with this
//  file, you can obtain one at http://opensource.org/licenses/GPL-3.0.
// ========================================================================================

#ifndef CudaFramework_Kernels_MatrixVectorMultGPU_KernelsMatrixVectorMult_cuh
#define CudaFramework_Kernels_MatrixVectorMultGPU_KernelsMatrixVectorMult_cuh

#include "CudaFramework/General/GPUMutex.hpp"
#include "CudaFramework/CudaModern/CudaMatrix.hpp"

namespace matrixVectorMultKernels{


   /**
   * Global syncronization function, for all blocks!
   */
   __forceinline__ __device__ void syncronizeGlobal(GPUAtomicCounter & global_sync, int threadIdxX,  int blockIdxX, int gridDimX){

      if (threadIdxX == 0){
            atomicAdd(global_sync.counter, 1);
           if (blockIdxX == 0){
             while(atomicAdd(global_sync.counter,0) < gridDimX);
             atomicExch(global_sync.counter,0);
           }else{
             while(atomicAdd(global_sync.counter,0) > 0);
           }
      }
       __syncthreads();
   }

#define X_ELEMS_PER_THREAD (4)
#define THREADS_PER_BLOCK (128) 	// How many threads we have assigned to each block ( 128 is Blas configuration!)

#define BLOCK_DIM (128) 		// How many threads calculate 1 result in y_dev, only BLOCK_DIM results are computed. THREADS_PER_BLOCK - BLOCK_DIM threads are idle!. Can also be less than THREADS_PER_BLOCK
                              // 128 is Blas configuration!

#if (BLOCK_DIM > THREADS_PER_BLOCK)
#error Block dim needs to be smaller or equal to the amount of threads per block
#endif
#define UNROLL_BLOCK_DOTPROD (6)
#define DOT_PROD_SEGMENT (UNROLL_BLOCK_DOTPROD)

#define XXINCR (THREADS_PER_BLOCK)
#define JINCR (X_ELEMS_PER_THREAD * THREADS_PER_BLOCK)
#define IDXX(row) ((row)*incr_x)
#define IDXB(row) ((row)*incr_b)
#define IDXY(row) ((row)*incr_y)
   /*
   * @brief This multiplication has been adapted from CUBLAS, renamed and commented, because the CUBLAS code was hard to understand!
   * Output y_dev MUST not be the same as x_dev!
   */
   template<typename TCudaMatrix>
   __global__ void matrixVectorMultiply_kernel( TCudaMatrix y_dev,
                                                int incr_y,
                                                typename TCudaMatrix::PREC alpha,
                                                TCudaMatrix A_dev,
                                                TCudaMatrix x_dev,
                                                int incr_x,
                                                typename TCudaMatrix::PREC beta,
                                                TCudaMatrix b_dev,
                                                int incr_b){

      typedef typename TCudaMatrix::PREC PREC;

      __shared__ PREC XX[JINCR]; // Shared values for the x_dev;

      int thid = threadIdx.x;

      int i,j, ii, jj, row,  jjLimit, idx, incr;
      // i: first level index in row direction to the start of the current grid (grid level),
      // ii: second level index in row direction to the start of the current block (block level),

      PREC dotp; // Dot product which is accumulated by 1 thread.

      // Shift the whole cuda grid over the rows!

      for(i = 0 ; i < A_dev.m_M ; i += gridDim.x * BLOCK_DIM ){

         ii = i + blockIdx.x * BLOCK_DIM;   // Represents the row index to the current block start
         if( ii >= A_dev.m_M) break;  // The blocks which lie outside the matrix can be rejected already here
         // Set the dot product to zero!
         dotp = 0;

         // Shift the block with dim[ BLOCK_DIM , (THREADS_PER_BLOCK*X_ELEMS_PER_THREAD)]
         // horizontally over A_dev, each thread calculates the sub dot product of
         // THREADS_PER_BLOCK*X_ELEMS_PER_THREAD elements.
         // j points to the beginning of the block and is shifted with JINCR = THREADS_PER_BLOCK*X_ELEMS_PER_THREAD
         for(j = 0 ; j < A_dev.m_N; j += JINCR){

            jj = j + thid;                            // Represents here the start index into x_dev for each thread!
            jjLimit = min (j + JINCR, A_dev.m_N);       // Represents here the maximal index for jj for this shifted block!

            // Syncronize before loading the (THREADS_PER_BLOCK*X_ELEMS_PER_THREAD) shared values of x_dev
            // Each thread loads X_ELEMS_PER_THREAD elements.
            __syncthreads();

            // Load loop (all THREADS_PER_BLOCK participate!)
            // Each of this BLOCK_DIM threads goes into another if statement if the
            incr = incr_x * XXINCR;
            idx = IDXX(jj);
 #if (X_ELEMS_PER_THREAD == 4)
            if ((jj + 3 * XXINCR)< jjLimit ) { // See if we can read one plus  3 shifted values?
                XX[thid+ 0*XXINCR] = alpha * x_dev.m_pDevice[idx + 0 * incr];
                XX[thid+ 1*XXINCR] = alpha * x_dev.m_pDevice[idx + 1 * incr];
                XX[thid+ 2*XXINCR] = alpha * x_dev.m_pDevice[idx + 2 * incr];
                XX[thid+ 3*XXINCR] = alpha * x_dev.m_pDevice[idx + 3 * incr];
            }
            else if ((jj + 2 * XXINCR) < jjLimit ) {// See if we can read one plus 2 shifted values?
                XX[thid+ 0*XXINCR] = alpha * x_dev.m_pDevice[idx + 0 * incr];
                XX[thid+ 1*XXINCR] = alpha * x_dev.m_pDevice[idx + 1 * incr];
                XX[thid+ 2*XXINCR] = alpha * x_dev.m_pDevice[idx + 2 * incr];
            }
            else if ((jj + 1 * XXINCR) < jjLimit) {// See if we can read one plus 1 shifted values?
                XX[thid+ 0*XXINCR] = alpha * x_dev.m_pDevice[idx + 0 * incr];
                XX[thid+ 1*XXINCR] = alpha * x_dev.m_pDevice[idx + 1 * incr];
            }
            else if (jj < jjLimit) {// See if we can read one plus 0 shifted values?
                XX[thid+0*XXINCR] = alpha * x_dev.m_pDevice[idx + 0 * incr];
            }
#elif (X_ELEMS_PER_THREAD == 2)
            if (jj + 1 * XXINCR < jjLimit) {// See if we can read one plus 1 shifted values?
                XX[thid+ 0*XXINCR] = alpha * x_dev.m_pDevice[idx + 0 * incr];
                XX[thid+ 1*XXINCR] = alpha * x_dev.m_pDevice[idx + 1 * incr];
            }
            else if (jj < jjLimit) {// See if we can read one plus 0 shifted values?
                XX[thid+0*XXINCR] = alpha * x_dev.m_pDevice[idx + 0 * incr];
            }
#elif (X_ELEMS_PER_THREAD == 1)
            if (jj < jjLimit) {// See if we can read one plus 0 shifted values?
                XX[thid+0*XXINCR] = alpha * x_dev.m_pDevice[idx + 0 * incr];
            }
#else
#error Current code cannot handle X_ELEMS_PER_THREAD != 4
#endif

            // Syncronize again
            __syncthreads();

			   row = ii + thid; // Global index into A_dev. The row which is processed by this thread!
            // Accumulate the dot product (all BLOCK_DIM Threads participate!)
            if(row < A_dev.m_M && thid < BLOCK_DIM){ // If this row is active
               PREC * A_ptr = PtrElem_ColM(A_dev,row,j); //  IDXA(row,j); // Start into A_dev for this block
               jjLimit = jjLimit - j; // represents the length of the dot product!
               jj=0;

               // Do block dot product in segments of DOT_PROD_SEGMENT
               // Register blocking, first load all stuff
               PREC regA[DOT_PROD_SEGMENT]; PREC regx[DOT_PROD_SEGMENT];
               incr = A_dev.m_outerStrideBytes;
               while ( (jj + (DOT_PROD_SEGMENT-1)) < jjLimit) { // see if we can do one and DOT_PROD_SEGMENT-1 more values
                  // Dot product , 6er elements
                  #pragma unroll
                  for(int k = 0 ; k < DOT_PROD_SEGMENT ; k++){
                     regA[k] = *(A_ptr);
                     regx[k] = XX[jj + k];
                     A_ptr = PtrColOffset_ColM(A_ptr,1,incr);
                  }
                  #pragma unroll
                  for(int k = 0 ; k < DOT_PROD_SEGMENT ; k++){
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


         if( row < A_dev.m_M && thid < BLOCK_DIM){
            idx = IDXB(row);
            if (beta != 0.0) {
                dotp += beta * b_dev.m_pDevice[idx];
            }
            idx = IDXY(row);
           y_dev.m_pDevice[idx] = dotp;
         }

      } // Shift grid

   }


    /*
   * @brief This multiplication has been adapted from CUBLAS, renamed and commented, because the CUBLAS code was hard to understand!
   * This is the symetric version, to test the L1/L2 cache
   * Not finished!! TODO!
   * Output y_dev MUST not be the same as x_dev!
   */
   template<typename TCudaMatrix>
   __global__ void matrixVectorMultiplySym_kernel( TCudaMatrix y_dev,
                                                int incr_y,
                                                typename TCudaMatrix::PREC alpha,
                                                TCudaMatrix A_dev,
                                                TCudaMatrix x_dev,
                                                int incr_x,
                                                typename TCudaMatrix::PREC beta,
                                                TCudaMatrix b_dev,
                                                int incr_b){
    typedef typename TCudaMatrix::PREC PREC;
      __shared__ PREC XX[JINCR]; // Shared values for the x_dev;

      int thid = threadIdx.x;

      int i,j, ii, jj, row,  jjLimit, idx, incr;
      // i: first level index in row direction to the start of the current grid (grid level),
      // ii: second level index in row direction to the start of the current block (block level),

      PREC dotp; // Dot product which is accumulated by 1 thread.

      // Shift the whole cuda grid over the rows!

      for(i = 0 ; i < A_dev.m_M ; i += gridDim.x * BLOCK_DIM ){

         ii = i + blockIdx.x * BLOCK_DIM;   // Represents the row index to the current block start
         if( ii >= A_dev.m_M) break;  // The blocks which lie outside the matrix can be rejected already here
         // Set the dot product to zero!
         dotp = 0;

         // Shift the block with dim[ BLOCK_DIM , (THREADS_PER_BLOCK*X_ELEMS_PER_THREAD)]
         // horizontally over A_dev, each thread calculates the sub dot product of
         // THREADS_PER_BLOCK*X_ELEMS_PER_THREAD elements.
         // j points to the beginning of the block and is shifted with JINCR = THREADS_PER_BLOCK*X_ELEMS_PER_THREAD
         for(j = 0 ; j < A_dev.m_N; j += JINCR){

            jj = j + thid;                            // Represents here the start index into x_dev for each thread!
            jjLimit = min (j + JINCR, A_dev.m_N);       // Represents here the maximal index for jj for this shifted block!

            // Syncronize before loading the (THREADS_PER_BLOCK*X_ELEMS_PER_THREAD) shared values of x_dev
            // Each thread loads X_ELEMS_PER_THREAD elements.
            __syncthreads();

            // Load loop (all THREADS_PER_BLOCK participate!)
            // Each of this BLOCK_DIM threads goes into another if statement if the
            incr = incr_x * XXINCR;
            idx = IDXX(jj);
 #if (X_ELEMS_PER_THREAD == 4)
            if ((jj + 3 * XXINCR)< jjLimit ) { // See if we can read one plus  3 shifted values?
                XX[thid+ 0*XXINCR] = alpha * x_dev.m_pDevice[idx + 0 * incr];
                XX[thid+ 1*XXINCR] = alpha * x_dev.m_pDevice[idx + 1 * incr];
                XX[thid+ 2*XXINCR] = alpha * x_dev.m_pDevice[idx + 2 * incr];
                XX[thid+ 3*XXINCR] = alpha * x_dev.m_pDevice[idx + 3 * incr];
            }
            else if ((jj + 2 * XXINCR) < jjLimit ) {// See if we can read one plus 2 shifted values?
                XX[thid+ 0*XXINCR] = alpha * x_dev.m_pDevice[idx + 0 * incr];
                XX[thid+ 1*XXINCR] = alpha * x_dev.m_pDevice[idx + 1 * incr];
                XX[thid+ 2*XXINCR] = alpha * x_dev.m_pDevice[idx + 2 * incr];
            }
            else if ((jj + 1 * XXINCR) < jjLimit) {// See if we can read one plus 1 shifted values?
                XX[thid+ 0*XXINCR] = alpha * x_dev.m_pDevice[idx + 0 * incr];
                XX[thid+ 1*XXINCR] = alpha * x_dev.m_pDevice[idx + 1 * incr];
            }
            else if (jj < jjLimit) {// See if we can read one plus 0 shifted values?
                XX[thid+0*XXINCR] = alpha * x_dev.m_pDevice[idx + 0 * incr];
            }
#elif (X_ELEMS_PER_THREAD == 2)
            if (jj + 1 * XXINCR < jjLimit) {// See if we can read one plus 1 shifted values?
                XX[thid+ 0*XXINCR] = alpha * x_dev.m_pDevice[idx + 0 * incr];
                XX[thid+ 1*XXINCR] = alpha * x_dev.m_pDevice[idx + 1 * incr];
            }
            else if (jj < jjLimit) {// See if we can read one plus 0 shifted values?
                XX[thid+0*XXINCR] = alpha * x_dev.m_pDevice[idx + 0 * incr];
            }
#elif (X_ELEMS_PER_THREAD == 1)
            if (jj < jjLimit) {// See if we can read one plus 0 shifted values?
                XX[thid+0*XXINCR] = alpha * x_dev.m_pDevice[idx + 0 * incr];
            }
#else
#error Current code cannot handle X_ELEMS_PER_THREAD != 4
#endif

            // Syncronize again
            __syncthreads();

			   row = ii + thid; // Global index into A_dev. The row which is processed by this thread!
            // Accumulate the dot product (all BLOCK_DIM Threads participate!)
            if(row < A_dev.m_M && thid < BLOCK_DIM){ // If this row is active
               PREC * A_ptr = PtrElem_ColM(A_dev,row,j); //  IDXA(row,j); // Start into A_dev for this block
               jjLimit = jjLimit - j; // represents the length of the dot product!
               jj=0;

               // Do block dot product in segments of DOT_PROD_SEGMENT
               // Register blocking, first load all stuff
               PREC regA[DOT_PROD_SEGMENT]; PREC regx[DOT_PROD_SEGMENT];
               incr = A_dev.m_outerStrideBytes;
               while ( (jj + (DOT_PROD_SEGMENT-1)) < jjLimit) { // see if we can do one and DOT_PROD_SEGMENT-1 more values
                  // Dot product , 6er elements
                  #pragma unroll
                  for(int k = 0 ; k < DOT_PROD_SEGMENT ; k++){
                     regA[k] = *(A_ptr);
                     regx[k] = XX[jj + k];
                     A_ptr = PtrColOffset_ColM(A_ptr,1,incr);
                  }
                  #pragma unroll
                  for(int k = 0 ; k < DOT_PROD_SEGMENT ; k++){
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


         if( row < A_dev.m_M && thid < BLOCK_DIM){
            idx = IDXB(row);
            if (beta != 0.0) {
                dotp += beta * b_dev.m_pDevice[idx];
            }
            idx = IDXY(row);
           y_dev.m_pDevice[idx] = dotp;
         }

      } // Shift grid

   }




   /** Needs some syncronization overhead to make the above kernel in place!
   * Works also for not in place execution...
   * if( ii >= A_dev.m_M){
            break; _____> DEADLOCK
   */
   template<typename TCudaMatrix>
   __global__ void matrixVectorMultiply_inPlace_kernel(
                                                      TCudaMatrix y_dev,
                                                      int incr_y,
                                                      typename TCudaMatrix::PREC alpha,
                                                      TCudaMatrix A_dev,
                                                      TCudaMatrix x_dev,
                                                      int incr_x,
                                                      typename TCudaMatrix::PREC beta,
                                                      TCudaMatrix b_dev,
                                                      int incr_b,
                                                      GPUAtomicCounter global_sync){
      typedef typename TCudaMatrix::PREC PREC;
      __shared__ PREC XX[JINCR]; // Shared values for the x_dev;

      int thid = threadIdx.x;

      int i,j, ii, jj, row,  jjLimit, idx, incr;
      // i: first level index in row direction to the start of the current grid (grid level),
      // ii: second level index in row direction to the start of the current block (block level),

      PREC dotp; // Dot product which is accumulated by 1 thread.

      // Shift the whole cuda grid over the rows!

      for(i = 0 ; i < A_dev.m_M ; i += gridDim.x * BLOCK_DIM )
      {

         ii = i + blockIdx.x * BLOCK_DIM;   // Represents the row index to the current block start
         if( ii >= A_dev.m_M){
            //break;  // The blocks which lie outside the matrix can be rejected already here
         }
         // Set the dot product to zero!
         dotp = 0;

         // Shift the block with dim[ BLOCK_DIM , (THREADS_PER_BLOCK*X_ELEMS_PER_THREAD)]
         // horizontally over A_dev, each thread calculates the sub dot product of
         // blockDim.x*X_ELEMS_PER_THREAD elements.
         // j points to the beginning of the block and is shifted with JINCR = BLOCK_DIM
         for(j = 0 ; j < A_dev.m_N; j += JINCR){

            jj = j + thid;                            // Represents here the start index into x_dev for each thread!
            jjLimit = min (j + JINCR, A_dev.m_N);       // Represents here the maximal index for jj for this shifted block!

            // Syncronize before loading the (THREADS_PER_BLOCK*X_ELEMS_PER_THREAD) shared values of x_dev
            // Each thread loads X_ELEMS_PER_THREAD elements.
            __syncthreads();

            // Load loop (all THREADS_PER_BLOCK participate!)
            // Each of this BLOCK_DIM threads goes into another if statement if the
            incr = incr_x * XXINCR;
            idx = IDXX(jj);
 #if (X_ELEMS_PER_THREAD == 4)
            if ((jj + 3 * XXINCR)< jjLimit ) { // See if we can read one plus  3 shifted values?
                XX[thid+ 0*XXINCR] = alpha * x_dev.m_pDevice[idx + 0 * incr];
                XX[thid+ 1*XXINCR] = alpha * x_dev.m_pDevice[idx + 1 * incr];
                XX[thid+ 2*XXINCR] = alpha * x_dev.m_pDevice[idx + 2 * incr];
                XX[thid+ 3*XXINCR] = alpha * x_dev.m_pDevice[idx + 3 * incr];
            }
            else if ((jj + 2 * XXINCR) < jjLimit ) {// See if we can read one plus 2 shifted values?
                XX[thid+ 0*XXINCR] = alpha * x_dev.m_pDevice[idx + 0 * incr];
                XX[thid+ 1*XXINCR] = alpha * x_dev.m_pDevice[idx + 1 * incr];
                XX[thid+ 2*XXINCR] = alpha * x_dev.m_pDevice[idx + 2 * incr];
            }
            else if ((jj + 1 * XXINCR) < jjLimit) {// See if we can read one plus 1 shifted values?
                XX[thid+ 0*XXINCR] = alpha * x_dev.m_pDevice[idx + 0 * incr];
                XX[thid+ 1*XXINCR] = alpha * x_dev.m_pDevice[idx + 1 * incr];
            }
            else if (jj < jjLimit) {// See if we can read one plus 0 shifted values?
                XX[thid+0*XXINCR] = alpha * x_dev.m_pDevice[idx + 0 * incr];
            }
#elif (X_ELEMS_PER_THREAD == 2)
            if (jj + 1 * XXINCR < jjLimit) {// See if we can read one plus 1 shifted values?
                XX[thid+ 0*XXINCR] = alpha * x_dev.m_pDevice[idx + 0 * incr];
                XX[thid+ 1*XXINCR] = alpha * x_dev.m_pDevice[idx + 1 * incr];
            }
            else if (jj < jjLimit) {// See if we can read one plus 0 shifted values?
                XX[thid+0*XXINCR] = alpha * x_dev.m_pDevice[idx + 0 * incr];
            }
#elif (X_ELEMS_PER_THREAD == 1)
            if (jj < jjLimit) {// See if we can read one plus 0 shifted values?
                XX[thid+0*XXINCR] = alpha * x_dev.m_pDevice[idx + 0 * incr];
            }
#else
#error Current code cannot handle X_ELEMS_PER_THREAD != 4
#endif

            // Syncronize again
            __syncthreads();

			   row = ii + thid; // Global index into A_dev. The row which is processed by this thread!
            // Accumulate the dot product (all BLOCK_DIM Threads participate!)
            if(row < A_dev.m_M && thid < BLOCK_DIM){ // If this row is active
               PREC * A_ptr = PtrElem_ColM(A_dev,row,j); //  IDXA(row,j); // Start into A_dev for this block
               jjLimit = jjLimit - j; // represents the length of the dot product!
               jj=0;

               // Do block dot product in segments of DOT_PROD_SEGMENT
               // Register blocking, first load all stuff
               PREC regA[DOT_PROD_SEGMENT]; PREC regx[DOT_PROD_SEGMENT];
               incr = A_dev.m_outerStrideBytes;
               while ( (jj + (DOT_PROD_SEGMENT-1)) < jjLimit) { // see if we can do one and DOT_PROD_SEGMENT-1 more values
                  // Dot product , 6er elements
                  #pragma unroll
                  for(int k = 0 ; k < DOT_PROD_SEGMENT ; k++){
                     regA[k] = *(PtrColOffset_ColM(A_ptr,k,incr));
                     regx[k] = XX[jj + k];
                  }
                  #pragma unroll
                  for(int k = 0 ; k < DOT_PROD_SEGMENT ; k++){
                     dotp += regA[k] * regx[k] ;
                  }
                  // ==========================

                  jj     += DOT_PROD_SEGMENT;          // jump DOT_PROD_SEGMENT rows in x_dev
                  A_ptr = PtrColOffset_ColM(A_ptr, DOT_PROD_SEGMENT, incr);   // jump DOT_PROD_SEGMENT cols in A_dev
               }
               // if no more DOT_PROD_SEGMENT segments are available do the rest
               while (jj < jjLimit) {
                  dotp += (*A_ptr) * XX[jj + 0];
                  jj   += 1;           // jump 1 rows in XX
                  A_ptr = PtrColOffset_ColM(A_ptr, 1, incr);   // jump 1 col in A_dev
               }

            } // Accumulate dot product
         } // shift blocks horizontally to obtain overall dot product!
         __syncthreads();

         // Global thread block syncronization!
         syncronizeGlobal(global_sync, thid, blockIdx.x, gridDim.x);
         // ===================================

         if( row < A_dev.m_M && thid < BLOCK_DIM){
            idx = IDXB(row);
            if (beta != 0.0) {
                dotp += beta * b_dev.m_pDevice[idx];
            }
            idx = IDXY(row);
           y_dev.m_pDevice[idx] = dotp;
         }

         __syncthreads();
         // Global thread block syncronization!
         syncronizeGlobal(global_sync, thid, blockIdx.x, gridDim.x);
         // ===================================

      } // Shift grid, blocks which completely over lap exit here

   }




}
#endif
