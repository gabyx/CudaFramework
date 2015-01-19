// ========================================================================================
//  CudaFramework
//  Copyright (C) 2014 by Gabriel Nützi <nuetzig (at) imes (d0t) mavt (d0t) ethz (d0t) ch>
//
//  This Source Code Form is subject to the terms of the GNU GPL 3.0 licence.
//  If a copy of the GNU GPL 3.0 was not distributed with this
//  file, you can obtain one at http://opensource.org/licenses/GPL-3.0.
// ========================================================================================

#ifndef CudaFramework_Kernels_VectorAddGPU_KernelsVectorAdd_cuh
#define CudaFramework_Kernels_VectorAddGPU_KernelsVectorAdd_cuh


namespace vectorAddKernels{


   template<typename PREC>
   __global__ void vectorAdd_kernel( PREC *C, PREC * A, PREC * B,  int N){

      // GLOBAL MEMORY VECTOR ADDITION

      // Calculate  indexes
      int index_x = threadIdx.x + blockIdx.x * blockDim.x;
      int stride_x = gridDim.x * blockDim.x;

      while(index_x < N){
         // Do vector addition, each thread does one addition
         C[index_x] = A[index_x] + B[index_x];
         // =========================================================
         index_x += stride_x;
      }

   }

   extern __shared__  char shared_arrayDynAlloc[];
   template<typename PREC>
   __global__ void vectorAddShared_kernel( PREC *C, PREC * A, PREC * B,  int N){

      // SHARED MEMORY VECTOR ADDITION


      // Grid size
      int gDimX = gridDim.x;

      // Block size need to be (?,1) !!! checked in kernel wrapper
      int bDim = blockDim.x;

      // Block row and column
      int bx = blockIdx.x;

      // Thread index
      int tx = threadIdx.x;


      // Number of Blocks needed to fully fill C
      int maxBlocksCX = (N + bDim-1) / bDim;

      dim3 currentCBlockIdx(bx);

      // Shifting Blocks, only done beacuse it may happen that the grid is two small for the vector!
      while( currentCBlockIdx.x < maxBlocksCX ){

         // This vectors are shared in each block [bDimX,bDimY]! =======================================
         // Dynamic Size
         int size = (bDim)* sizeof(PREC);
         PREC* Asub_sh = (PREC *) shared_arrayDynAlloc;
         PREC* Bsub_sh = (PREC *) &shared_arrayDynAlloc[size];

         // Fixed Size TEST (blockDim = 16!!!)
         //__shared__ GPUPrec Asub_sh[16];
         //__shared__ GPUPrec Bsub_sh[16];
         // =============================================================================================

         int rowC = currentCBlockIdx.x*bDim + tx;

         // Load block sub vectors A and B into shared memory, each thread does one element.
         if(rowC < N){
            Asub_sh[tx] = A[rowC];
            Bsub_sh[tx] = B[rowC];
         }
         __syncthreads();

         // Add temporal value back to the C Vector, do this safe, because the currentCBlock may be out of the reach in C...
         if(rowC < N){
            C[rowC] = Asub_sh[tx] + Bsub_sh[tx];
         }

         // Sync threads befor one thread loads a new vector!
         __syncthreads();


         //=========================================================================================================================

         // Jump to next block column
         currentCBlockIdx.x += gDimX;
      }
   }


}

#endif
