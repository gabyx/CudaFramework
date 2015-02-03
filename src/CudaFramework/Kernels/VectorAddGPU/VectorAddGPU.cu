// ========================================================================================
//  CudaFramework
//  Copyright (C) 2014 by Gabriel Nützi <nuetzig (at) imes (d0t) mavt (d0t) ethz (d0t) ch>
//
//  This Source Code Form is subject to the terms of the GNU GPL 3.0 licence. 
//  If a copy of the GNU GPL 3.0 was not distributed with this
//  file, you can obtain one at http://opensource.org/licenses/GPL-3.0.
// ========================================================================================



#include "CudaFramework/Kernels/VectorAddGPU/KernelsVectorAdd.cuh"

#include <cuda_runtime.h>


namespace vectorAddGPU{

   using namespace vectorAddKernels;

   template<typename PREC>
   __host__ void vectorAdd_kernelWrap( PREC *Cdev, PREC * Adev, PREC * Bdev,  int NB,  const dim3 &threads, const dim3 &blocks){
	   vectorAdd_kernel<<< blocks, threads >>>( Cdev, Adev, Bdev, NB);
   }

    template<typename PREC>
   __host__ void vectorAddShared_kernelWrap( PREC *Cdev, PREC * Adev, PREC * Bdev,  int NB,  const dim3 &threads, const dim3 &blocks){
	   // Calcualte the amount of shared memory for A_sh and B_sh
	   size_t size_Ash_and_Bsh = 2 * (threads.x) * sizeof(PREC);
	   vectorAddShared_kernel<<< blocks, threads , size_Ash_and_Bsh >>>( Cdev, Adev, Bdev, NB);
   }


   template<typename PREC,  int nAddsPerThread>
   __host__ void vectorAddShared_kernelWrap( PREC *Cdev, PREC * Adev, PREC * Bdev,  int NB,  const dim3 &threads, const dim3 &blocks){
	   // Calcualte the amount of shared memory for A_sh and B_sh
	   size_t size_Ash_and_Bsh = 2 * (threads.x) * sizeof(PREC);
	   vectorAddShared_kernel<<< blocks, threads , size_Ash_and_Bsh >>>( Cdev, Adev, Bdev, NB);
   }

  

   // Explicit instantiate the types which are need in C++, other wise the code is not available for linking
   template __host__ void vectorAdd_kernelWrap( float *Cdev, float * Adev, float * Bdev,  int NB,  const dim3 &threads, const dim3 &blocks);
   template __host__ void vectorAddShared_kernelWrap( float *Cdev, float * Adev, float * Bdev,  int NB,  const dim3 &threads, const dim3 &blocks);
   
   template __host__ void vectorAdd_kernelWrap( double *Cdev, double * Adev, double * Bdev,  int NB,  const dim3 &threads, const dim3 &blocks);
   template __host__ void vectorAddShared_kernelWrap( double *Cdev, double * Adev, double * Bdev,  int NB,  const dim3 &threads, const dim3 &blocks);
}
