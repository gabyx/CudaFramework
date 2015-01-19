

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
