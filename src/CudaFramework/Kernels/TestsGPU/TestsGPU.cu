#include <stdio.h>

#include "CudaFramework/Kernels/TestsGPU/KernelsTests.cuh"

#include "CudaFramework/General/AssertionDebugC.hpp"

#include "CudaFramework/General/GPUMutex.hpp"



namespace testsGPU{

   template<typename PREC>
   __host__ void  branchTest_kernelWrap(PREC * a){
      dim3 threads(33);
	   dim3 blocks(1);

      GPUAtomicCounter c;
      c.init();
 

      testsKernels::branchTest_kernel<<<blocks,threads>>>(a,c);
      //testsKernels::branchTest_kernel2<<<blocks,threads>>>(a,c.counter);

      //testsKernels::registerCheck<<<blocks,threads>>>(a);
      //CHECK_CUDA(cudaFree( c.counter ));
      c.free();
   }

   template<typename PREC>
   __host__ void strangeCudaBehaviour_wrap(PREC* output, PREC * mu, PREC * d, PREC * t, PREC * input){
      testsKernels::strangeCudaBehaviour<<<1,92>>>(output,mu,d,t,input);
   }


   template void  branchTest_kernelWrap(float * a);
   template void  branchTest_kernelWrap(double * a);


   #define PREC float
   template void strangeCudaBehaviour_wrap(PREC* output, PREC * mu, PREC * d, PREC * t, PREC * input);

}





