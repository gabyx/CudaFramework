// ========================================================================================
//  CudaFramework
//  Copyright (C) 2014 by Gabriel Nützi <nuetzig (at) imes (d0t) mavt (d0t) ethz (d0t) ch>
//
//  This Source Code Form is subject to the terms of the GNU GPL 3.0 licence.
//  If a copy of the GNU GPL 3.0 was not distributed with this
//  file, you can obtain one at http://opensource.org/licenses/GPL-3.0.
// ========================================================================================

#ifndef CudaFramework_Kernels_TestsGPU_KernelsTests_cuh
#define CudaFramework_Kernels_TestsGPU_KernelsTests_cuh


#include "CudaFramework/General/GPUMutex.hpp"

namespace testsKernels{

   template<typename PREC>
   __global__ void branchTest_kernel( PREC * a, GPUAtomicCounter c){

      int tx = threadIdx.x;

      // Test Case 1
      // 1 block, 2 threads
      //if(tx==1){
      //   a[1] = a[0] + 1;
      //}else if(tx==0){
      //   a[0] = a[1] + 1;
      //} // gives (1,2)

         // Test Case 1
      // 1 block, 33 threads
      if(tx==1){
         a[1] = a[0] + 1;
      }else if(tx==32){
         a[0] = a[1] + 1;
      } // gives (1,1)


      // Test Case 2
      // 1 block, 2 threads
      //if(tx==0){
      //   //while( atomicCAS(c.counter, 1,2) != 1){};
      //   a[1] = a[0] + 1;
      //}else{
      //   a[0] = a[1] + 1;
      //   //atomicAdd(c.counter,1);
      //} // gives (1,1)

      // Test Case 3
      // 1 block, 2 threads
      //if(tx==0){
      //   while( atomicCAS(c.counter, 1,2) != 1){};
      //   a[1] = a[0] + 1;
      //}else{
      //   a[0] = a[1] + 1;
      //   atomicAdd(c.counter,1);
      //} // gives (1,2)

      // Test Case 4
      // 1 block, 2 threads
      /*if(tx==0){
      a[0] = 10;
      atomicAdd(c.counter,1);
      }else if(tx==1){
      while( atomicCAS(c.counter, 1,1) != 1){};
      a[1] = a[0] + 5;
      } */
      // hangs completety because the two threads are in one warp which causes a hang, because interwarp scheduling executes
      // first else if and then the spin lock, which will never release and which will never make thread 0 eb able to atomicAdd! Read thread divergence in glossary!

      // Test Case 5
      // 1 block, 33 threads
      //if(tx==32){
      //   a[0] = 10;
      //   atomicAdd(c.counter,1);
      //}else if(tx==0){ // 31 no more works!
      //   while( atomicCAS(c.counter, 1,1) != 1){};
      //   a[1] = a[0] + 5;
      //}
      // This works because threads are not in the same warp, and thread 0 and thread 33 can both proceed!
   }

    template<typename PREC>
   __global__ void branchTest_kernel2( PREC * a,  int * b){

      int tx = threadIdx.x;
       *a = *b;
   }
   __global__ void registerCheck(int * b){
      int a = 3;
      b[2] = a;
   }

   template<typename PREC>
  __global__ void strangeCudaBehaviour_(PREC* output, PREC * mu, PREC * d, PREC * t, PREC * input){

      __shared__ PREC xx[3];
    //  __shared__ PREC tt[3];


      int thid = threadIdx.x;

      if(thid < 3){
            xx[thid] = input[thid];
      //      tt[thid] = t[thid];
      }


      // DO Something with the first thread!
      if( thid == 0 ){ // select thread on the diagonal ...
         /*
               PREC x_new_n = (d[thid] + tt[thid]);

               //Prox Normal!
               if(x_new_n <= 0.0){
                  x_new_n = 0.0;
               }

               xx[thid] = x_new_n;
               tt[thid] = 0.0;*/
         xx[0] =  4;
      }
      __syncthreads();

      /*
      // ALL THREADS ADD SOME NEW  values to tt
      if(thid < 3){
       tt[thid] += 3 * xx[0];
      }
      __syncthreads();
      */


      // Do something with the second thread!
      //PREC * mu_dev_ptr = &mu[0];
      if( thid == 1){

               // Prox tangential
            //   PREC radius = /*(*(xx+thid-1))*/ xx[thid-1] * mu[0]; /// TAKE THE FIRST VALUE!
             /*  PREC lambda_T1 =  d[thid] + tt[thid];
               PREC lambda_T2 =  d[thid+1] + tt[thid+1];


               PREC absvalue = sqrt(lambda_T1*lambda_T1 + lambda_T2*lambda_T2);

               if(absvalue >= radius){
                     lambda_T1   =  (lambda_T1  * radius ) / absvalue;
                     lambda_T2   =  (lambda_T2  * radius ) / absvalue;
               }
               */
               //Write the two values back!
               xx[thid] =  xx[thid-1] * mu[0];
              /* tt[thid] = 0.0;
               xx[thid+1] = lambda_T2;
               tt[thid+1] = 0.0;       */
     }
    __syncthreads();


    if(thid < 3){
       output[thid] = xx[thid];
     //  t[thid] = tt[thid];
    }
  }

    __global__ void strangeCudaBehaviour__(float* output, float * mu, float * d, float * t, float * input){

      __shared__ float xx[3];

       xx[threadIdx.x] = threadIdx.x + 5;

      __syncthreads();

       xx[threadIdx.x] = xx[threadIdx.x] * mu[0];  // iF WE TAKE MU THEN -> ADDRESSING ABOVE IS WRONG THREADiDX.X + 1 ....

       output[threadIdx.x] = xx[threadIdx.x];
  }

  __global__ void strangeCudaBehaviour(float* output, float * mu, float * d, float * t, float * input){

      __shared__ float xx[3];

      if(threadIdx.x < 3){
         xx[threadIdx.x] = 6 + threadIdx.x;
      }
      __syncthreads();

      if( threadIdx.x == 1){
         xx[threadIdx.x] = xx[threadIdx.x - 1];
         xx[threadIdx.x] *= mu[0];
      }

      __syncthreads();


      if(threadIdx.x < 3){
         output[threadIdx.x] = xx[threadIdx.x];
      }
  }

}







#endif
