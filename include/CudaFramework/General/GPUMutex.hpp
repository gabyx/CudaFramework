// ========================================================================================
//  CudaFramework
//  Copyright (C) 2014 by Gabriel Nützi <nuetzig (at) imes (d0t) mavt (d0t) ethz (d0t) ch>
//
//  This Source Code Form is subject to the terms of the GNU GPL 3.0 licence.
//  If a copy of the GNU GPL 3.0 was not distributed with this
//  file, you can obtain one at http://opensource.org/licenses/GPL-3.0.
// ========================================================================================

#ifndef CudaFramework_General_GPUMutex_hpp
#define CudaFramework_General_GPUMutex_hpp

#include <cuda_runtime.h>


struct GPUMutex{
   int *mutex;
   __host__ GPUMutex( void ) {}

   __host__ void  init() {
      cudaMalloc( (void**)&mutex,sizeof(int) );
      cudaMemset( mutex, 0, sizeof(int) );
   }

   __host__ void free() {
      cudaFree( mutex );
   }

   __device__ void lock( void ) {
      while( atomicCAS( mutex, 0, 1 ) != 0 );
   }

   __device__ void unlock( void ) {
      atomicExch( mutex, 0 );
   }
};



struct GPUAtomicCounter {
    int *counter;

#if defined(__cplusplus)
    __host__ GPUAtomicCounter( void ) {}

    __host__ void init() {
        cudaMalloc( (void**)&counter,sizeof(int) );
        cudaMemset( counter, 0, sizeof(int) );
    }

    // Ein Deconstructor gibt irgendwie fehler??
    __host__ void free(){
        cudaFree( counter );
    }
#endif

};




#endif
