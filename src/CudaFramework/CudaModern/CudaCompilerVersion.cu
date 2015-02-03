// ========================================================================================
//  CudaFramework
//  Copyright (C) 2014 by Gabriel Nützi <nuetzig (at) imes (d0t) mavt (d0t) ethz (d0t) ch>
//
//  This Source Code Form is subject to the terms of the GNU GPL 3.0 licence. 
//  If a copy of the GNU GPL 3.0 was not distributed with this
//  file, you can obtain one at http://opensource.org/licenses/GPL-3.0.
// ========================================================================================


#include <cuda.h>
#include <cuda_runtime.h>

namespace utilCuda{

    __global__ void kernelVersionShim(){};

    __host__ cudaError_t getCudaCompilerVersion(unsigned int ordinal, cudaFuncAttributes & attr){
        cudaSetDevice(ordinal);
        void (*p1)(void) = kernelVersionShim;
        const char * p2 = (char*)p1;
        return cudaFuncGetAttributes(&attr, p2);
    }

};
