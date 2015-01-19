
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
