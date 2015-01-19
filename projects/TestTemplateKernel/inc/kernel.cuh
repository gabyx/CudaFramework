#ifndef kernel_cuh
#define kernel_cuh


__device__ int devicePower(int base, int n) {
    
    int p = 1;
    
    for (int i = 1; i <= n; ++i) {
        p = p * base;
    }
        
    return p;
}

template<typename T>
__global__ void power( int *base, int *n, int *output, int elementCount ) {

    T tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid < elementCount) {
        output[tid] = devicePower(base[tid], n[tid]);
    }

}

#endif