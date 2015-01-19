

#include <cuda_runtime.h>

#include <kernel.cuh>


#define ThreadsPerBlock 128
template<typename T>
__host__ void call_kernel_power(T *base, T *n, T *output, T elementCount) {


    int *dev_base, *dev_n, *dev_output;
    int gridX = (elementCount+ThreadsPerBlock-1)/ThreadsPerBlock;

   cudaMalloc( (void**)&dev_base, elementCount * sizeof(int) );
	cudaMalloc( (void**)&dev_n, elementCount * sizeof(int) );
	cudaMalloc( (void**)&dev_output, elementCount * sizeof(int) );

   cudaMemcpy( dev_base, base, elementCount * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy( dev_n, n, elementCount * sizeof(int), cudaMemcpyHostToDevice);

	power<T><<<gridX,ThreadsPerBlock>>>(dev_base, dev_n, dev_output, elementCount);

    cudaMemcpy( output, dev_output, elementCount * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree( dev_base );
	cudaFree( dev_n );
	cudaFree( dev_output );
}


// explicit instantiation
template __host__ void call_kernel_power(int *base, int *n, int *output, int elementCount);

