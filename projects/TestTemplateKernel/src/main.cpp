#include "main.hpp"
#include "CudaFramework/CudaModern/CudaError.hpp"

#define N   80000

int main() {

    printf("Power Cuda kernel test from C++\n");
    printf("Testing %d elements\n", N);

    int base[N], n[N], output[N];

    for(int i = 0; i < N; i++) {
	    base[i] = 2;
	    n[i] = i+1;
	    output[i] = 0;
	}

    CHECK_CUDA(cudaDeviceReset());
    CHECK_CUDA(cudaSetDevice(0));

    call_kernel_power(base, n, output, N);


    for(int i = 0; i < N && i < 15; i++) {

	    printf("%d^%d = %d\n", base[i], n[i], output[i]);

	}

	printf("Done\n");

    return 0;
}
