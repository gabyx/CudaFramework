
#include <stdio.h>
#include "TypenameComparision.hpp"
#include "CudaFramework/CudaModern/CudaError.hpp"

static char *sSDKsample = "CmakeTestSimple";


////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char** argv);
__global__ void test_kernel(float * c, float * a, float * b);
////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
	printf("[ %s ]\n", sSDKsample);


	runTest(argc, argv);
	system("pause");
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void runTest(int argc, char** argv)
{

	// =========================== WITHOUT CUUTIL

	cudaSetDevice(0);

	int devID;
	cudaDeviceProp props;

	// get number of SMs on this GPU
	cudaGetDevice(&devID);
	(cudaGetDeviceProperties(&props, devID));

	printf("Device %d: \"%s\" with Compute %d.%d capability\n", devID, props.name, props.major, props.minor);


	// allocate device memory
	float* d_A;
	float* A = (float*) malloc(sizeof(float));
	*A = 20;
   printf("A is: %f",*A);
	CHECK_CUDA(cudaMalloc((void**) &d_A, sizeof(float)));
	float* d_B;
	float* B = (float*) malloc(sizeof(float));
	*B = 10;
   printf("A is: %f",*B);
	CHECK_CUDA(cudaMalloc((void**) &d_B, sizeof(float)));

	// copy host memory to device
	CHECK_CUDA(cudaMemcpy(d_A, A, sizeof(float),cudaMemcpyHostToDevice) );
	CHECK_CUDA(cudaMemcpy(d_B, B, sizeof(float),cudaMemcpyHostToDevice) );

	// allocate device memory for result
	float* d_C;
	CHECK_CUDA(cudaMalloc((void**) &d_C, sizeof(float)));

	// allocate host memory for the result
	float* C = (float*) malloc(sizeof(float));
   *C = 0;

	test_kernel<<< 1, 1 >>>(d_C, d_A, d_B);
   cudaThreadSynchronize();
	CHECK_CUDA(cudaMemcpy(C, d_C, sizeof(float), cudaMemcpyDeviceToHost) );

	printf("Result is: %f",*C);

	// check if kernel execution generated and error
	//cutilCheckMsg("Kernel execution failed");


	// clean up memory
	free(A);
	free(B);
	free(C);
	(cudaFree(d_A));
	(cudaFree(d_B));
	(cudaFree(d_C));

	cudaThreadExit();


}


__global__
void test_kernel(float * c, float * a, float * b){
		*c = *a + *b;
}
