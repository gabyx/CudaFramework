#include "CudaFramework/Kernels/VectorAddGPU/VectorAddGPU.hpp"

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <time.h>

#include <boost/format.hpp>

#include "CudaFramework/CudaModern/CudaError.hpp"
#include "CudaFramework/General/Utilities.hpp"
#include "CudaFramework/CudaModern/CudaUtilities.hpp"

#include <cuda_runtime.h>

using namespace std;
using namespace Utilities;
using namespace utilCuda;


#define CHECK_GPU_RESULTS 1

void vectorAddGPU::performanceTestVectorAdd(std::ostream & data, std::ostream & log, int kernel = 0){

   typedef  double GPUPrec;

   // Open File
   data << "# Performance Test for Vector Addition with different Block Dim and Thread Dim." << endl;
   data << "GPUPrec : "<<"\t"<< typeid(GPUPrec).name() <<std::endl;

	// Set the function pointer
	void (*kernel_ptr)(GPUPrec *, GPUPrec * , GPUPrec * ,  int, const dim3 &, const dim3 & ) = NULL;
	switch(kernel){
		case 0:
			kernel_ptr = &vectorAddGPU::vectorAdd_kernelWrap;
         data << "Kernel used :"<<"\t"<< "vectorAddGPU::vectorAdd_kernelWrap" << endl;
			break;
		case 1:
			kernel_ptr = &vectorAddGPU::vectorAddShared_kernelWrap;
         data << "Kernel used :"<<"\t"<< "vectorAddGPU::vectorAddShared_kernelWrap" << endl;
			break;
	}



	// Fill all dimensions to check
	int blockDimMin = 16;
	int blockDimStep = 16;
	int blockDimMax = 1024;
	int gridDimMin = 16;
	int gridDimStep = 16;
	int gridDimMax = 1024;

	data << "blockDimMin : "<< "\t" << blockDimMin <<std::endl;
	data << "blockDimStep : "<< "\t" << blockDimStep <<std::endl;
	data << "blockDimMax : "<< "\t" << blockDimMax <<std::endl;
	data << "gridDimMin : "<< "\t" << gridDimMin <<std::endl;
	data << "gridDimStep : "<< "\t" << gridDimStep <<std::endl;
	data << "gridDimMax : "<< "\t" << gridDimMax <<std::endl;

	// Add Vector C = A + B
	int maxKernelLoops = 10;
	data << "nLoops : "<< "\t" << maxKernelLoops <<std::endl;
	// Allocate Memory for Vector A
	int MA = 1024*1024;
	int NA = 1;
	GPUPrec * A =(GPUPrec*) malloc(MA * NA * sizeof(GPUPrec));
	data << "Size A: "<< "\t"  << MA << "\t" << NA <<std::endl;

	// Allocate Memory for Vector B
	int MB = MA;
	int NB = 1;
	GPUPrec * B= (GPUPrec*) malloc(MB * NB * sizeof(GPUPrec));
	data << "Size B: "<< "\t"  << MB << "\t" << NB <<std::endl;

	// Allocate Memory for Vector C
	int MC = MA;
	int NC = 1;
	GPUPrec * C= (GPUPrec*) malloc(MC * NC * sizeof(GPUPrec));
	data << "Size C: "<< "\t"  << MC << "\t" << NC <<std::endl;

	// Number of Operations
	double nOps = 2.0*MA;
	data << "Num. Ops: "<< "\t"  << nOps << endl;

	// Fill Matrix with random values...
	fillRandom(A,MA*NA);
	fillRandom(B,MB*NB);
	log << "# Filled Vectors, with random values..."<<std::endl;
	// Cuda Event Create =======================================================
	cudaEvent_t startKernel,stopKernel;

	CHECK_CUDA(cudaEventCreate(&startKernel));
	CHECK_CUDA(cudaEventCreate(&stopKernel));


	// Allocate Memory on Device ===============================================
	GPUPrec *Adev, *Bdev, *Cdev;
	CHECK_CUDA(cudaMalloc((void**) &Adev, MA * NA * sizeof(GPUPrec)));
	CHECK_CUDA(cudaMalloc((void**) &Bdev, MB * NB * sizeof(GPUPrec)));
	CHECK_CUDA(cudaMalloc((void**) &Cdev, MC * NC * sizeof(GPUPrec)));

	// Copy host memory to device ==============================================
	CHECK_CUDA(cudaMemcpy(Adev, A, MA * NA * sizeof(GPUPrec), cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemcpy(Bdev, B, MB * NB * sizeof(GPUPrec), cudaMemcpyHostToDevice));

	data << "# GridDim X "<< "\t\t" << "GridDim Y" << "\t\t" << "ThreadDim X" << "\t\t" << "ThreadDim Y" << "\t\t" << "Time in [ms]" << "\t\t" << "GFlops/sec"<<std::endl;

	// Define appropriate Grid size and Blocksize ==============================
	for(int dimGrid = gridDimMin;		dimGrid <= gridDimMax;	dimGrid+=gridDimStep ){
		for(int dimBlock = blockDimMin; dimBlock  <= blockDimMax;	dimBlock+=blockDimStep){

			dim3 threads(dimBlock);
			dim3 blocks(dimGrid);

			// Launch kernel ===========================================================
			log << "Launch kernel, B:" << boost::format("%1$i,%2$i, T: %3$i,%4$i") % blocks.x % blocks.y % threads.x % threads.y << endl;
			CHECK_CUDA(cudaEventRecord(startKernel,0));
			for(int nloops = 1; nloops<=maxKernelLoops; nloops++){
				kernel_ptr(Cdev, Adev, Bdev, MA, threads, blocks);
			}

			CHECK_CUDA(cudaThreadSynchronize());
			CHECK_CUDA(cudaEventRecord(stopKernel,0));
			CHECK_CUDA(cudaEventSynchronize(stopKernel));
			log << "Kernel finished!" << endl;

			// Record Time =============================================================
			float elapsedKernelTime;
			CHECK_CUDA( cudaEventElapsedTime(&elapsedKernelTime,startKernel,stopKernel));
			double nMillisecs = (double)elapsedKernelTime / (double)maxKernelLoops;
			double nGFlops = (1e-9* nOps/(nMillisecs/1000.0));
			log << "GPU Kernel Time :"<< boost::format("%1$8.6f ms") % nMillisecs <<std::endl;
			log << "GPU Gigaflops : " << boost::format("%1$5.6f Gflops/s , Ops:  %2$f ") % nGFlops % nOps <<std::endl;

			// Save to file
			data << boost::format("%1$.9d\t\t%2$.9d\t\t%3$.9d\t\t%4$.9d\t\t%5$.9d\t\t%6$.9d") % blocks.x % blocks.y % threads.x % threads.y % nMillisecs % nGFlops << endl;
		}
	}


	// Copy result back ========================================================
	// CHECK_CUDA(cudaMemcpy(C, Cdev, MC * NC * sizeof(GPUPrec), cudaMemcpyDeviceToHost));

	// PRINT THE CHECK =======================================================
	//#if CHECK_GPU_RESULTS == 1
	//	printf("Check results...\n");
	//	GPUPrec * CRef = (GPUPrec*) malloc(MC * NC * sizeof(GPUPrec));
	//	matrixMultiply(CRef,A,B, MA,NA,MB,NB);

	//	/*for(int i=0; i<MC*NC && i < 300; i++){
	//		printf(" %i: \t %.9f \t %.9f \n",i, CRef[i], C[i]);
	//	}*/
	//
	//	if(compareArrays(CRef,C, MC*NC, 1.0e-4f ) == true){
	//		printf("Matrix Mult. Test passed!\n");
	//	}else{
	//		printf("Matrix Mult. Test NOT passed\n");
	//	}
	// free(CRef);
	//#endif


	// Free resources ==========================================================
	free(A);
	free(B);
	free(C);
	cudaFree(Adev);
	cudaFree(Bdev);
	cudaFree(Cdev);
	cudaEventDestroy(startKernel);
	cudaEventDestroy(stopKernel);
}
void vectorAddGPU::randomVectorAdd(int kernel = 0){

    typedef  double GPUPrec;

	// Set the function pointer
	void (*kernel_ptr)(GPUPrec *, GPUPrec * , GPUPrec * ,  int,  const dim3 &, const dim3 &) = NULL;
	switch(kernel){
		case 0:
			printf("Random Vector Add ======================================\n");
			kernel_ptr = &vectorAddGPU::vectorAdd_kernelWrap;
			break;
		case 1:
			printf("Random Vector Add Shared ======================================\n");
			kernel_ptr = &vectorAddGPU::vectorAddShared_kernelWrap;
			break;
	}


	// Vector Add C = A + B
	int maxKernelLoops = 10;

	// Allocate Memory for Matrix A
	int MA = 1024*1024;
	GPUPrec * A =(GPUPrec*) malloc(MA * sizeof(GPUPrec));

	// Allocate Memory for Matrix B
	int MB = MA;
	GPUPrec * B= (GPUPrec*) malloc(MB * sizeof(GPUPrec));


	// Allocate Memory for Matrix C
	int MC = MA;
	GPUPrec * C= (GPUPrec*) malloc(MC * sizeof(GPUPrec));

	// Fill Matrix with random values...
	fillRandom(A,MA);
	fillRandom(B,MB);

	// Cuda Event Create ==============================================================
	cudaEvent_t start, stop, startKernel, stopKernel;
	CHECK_CUDA(cudaEventCreate(&start));
	CHECK_CUDA(cudaEventCreate(&startKernel));
	CHECK_CUDA(cudaEventCreate(&stopKernel));
	CHECK_CUDA(cudaEventCreate(&stop));

	CHECK_CUDA(cudaEventRecord(start,0));

	// Allocate Memory on Device ======================================================
	GPUPrec *Adev, *Bdev, *Cdev;
	CHECK_CUDA(cudaMalloc((void**) &Adev, MA  * sizeof(GPUPrec)));
	CHECK_CUDA(cudaMalloc((void**) &Bdev, MB  * sizeof(GPUPrec)));
	CHECK_CUDA(cudaMalloc((void**) &Cdev, MC  * sizeof(GPUPrec)));

	// Copy host memory to device =====================================================
	CHECK_CUDA(cudaMemcpy(Adev, A, MA  * sizeof(GPUPrec), cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemcpy(Bdev, B, MB  * sizeof(GPUPrec), cudaMemcpyHostToDevice));

	// Define appropriate Grid size and Blocksize
	cudaDeviceProp prop;
	CHECK_CUDA(cudaGetDeviceProperties(&prop,0));
	int nMP = prop.multiProcessorCount;
	printf("Multiprocessor Count: %i \n",nMP);

    dim3 threads(256);
    dim3 blocks(64);

	// Launch kernel ==================================================================
   cout << "Launch kernel, B:" << boost::format("%1$i,%2$i, T: %3$i,%4$i") % blocks.x % blocks.y % threads.x % threads.y << endl;
	CHECK_CUDA(cudaEventRecord(startKernel,0));
	for(int nloops = 1;nloops<=maxKernelLoops;nloops++){
		kernel_ptr( Cdev, Adev, Bdev, MA, threads, blocks);
	}
	CHECK_CUDA(cudaEventRecord(stopKernel,0));
	CHECK_CUDA(cudaEventSynchronize(stopKernel));

	printf("Kernel started...\n");
	CHECK_CUDA(cudaThreadSynchronize());


	printf("Kernel finished!\n");

	// Copy result back ============================================================
	CHECK_CUDA(cudaMemcpy(C, Cdev, MC * sizeof(GPUPrec), cudaMemcpyDeviceToHost));

	// Record Time =================================================================
	CHECK_CUDA(cudaEventRecord(stop,0));
	CHECK_CUDA(cudaEventSynchronize(stop));

	float elapsedTime,elapsedKernelTime;
	CHECK_CUDA( cudaEventElapsedTime(&elapsedKernelTime,startKernel,stopKernel));
	CHECK_CUDA( cudaEventElapsedTime(&elapsedTime,start,stop));
	double nMillisecs = (double)elapsedKernelTime / (double)maxKernelLoops;
	double nOps = 2*MA;
	printf("GPU Kernel Time :  %8.6f ms \n",nMillisecs);
	printf("GPU Gigaflops :  %5.6f Gflops/s , Ops:  %f \n", 1e-9 * nOps/(nMillisecs/1000.0), nOps);
	printf("GPU Total Time (malloc,cpy,kernel,cpy): %3.6f ms \n",elapsedTime);


	// PRINT THE CHECK ==============================================================
	#if CHECK_GPU_RESULTS == 1
		printf("Check results...\n");
		GPUPrec * CRef = (GPUPrec*) malloc(MC * sizeof(GPUPrec));

		vectorAdd(CRef,A,B, MA );

		/*for(int i=0; i<MC; i++){
			printf(" %i: \t %.9f \t %.9f \n",i, CRef[i], C[i]);
		}*/

		if(compareArraysEach(CRef,C, MC, (GPUPrec)1.0e-4 ) == true){
			printf("Vector Add Test passed!\n");
		}else{
			printf("Vector Add Test NOT passed\n");
		}
		free(CRef);
	#endif


	// Free resources ==================================================================
	free(A);
	free(B);
	free(C);
	cudaFree(Adev);
	cudaFree(Bdev);
	cudaFree(Cdev);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaEventDestroy(startKernel);
	cudaEventDestroy(stopKernel);
}
