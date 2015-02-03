// ========================================================================================
//  CudaFramework
//  Copyright (C) 2014 by Gabriel Nützi <nuetzig (at) imes (d0t) mavt (d0t) ethz (d0t) ch>
//
//  This Source Code Form is subject to the terms of the GNU GPL 3.0 licence. 
//  If a copy of the GNU GPL 3.0 was not distributed with this
//  file, you can obtain one at http://opensource.org/licenses/GPL-3.0.
// ========================================================================================

#include "CudaFramework/Kernels/MatrixMultGPU/MatrixMultGPU.hpp"

#include <stdlib.h>
#include <stdio.h>
#include <iostream>

#include <Eigen/Dense>
#include <cuda_runtime.h>
#include <tinyformat/TinyFormatInclude.hpp>

#include "CudaFramework/CudaModern/CudaError.hpp"
#include "CudaFramework/General/Utilities.hpp"
#include "CudaFramework/CudaModern/CudaMatrixUtilities.hpp"



using namespace std;
using namespace Utilities;
using namespace utilCuda;

#define CHECK_GPU_RESULTS 1

void matrixMultGPU::randomMatrixMult(int kernel) {

    typedef  double GPUPrec;

    // Set the function pointer
    void (*kernel_ptr)(utilCuda::CudaMatrix<GPUPrec,CudaMatrixFlags::RowMajor>&,utilCuda::CudaMatrix<GPUPrec,CudaMatrixFlags::RowMajor>&,utilCuda::CudaMatrix<GPUPrec,CudaMatrixFlags::RowMajor>&) = NULL;

    cublasHandle_t handleCublas;

    switch(kernel) {
    case 2:
        cout<<"Random Matrix Mulitplication BLAS ==================================" << endl;
        cout<<"Kernel used: matrixMultGPU::matrixMultiplyCUBLAS "<<std::endl;
        CHECK_CUBLAS(cublasCreate(&handleCublas));
        break;
    case 3:
        cout<<"Random Matrix Mulitplication Shared 16x16 tiles in C ==============="<<std::endl;
        cout<<"Kernel used: matrixMultGPU::matrixMultiplySharedFixed_kernelWrap "<<std::endl;
        kernel_ptr = &matrixMultGPU::matrixMultiplySharedFixed_kernelWrap;
        break;
    case 4:
        cout<<"Random Matrix Mulitplication Shared 256x16 tiles in C =============="<<std::endl;
        cout<<"Kernel used: matrixMultGPU::matrixMultiplySharedFixedLargeRow_kernelWrap "<<std::endl;
        kernel_ptr = &matrixMultGPU::matrixMultiplySharedFixedLargeRow_kernelWrap;
        break;
    case 5:
        cout<<"Random Matrix Mulitplication Shared 16x256 tiles in C =============="<<std::endl;
        cout<<"Kernel used: matrixMultGPU::matrixMultiplySharedFixedLargeCol_kernelWrap "<<std::endl;
        kernel_ptr = &matrixMultGPU::matrixMultiplySharedFixedLargeCol_kernelWrap;
        break;
    case 6:
        cout<<"Random Matrix Mulitplication Shared 16x256 tiles in C (very unsafe!, optimized)=============="<<std::endl;
        cout<<"Kernel used: matrixMultGPU::matrixMultiplySharedFixedLargeColOptimized_kernelWrap "<<std::endl;
        kernel_ptr = &matrixMultGPU::matrixMultiplySharedFixedLargeColOptimized_kernelWrap;
        break;
    case 7:
        cout<<"Random Matrix Mulitplication Shared 16x256 tiles in C (from internet) =============="<<std::endl;
        cout<<"Kernel used: matrixMultGPU::matrixMultiplySharedFixedLargeBase_kernelWrap "<<std::endl;
        kernel_ptr = &matrixMultGPU::matrixMultiplySharedFixedLargeBase_kernelWrap;
        break;
    default:
        std::cerr << "No kernel specified!";
        break;
    }


    // Multiply Matrix C = A * B
    int maxKernelLoops = 10;
    // Allocate Memory for Matrix A
    int MA = 4096;
    int NA = MA;
    Eigen::Matrix<GPUPrec,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> A(MA,NA);

    // Allocate Memory for Matrix B

    int MB = MA;
    int NB = MA;
    Eigen::Matrix<GPUPrec,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> B(MB,NB);

    // Allocate Memory for Matrix C
    int MC = MA;
    int NC = NB;
    Eigen::Matrix<GPUPrec,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> C(MC,NC);
    //setMatrix(C,-1);

    cout << "Matrix size: ("<< MA<<"x"<<NA<<") * (" << MB<<"x"<<NB<<")" <<std::endl;

    // Fill Matrix wit random values...
    A.setRandom();
    B.setRandom();

    // Cuda Event Create =======================================================
    cudaEvent_t start, stop,startKernel,stopKernel;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&startKernel));
    CHECK_CUDA(cudaEventCreate(&stopKernel));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start,0));

    // Allocate Memory on Device ===============================================
    utilCuda::CudaMatrix<GPUPrec,CudaMatrixFlags::RowMajor> Adev, Bdev, Cdev;
    CHECK_CUDA(mallocMatrixDevice<false>(Adev, MA , NA));
    CHECK_CUDA(mallocMatrixDevice<false>( Bdev, MB , NB ));
    CHECK_CUDA(mallocMatrixDevice<false>(Cdev, MC , NC ));

    // Copy host memory to device ==============================================
    CHECK_CUDA(copyMatrixToDevice(Adev, A));
    CHECK_CUDA(copyMatrixToDevice(Bdev, B));
    //CHECK_CUDA(copyMatrixToDevice(Cdev, C));

    if(kernel == 2) {
        // use cublas
        //Warmup
        matrixMultGPU::matrixMultiplyCUBLAS(handleCublas, Cdev, Adev, Bdev);

        CHECK_CUDA(cudaEventRecord(startKernel,0));
        for(int nloops = 1; nloops<=maxKernelLoops; nloops++) {
            matrixMultGPU::matrixMultiplyCUBLAS(handleCublas, Cdev, Adev, Bdev);
            //CHECK_CUDA(cudaThreadSynchronize());
        }
        CHECK_CUDA(cudaEventRecord(stopKernel,0));
    } else {
        // use own kernels
        //Warmup
        kernel_ptr( Cdev, Adev, Bdev);

        CHECK_CUDA(cudaEventRecord(startKernel,0));
        for(int nloops = 1; nloops<=maxKernelLoops; nloops++) {
            kernel_ptr( Cdev, Adev, Bdev);
            //CHECK_CUDA(cudaThreadSynchronize());
        }
        CHECK_CUDA(cudaEventRecord(stopKernel,0));

    }

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaThreadSynchronize());
    CHECK_CUDA(cudaEventSynchronize(stopKernel));
    cout<<"Kernel finished!"<<std::endl;


    // Copy result back ========================================================
    CHECK_CUDA(copyMatrixToHost(C, Cdev));

    // Record Time =============================================================
    CHECK_CUDA(cudaEventRecord(stop,0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsedTime,elapsedKernelTime;
    CHECK_CUDA( cudaEventElapsedTime(&elapsedKernelTime,startKernel,stopKernel));
    CHECK_CUDA( cudaEventElapsedTime(&elapsedTime,start,stop));
    double nMillisecs = (double)elapsedKernelTime / (double)maxKernelLoops;
    double nOps = (2.0*MA*NA - MA) * NB;
#if FLOPS_WITH_ASSIGNMENTS == 1
    nOps += MC*NC;
#endif
    cout<< tinyformat::format("Kernel Time :  %8.6f ms \n",  nMillisecs);
    cout<< tinyformat::format("GPU Gigaflops :  %5.6f Gflops/s , Ops:  %f \n", (1e-9* nOps/(nMillisecs/1000.0)) , nOps);
    cout<< tinyformat::format("GPU Total Time (malloc,cpy,kernel,cpy): %3.6f ms \n", elapsedTime);


    // PRINT THE CHECK =======================================================
#if CHECK_GPU_RESULTS == 1
    cout<<"Check results..."<<std::endl;
    Eigen::Matrix<GPUPrec,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> CRef;
    CRef = A*B;

    /*for(int i=0; i<MC*NC && i < 100; i++){
    	cout<< boost::format(" %i: \t %.9f \t %.9f \n") % i % CRef.data[i] % C.data[i];
    }*/

    if(compareArraysEach(CRef.data(),C.data(),C.size(), (GPUPrec)1.0e-4 ) == true) {
        cout<<"Matrix Mult. Test passed!"<<std::endl;
    } else {
        cout<<"Matrix Mult. Test NOT passed"<<std::endl;
    }

#endif


    // Free resources ==========================================================
    freeMatrixDevice(Adev);
    freeMatrixDevice(Bdev);
    freeMatrixDevice(Cdev);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(startKernel);
    cudaEventDestroy(stopKernel);

    if(kernel == 2) {
        CHECK_CUBLAS(cublasDestroy(handleCublas));
    }
}

void matrixMultGPU::performanceTestMatrixMult(std::ostream & data, std::ostream & log, int kernel) {

    typedef  double GPUPrec;

    // Open File
    data << "# Performance Test for Matrix Multiplication with different Block Dim and Thread Dim." << endl;
    data << "GPUPrec : "<<"\t"<< typeid(GPUPrec).name() <<std::endl;
    // Set the function pointer
    void (*kernel_ptr)(CudaMatrix<GPUPrec,CudaMatrixFlags::RowMajor>&, CudaMatrix<GPUPrec,CudaMatrixFlags::RowMajor> &, CudaMatrix<GPUPrec,CudaMatrixFlags::RowMajor>&, const dim3 &threads, const dim3 &blocks) = NULL;
    switch(kernel) {
    case 0:
        kernel_ptr = &matrixMultGPU::matrixMultiply_kernelWrap;
        data << "Kernel used: "<<"\t"<<  "matrixMultGPU::matrixMultiply_kernelWrap" << endl;
        break;
    case 1:
        kernel_ptr = &matrixMultGPU::matrixMultiplyShared_kernelWrap;
        data << "Kernel used: "<<"\t"<<  "matrixMultGPU::matrixMultiplyShared_kernelWrap" << endl;
        break;
    }



    // Fill all dimensions to check
    int blockDimMin = 8;
    int blockDimStep = 4;
    int blockDimMax = 32;
    int gridDimMin = 16;
    int gridDimStep = 16;
    int gridDimMax = 128;

    data << "blockDimMin : "<< "\t" << blockDimMin <<std::endl;
    data << "blockDimStep : "<< "\t" << blockDimStep <<std::endl;
    data << "blockDimMax : "<< "\t" << blockDimMax <<std::endl;
    data << "gridDimMin : "<< "\t" << gridDimMin <<std::endl;
    data << "gridDimStep : "<< "\t" << gridDimStep <<std::endl;
    data << "gridDimMax : "<< "\t" << gridDimMax <<std::endl;

    // Multiply Matrix C = A * B
    int maxKernelLoops = 10;
    data << "nLoops : " << maxKernelLoops <<std::endl;
    // Allocate Memory for Matrix A

    int MA = 4096;
    int NA = 4096;
    Eigen::Matrix<GPUPrec,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> A(MA,NA);

    data << "Size A: "<< "\t"  << MA << "\t" << NA <<std::endl;

    // Allocate Memory for Matrix B
    int MB = 4096;
    int NB = 4096;
    Eigen::Matrix<GPUPrec,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> B(MB,NB);
    data << "Size B: "<< "\t"  << MB << "\t" << NB <<std::endl;

    // Allocate Memory for Matrix C
    int MC = MA;
    int NC = NB;
    Eigen::Matrix<GPUPrec,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> C(MC,NC);
    data << "Size C: "<< "\t"  << MC << "\t" << NC <<std::endl;

    // Number of Operations
    double nOps = (2.0*MA*NA - MA) * NB;
#if FLOPS_WITH_ASSIGNMENTS == 1
    nOps += MC*NC;
#endif
    data << "Num. Ops: "<< "\t"  << nOps << endl;

    // Fill Matrix wit random values...
    A.setRandom();
    B.setRandom();

    log << "# Filled Matrices, with random values..."<<std::endl;
    // Cuda Event Create =======================================================
    cudaEvent_t startKernel,stopKernel;

    CHECK_CUDA(cudaEventCreate(&startKernel));
    CHECK_CUDA(cudaEventCreate(&stopKernel));


    // Allocate Memory on Device ===============================================
    CudaMatrix<GPUPrec,CudaMatrixFlags::RowMajor> Adev, Bdev, Cdev;
    CHECK_CUDA(mallocMatrixDevice<false>(Adev, MA , NA));
    CHECK_CUDA(mallocMatrixDevice<false>(Bdev, MB , NB));
    CHECK_CUDA(mallocMatrixDevice<false>(Cdev, MC , NC));

    // Copy host memory to device ==============================================
    CHECK_CUDA(copyMatrixToDevice(Adev, A));
    CHECK_CUDA(copyMatrixToDevice(Bdev, B));


    data << "# GridDim X "<< "\t\t" << "GridDim Y" << "\t\t" << "ThreadDim X" << "\t\t" << "ThreadDim Y" << "\t\t" << "Time in [ms]" << "\t\t" << "GFlops/sec"<<std::endl;

    // Define appropriate Grid size and Blocksize ==============================
    for(int dimGrid = gridDimMin;		dimGrid <= gridDimMax;	dimGrid+=gridDimStep ) {
        for(int dimBlock = blockDimMin; dimBlock  <= blockDimMax;	dimBlock+=blockDimStep) {

            dim3 threads(dimBlock,dimBlock);
            dim3 blocks(dimGrid,dimGrid);

            // Launch kernel ===========================================================
            log << "Launch kernel, B:" << tinyformat::format("%i,%i, T: %i,%i", blocks.x , blocks.y , threads.x , threads.y) << endl;
            //Warmup
            kernel_ptr( Cdev, Adev, Bdev,threads,blocks);

            CHECK_CUDA(cudaEventRecord(startKernel,0));
            for(int nloops = 1; nloops<=maxKernelLoops; nloops++) {
                kernel_ptr( Cdev, Adev, Bdev,threads,blocks);
                //CHECK_CUDA(cudaThreadSynchronize());
            }
            CHECK_CUDA(cudaEventRecord(stopKernel,0));
            CHECK_CUDA(cudaThreadSynchronize());

            CHECK_CUDA(cudaEventSynchronize(stopKernel));
            log << "Kernel finished!" << endl;

            // Record Time =============================================================
            float elapsedKernelTime;
            CHECK_CUDA( cudaEventElapsedTime(&elapsedKernelTime,startKernel,stopKernel));
            double nMillisecs = (double)elapsedKernelTime / (double)maxKernelLoops;
            double nGFlops = (1e-9* nOps/(nMillisecs/1000.0));
            log << "GPU Kernel Time :"<< tinyformat::format("%8.6f ms", nMillisecs) <<std::endl;
            log << "GPU Gigaflops : " << tinyformat::format("%5.6f Gflops/s , Ops:  %f ",  nGFlops , nOps) <<std::endl;

            // Save to file
            data << tinyformat::format("%.9d\t\t%.9d\t\t%.9d\t\t%.9d\t\t%.9d\t\t%.9d" , blocks.x , blocks.y , threads.x , threads.y , nMillisecs,  nGFlops) << endl;
        }
    }


    // Copy result back ========================================================
    //CHECK_CUDA(copyMatrixToHost(C, Cdev));

    // PRINT THE CHECK =======================================================
    /*#if CHECK_GPU_RESULTS == 1
    	printf("Check results...\n");
    	Matrix<GPUPrec> CRef;
    	mallocMatrixHost(CRef,MC,NC);
    	multiplyMatrix(CRef,A,B);

    	for(int i=0; i<MC*NC && i < 300; i++){
    		printf(" %i: \t %.9f \t %.9f \n",i, CRef.data[i], C.data[i]);
    	}

    	if(compareMatrix(CRef,C, 1.0e-4f ) == true){
    		printf("Matrix Mult. Test passed!\n");
    	}else{
    		printf("Matrix Mult. Test NOT passed\n");
    	}
    	freeMatrix(CRef);
    #endif*/

    // Free resources ==========================================================

    freeMatrixDevice(Adev);
    freeMatrixDevice(Bdev);
    freeMatrixDevice(Cdev);
    cudaEventDestroy(startKernel);
    cudaEventDestroy(stopKernel);

}


