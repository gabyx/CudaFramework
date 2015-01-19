
// includes, system
#include <iostream>
#include <stdio.h>
#include <time.h>

////////////////////////////////////////////////////////////////////////////////
// Cuda extern C includes
#include "UtilitiesCuda.hpp"
#include "MatrixCuda.hpp"
#include "UtilitiesMatrixCuda.hpp"
#include "HandleError.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <Eigen/Dense>

#include <boost/format.hpp>
#include <platformstl/performance/performance_counter.hpp>


using namespace std;
using namespace utilCuda;
using namespace boost;

typedef double GPUPrec;

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
// Colmajor, normal
void sgemvNormal(int MG, int NG){
   // sgemv test

   srand ( time(NULL) );
   cout << " sGemv Test Normal ======================================="<<endl;


   cublasHandle_t cublasHandle; 
   CHECK_CUBLAS(cublasCreate(&cublasHandle));

   // Generate Matrices1024
   Eigen::Matrix<GPUPrec,Eigen::Dynamic, Eigen::Dynamic> G(MG,NG);
   G.setRandom();


   Eigen::Matrix<GPUPrec,Eigen::Dynamic, 1> x_old(NG);
   Eigen::Matrix<GPUPrec,Eigen::Dynamic, 1> y_newCPU(MG);
   Eigen::Matrix<GPUPrec,Eigen::Dynamic, 1> y_newGPU(MG);
   x_old.setRandom();



   // GPU GAUSS SEIDEL ====================================================================================
   // Load to GPU
   Matrix<GPUPrec> G_dev, x_old_dev, y_new_dev;




   // Cuda Event Create ==============================================================
   cudaEvent_t start, stop, startKernel, stopKernel, startCopy,stopCopy;
   CHECK_CUDA(cudaEventCreate(&start));
   CHECK_CUDA(cudaEventCreate(&stop));
   CHECK_CUDA(cudaEventCreate(&startKernel));
   CHECK_CUDA(cudaEventCreate(&stopKernel));
   CHECK_CUDA(cudaEventCreate(&startCopy));
   CHECK_CUDA(cudaEventCreate(&stopCopy));

   CHECK_CUDA(cudaEventRecord(start,0));


   //Copy Data
   CHECK_CUDA(cudaEventRecord(startCopy,0));
   mallocAndCopyMatrixToDevice(G_dev, G);
   mallocAndCopyMatrixToDevice(x_old_dev,x_old);
   mallocMatrixDevice(y_new_dev,MG,1,false);


   CHECK_CUDA(cudaEventRecord(stopCopy,0));
   CHECK_CUDA(cudaEventSynchronize(stopCopy));

   float elapsedTimeCopy,elapsedKernelTime;
   CHECK_CUDA( cudaEventElapsedTime(&elapsedTimeCopy,startCopy,stopCopy));
   cout << "Copy time: " << elapsedTimeCopy <<" ms" << endl;

   GPUPrec a = 1.0;
   GPUPrec b = 0;

   int maxKernelLoops = 10;
   CHECK_CUDA(cudaEventRecord(startKernel,0));
   for(int i = 0; i<maxKernelLoops; i++){
      CHECK_CUBLAS(cublasDgemv(
         cublasHandle, 
         CUBLAS_OP_N ,
         G_dev.M, 
         G_dev.N,  
         &a,
         G_dev.data,
         G_dev.outerStride,
         x_old_dev.data,
         1,
         &b,
         y_new_dev.data,
         1)
         );
   }

   CHECK_CUDA(cudaEventRecord(stopKernel,0));
   CHECK_CUDA(cudaEventSynchronize(stopKernel));


   CHECK_CUDA(cudaThreadSynchronize());
   printf("Iterations finished!\n");
   CHECK_CUDA( cudaEventElapsedTime(&elapsedKernelTime,startKernel,stopKernel));
   cout << "GPU Iteration time: " << elapsedKernelTime <<" ms" << endl;
   double nMillisecs = (double)elapsedKernelTime / (double)maxKernelLoops;
   double nOps = 2.0*NG *(MG*1);
   cout<< format("Kernel Time :  %1$8.6f ms \n") % nMillisecs;
   cout<< format("GPU Gigaflops :  %1$5.6f Gflops/s , Ops:  %2$f \n") % (1e-9* nOps/(nMillisecs/1000.0)) % nOps;


   copyMatrixToHost(y_newGPU,y_new_dev);

   freeMatrixDevice(G_dev);

   freeMatrixDevice(x_old_dev);
   freeMatrixDevice(y_new_dev);
   // =============================================================================================================

   // Compute on CPU Gauss Seidel =================================================================================

   platformstl::performance_counter counter;
   counter.start();

   y_newCPU.noalias() = G * x_old;

   counter.stop();
   cout << "CPU  Iteration time: " << counter.get_milliseconds() <<" ms" <<endl;



   if(utilities::compareArrays(y_newGPU.data(),y_newCPU.data(),MG,(GPUPrec)1e-4)){
      cout << "Sgemv GPU/CPU identical!...." << endl;
   }else{
      cout << "Sgemv GPU/CPU NOT identical!...." << endl;

   }
}

// colmajor transposed
void sgemvTransposed(int MG , int NG){
   // sgemv test

   srand ( time(NULL) );
   cout << " sGemv Test  Transposed======================================="<<endl;


   cublasHandle_t cublasHandle;
   CHECK_CUBLAS(cublasCreate(&cublasHandle));

   // Generate Matrices
   Eigen::Matrix<GPUPrec,Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> G(MG,NG); // Saved as rowmajor -> which is interpreted as colmajor from blas which is the transposed of G
   G.setRandom();


   Eigen::Matrix<GPUPrec,Eigen::Dynamic, 1> x_old(NG);
   Eigen::Matrix<GPUPrec,Eigen::Dynamic, 1> y_newCPU(MG);
   Eigen::Matrix<GPUPrec,Eigen::Dynamic, 1> y_newGPU(MG);
   x_old.setRandom();



   // GPU GAUSS SEIDEL ====================================================================================
   // Load to GPU
   Matrix<GPUPrec> G_dev, x_old_dev, y_new_dev;




   // Cuda Event Create ==============================================================
   cudaEvent_t start, stop, startKernel, stopKernel, startCopy,stopCopy;
   CHECK_CUDA(cudaEventCreate(&start));
   CHECK_CUDA(cudaEventCreate(&stop));
   CHECK_CUDA(cudaEventCreate(&startKernel));
   CHECK_CUDA(cudaEventCreate(&stopKernel));
   CHECK_CUDA(cudaEventCreate(&startCopy));
   CHECK_CUDA(cudaEventCreate(&stopCopy));

   CHECK_CUDA(cudaEventRecord(start,0));


   //Copy Data
   CHECK_CUDA(cudaEventRecord(startCopy,0));
   mallocAndCopyMatrixToDevice(G_dev, G);
   mallocAndCopyMatrixToDevice(x_old_dev,x_old);
   mallocMatrixDevice(y_new_dev,MG,1,false);


   CHECK_CUDA(cudaEventRecord(stopCopy,0));
   CHECK_CUDA(cudaEventSynchronize(stopCopy));

   float elapsedTimeCopy,elapsedKernelTime;
   CHECK_CUDA( cudaEventElapsedTime(&elapsedTimeCopy,startCopy,stopCopy));
   cout << "Copy time: " << elapsedTimeCopy <<" ms" << endl;


   CHECK_CUDA(cudaEventRecord(startKernel,0));

   GPUPrec a = 1.0;
   GPUPrec b = 0;

   int maxKernelLoops = 10;
   for(int i = 0; i<maxKernelLoops; i++){
      CHECK_CUBLAS(cublasDgemv(
         cublasHandle, 
         CUBLAS_OP_T ,
         G_dev.N, 
         G_dev.M,  
         &a,
         G_dev.data,
         G_dev.outerStride,
         x_old_dev.data,
         1,
         &b,
         y_new_dev.data,
         1)
         );
   }

   CHECK_CUDA(cudaEventRecord(stopKernel,0));
   CHECK_CUDA(cudaEventSynchronize(stopKernel));


   CHECK_CUDA(cudaThreadSynchronize());
   printf("Iterations finished!\n");
   CHECK_CUDA( cudaEventElapsedTime(&elapsedKernelTime,startKernel,stopKernel));
   cout << "GPU Iteration time: " << elapsedKernelTime <<" ms" << endl;
   double nMillisecs = (double)elapsedKernelTime / (double)maxKernelLoops;
   double nOps = 2.0*NG *(MG*1);
   cout<< format("Kernel Time :  %1$8.6f ms \n") % nMillisecs;
   cout<< format("GPU Gigaflops :  %1$5.6f Gflops/s , Ops:  %2$f \n") % (1e-9* nOps/(nMillisecs/1000.0)) % nOps;


   copyMatrixToHost(y_newGPU,y_new_dev);

   freeMatrixDevice(G_dev);
   freeMatrixDevice(x_old_dev);
   freeMatrixDevice(y_new_dev);
   // =============================================================================================================

   // Compute on CPU Gauss Seidel =================================================================================

   platformstl::performance_counter counter;
   counter.start();

   y_newCPU.noalias() = G * x_old;

   counter.stop();
   cout << "CPU  Iteration time: " << counter.get_microseconds()*1e-3 <<" ms" <<endl;



   if(utilities::compareArrays(y_newGPU.data(),y_newCPU.data(),MG,(GPUPrec)1e-4)){
      cout << "Sgemv GPU/CPU identical!...." << endl;
   }else{
      cout << "Sgemv GPU/CPU NOT identical!...." << endl;
   }
}


////

// Observations show that sgemvNormal is as fast as transposed variant if size is big! 1024 and bigger
// If size is small , then the transposed variant is always faster...

////
int
   main(int argc, char** argv)
{
   printAllCudaDeviceSpecs();

   int MG = 1024;
   int NG = 256;
   
   sgemvNormal(MG,NG);
   sgemvTransposed(MG,NG);


   system("pause");
}
