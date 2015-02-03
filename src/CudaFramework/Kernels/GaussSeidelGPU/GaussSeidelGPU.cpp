// ========================================================================================
//  CudaFramework
//  Copyright (C) 2014 by Gabriel Nützi <nuetzig (at) imes (d0t) mavt (d0t) ethz (d0t) ch>
//
//  This Source Code Form is subject to the terms of the GNU GPL 3.0 licence. 
//  If a copy of the GNU GPL 3.0 was not distributed with this
//  file, you can obtain one at http://opensource.org/licenses/GPL-3.0.
// ========================================================================================


#include "CudaFramework/Kernels/GaussSeidelGPU/GaussSeidelGPU.hpp"

#include <stdio.h>
#include <iostream>
#include <fstream>

#include <stdlib.h>
#include <time.h>


#include "CudaFramework/CudaModern/CudaError.hpp"
#include "CudaFramework/General/Utilities.hpp"
#include "CudaFramework/CudaModern/CudaMatrixUtilities.hpp"

#include <cuda_runtime.h>

#include <fstream>


using namespace std;
using namespace Utilities;
using namespace utilCuda;


#define WRITE_TEST_TO_FILE 1

//void  gaussSeidelGPU::gaussSeidelTest(){
//
//   srand ( time(NULL) );
//
//   cout << " Gauss-Seidel Test Iteration ======================================="<<std::endl;
//
//#if WRITE_TEST_TO_FILE == 1
//   std::ofstream matlab_file;
//   matlab_file.close();
//   matlab_file.open("GaussSeidelStep.m", ios::trunc | ios::out);
//   matlab_file.clear();
//   if (matlab_file.is_open())
//   {
//      cout << " File opened: " << "GaussSeidelStep.m"<<std::endl;
//   }
//   Eigen::IOFormat Matlab(Eigen::FullPrecision, 0, ", ", ";\n", "", "", "[", "]");
//#endif
//
//
//   int MG = 1024*4;
//   int NG = MG;
//
//
//      // Generate Matrices
//      Eigen::Matrix<GPUPrec,Eigen::Dynamic, Eigen::Dynamic> G(MG,NG);
//      G.setRandom();
//      G += (Eigen::Matrix<GPUPrec,Eigen::Dynamic,1>::Ones(MG) * 100).asDiagonal();
//      Eigen::Matrix<GPUPrec,Eigen::Dynamic, 1> c(MG);
//      c.setRandom();
//
//      Eigen::Matrix<GPUPrec,Eigen::Dynamic, 1> t(MG);
//      Eigen::Matrix<GPUPrec,Eigen::Dynamic, 1> t_newCPU(MG);
//      Eigen::Matrix<GPUPrec,Eigen::Dynamic, 1> t_newGPU(MG);
//      t.setZero();
//
//      Eigen::Matrix<GPUPrec,Eigen::Dynamic, 1> x_old(MG);
//      Eigen::Matrix<GPUPrec,Eigen::Dynamic, 1> x_newCPU(MG);
//      Eigen::Matrix<GPUPrec,Eigen::Dynamic, 1> x_newGPU(MG);
//      x_old.setZero();
//
//#if WRITE_TEST_TO_FILE == 1
//   matlab_file << "G=" << G.format(Matlab)<<";" <<std::endl;
//   matlab_file << "c=" << c.transpose().format(Matlab)<<"';" <<std::endl;
//   matlab_file << "t=" << t.transpose().format(Matlab)<<"';" <<std::endl;
//   matlab_file << "x_old=" << x_old.transpose().format(Matlab)<<"';" <<std::endl;
//#endif
//
//   // GPU GAUSS SEIDEL ====================================================================================
//   // Load to GPU
//   Matrix<GPUPrec> G_dev,c_dev,t_dev, x_old_dev;
//   mallocMatrixDevice(G_dev,G);
//   mallocMatrixDevice(c_dev,c);
//   mallocMatrixDevice(t_dev,t);
//   mallocMatrixDevice(x_old_dev,c);
//
//
//
//   // Cuda Event Create ==============================================================
//   cudaEvent_t start, stop, startKernel, stopKernel, startCopy,stopCopy;
//   CHECK_CUDA(cudaEventCreate(&start));
//   CHECK_CUDA(cudaEventCreate(&stop));
//   CHECK_CUDA(cudaEventCreate(&startKernel));
//   CHECK_CUDA(cudaEventCreate(&stopKernel));
//   CHECK_CUDA(cudaEventCreate(&startCopy));
//   CHECK_CUDA(cudaEventCreate(&stopCopy));
//
//   CHECK_CUDA(cudaEventRecord(start,0));
//
//
//   //Copy Data
//   CHECK_CUDA(cudaEventRecord(startCopy,0));
//   copyMatrixToDevice(G_dev, G);
//   copyMatrixToDevice(c_dev, c);
//   copyMatrixToDevice(t_dev,t);
//   copyMatrixToDevice(x_old_dev,x_old);
//   CHECK_CUDA(cudaEventRecord(stopCopy,0));
//   CHECK_CUDA(cudaEventSynchronize(stopCopy));
//
//   float elapsedTimeCopy,elapsedKernelTime;
//   CHECK_CUDA( cudaEventElapsedTime(&elapsedTimeCopy,startCopy,stopCopy));
//   cout << "Copy time: " << elapsedTimeCopy <<" ms" << endl;
//
//
//   int m = 64;
//   int g = MG / m;
//
//
//   CHECK_CUDA(cudaEventRecord(startKernel,0));
//
//   int nMaxIterations = 100;
//   for (int nIter=0; nIter< nMaxIterations ; nIter++){
//      // Do one Gauss Seidel Step!
//      for(int j_g = 0 ; j_g < g; j_g++){
//
//         // Usaeble Kernels:
//         //    blockGaussSeidelStepA_kernelWrap ,
//         //    blockGaussSeidelStepACorrect_kernelWrap
//
//         gaussSeidelGPU::blockGaussSeidelStepACorrect_kernelWrap(G_dev,c_dev,t_dev,x_old_dev, j_g);
//         gaussSeidelGPU::blockGaussSeidelStepB_kernelWrap(G_dev,c_dev,t_dev,x_old_dev, j_g);
//
//      }
//   }
//   CHECK_CUDA(cudaEventRecord(stopKernel,0));
//   CHECK_CUDA(cudaEventSynchronize(stopKernel));
//
//   printf("Kernel started...\n");
//   CHECK_CUDA(cudaThreadSynchronize());
//   printf("Kernel finished!\n");
//   CHECK_CUDA( cudaEventElapsedTime(&elapsedKernelTime,startKernel,stopKernel));
//   cout << "GPU Iteration time: " << elapsedKernelTime <<" ms" << endl;
//
//   // Copy results back
//   copyMatrixToHost(t_newGPU,t_dev);
//   copyMatrixToHost(x_newGPU,x_old_dev);
//#if WRITE_TEST_TO_FILE == 1
//   matlab_file << "t_newGPU=" << t_newGPU.transpose().format(Matlab)<<"';" <<std::endl;
//   matlab_file << "x_newGPU=" << x_newGPU.transpose().format(Matlab)<<"';" <<std::endl;
//#endif
//   freeMatrixDevice(G_dev);
//   freeMatrixDevice(t_dev);
//   freeMatrixDevice(c_dev);
//   freeMatrixDevice(x_old_dev);
//   // =============================================================================================================
//
//   // Compute on CPU Gauss Seidel =================================================================================
//
//   platformstl::performance_counter counter;
//   counter.start();
//   utilities::blockGaussSeidelCorrect(G,c,x_old,nMaxIterations,m);
//   counter.stop();
//   cout << "CPU  Iteration time: " << counter.get_milliseconds() <<" ms" <<std::endl;
//
//   x_newCPU = x_old;
//   t_newCPU = t;
//
//   if(compareArrays(x_newGPU.data(),x_newCPU.data(),MG,(GPUPrec)1e-4)){
//      cout << "GaussSeidel GPU/CPU identical!...." << endl;
//   }else{
//      cout << "GaussSeidel GPU/CPU NOT identical!...." << endl;
//   }
//#if WRITE_TEST_TO_FILE == 1
//   matlab_file << "t_newCPU=" << t_newCPU.transpose().format(Matlab)<<"';" <<std::endl;
//   matlab_file << "x_newCPU=" << x_newCPU.transpose().format(Matlab)<<"';" <<std::endl;
//
//   matlab_file.close();
//#endif
//}
//
//void  gaussSeidelGPU::gaussSeidelTestNoDivision(){
//   srand ( time(NULL) );
//   cout << " Gauss-Seidel Test  NO Division Iteration ======================================="<<std::endl;
//
//#if WRITE_TEST_TO_FILE == 1
//   std::ofstream matlab_file;
//   matlab_file.close();
//   matlab_file.open("GaussSeidelStep.m", ios::trunc | ios::out);
//   matlab_file.clear();
//   if (matlab_file.is_open())
//   {
//      cout << " File opened: " << "GaussSeidelStep.m"<<std::endl;
//   }
//   Eigen::IOFormat Matlab(Eigen::FullPrecision, 0, ", ", ";\n", "", "", "[", "]");
//#endif
//
//
//   int MG = 1024;
//   int NG = MG;
//
//   // Generate Matrices
//   Eigen::Matrix<GPUPrec,Eigen::Dynamic, Eigen::Dynamic> G(MG,NG);
//   G.setRandom();
//   G += (Eigen::Matrix<GPUPrec,Eigen::Dynamic,1>::Ones(MG) * 100).asDiagonal();
//   Eigen::Matrix<GPUPrec,Eigen::Dynamic, 1> c(MG);
//   c.setRandom();
//
//   Eigen::Matrix<GPUPrec,Eigen::Dynamic, Eigen::Dynamic> T(MG,NG);
//   Eigen::Matrix<GPUPrec,Eigen::Dynamic, 1> d(MG);
//   T = G.diagonal().asDiagonal().inverse() * G;
//   d = G.diagonal().asDiagonal().inverse() * c;
//
//   Eigen::Matrix<GPUPrec,Eigen::Dynamic, 1> t(MG);
//   Eigen::Matrix<GPUPrec,Eigen::Dynamic, 1> t_newCPU(MG);
//   Eigen::Matrix<GPUPrec,Eigen::Dynamic, 1> t_newGPU(MG);
//   t.setZero();
//
//   Eigen::Matrix<GPUPrec,Eigen::Dynamic, 1> x_old(MG);
//   Eigen::Matrix<GPUPrec,Eigen::Dynamic, 1> x_newCPU(MG);
//   Eigen::Matrix<GPUPrec,Eigen::Dynamic, 1> x_newGPU(MG);
//   x_old.setZero();
//
//#if WRITE_TEST_TO_FILE == 1
//   matlab_file << "G=" << G.format(Matlab)<<";" <<std::endl;
//   matlab_file << "c=" << c.transpose().format(Matlab)<<"';" <<std::endl;
//   matlab_file << "t=" << t.transpose().format(Matlab)<<"';" <<std::endl;
//   matlab_file << "x_old=" << x_old.transpose().format(Matlab)<<"';" <<std::endl;
//#endif
//
//   // GPU GAUSS SEIDEL ====================================================================================
//   // Load to GPU
//   Matrix<GPUPrec> T_dev,d_dev,t_dev, x_old_dev;
//
//
//
//
//   // Cuda Event Create ==============================================================
//   cudaEvent_t start, stop, startKernel, stopKernel, startCopy,stopCopy;
//   CHECK_CUDA(cudaEventCreate(&start));
//   CHECK_CUDA(cudaEventCreate(&stop));
//   CHECK_CUDA(cudaEventCreate(&startKernel));
//   CHECK_CUDA(cudaEventCreate(&stopKernel));
//   CHECK_CUDA(cudaEventCreate(&startCopy));
//   CHECK_CUDA(cudaEventCreate(&stopCopy));
//
//   CHECK_CUDA(cudaEventRecord(start,0));
//
//
//   //Copy Data
//   CHECK_CUDA(cudaEventRecord(startCopy,0));
//   mallocAndCopyMatrixToDevice(T_dev, T);
//   mallocAndCopyMatrixToDevice(d_dev, d);
//   mallocAndCopyMatrixToDevice(t_dev,t);
//   mallocAndCopyMatrixToDevice(x_old_dev,x_old);
//   bool *convergedFlag_dev;
//   cudaMalloc(&convergedFlag_dev,sizeof(bool));
//
//   CHECK_CUDA(cudaEventRecord(stopCopy,0));
//   CHECK_CUDA(cudaEventSynchronize(stopCopy));
//
//   float elapsedTimeCopy,elapsedKernelTime;
//   CHECK_CUDA( cudaEventElapsedTime(&elapsedTimeCopy,startCopy,stopCopy));
//   cout << "Copy time: " << elapsedTimeCopy <<" ms" << endl;
//
//
//   int m = 64;
//   int g = MG / m;
//
//
//   CHECK_CUDA(cudaEventRecord(startKernel,0));
//
//   int nMaxIterations = 200;
//   for (int nIter=0; nIter< nMaxIterations ; nIter++){
//      // Do one Gauss Seidel Step!
//
//      for(int j_g = 0 ; j_g < g; j_g++){
//
//         // Usaeble Kernels:
//         //    blockGaussSeidelStepACorrectNoDivision_kernelWrap ,
//         //    blockGaussSeidelStepNoDivision_kernelWrap
//
//         gaussSeidelGPU::blockGaussSeidelStepACorrectNoDivision_kernelWrap(T_dev,d_dev,t_dev,x_old_dev, j_g);
//         gaussSeidelGPU::blockGaussSeidelStepB_kernelWrap(T_dev,d_dev,t_dev,x_old_dev, j_g);
//
//      }
//   }
//   CHECK_CUDA(cudaEventRecord(stopKernel,0));
//   CHECK_CUDA(cudaEventSynchronize(stopKernel));
//
//   printf("Kernel started...\n");
//   CHECK_CUDA(cudaThreadSynchronize());
//   printf("Kernel finished!\n");
//   CHECK_CUDA( cudaEventElapsedTime(&elapsedKernelTime,startKernel,stopKernel));
//   cout << "GPU Iteration time: " << elapsedKernelTime <<" ms" << endl;
//
//   // Copy results back
//   copyMatrixToHost(t_newGPU,t_dev);
//   copyMatrixToHost(x_newGPU,x_old_dev);
//#if WRITE_TEST_TO_FILE == 1
//   matlab_file << "t_newGPU=" << t_newGPU.transpose().format(Matlab)<<"';" <<std::endl;
//   matlab_file << "x_newGPU=" << x_newGPU.transpose().format(Matlab)<<"';" <<std::endl;
//#endif
//   freeMatrixDevice(T_dev);
//   freeMatrixDevice(t_dev);
//   freeMatrixDevice(d_dev);
//   freeMatrixDevice(x_old_dev);
//   // =============================================================================================================
//
//   // Compute on CPU Gauss Seidel =================================================================================
//
//   platformstl::performance_counter counter;
//   counter.start();
//   utilities::blockGaussSeidelCorrect(G,c,x_old,nMaxIterations,m);
//   counter.stop();
//   cout << "CPU  Iteration time: " << counter.get_milliseconds() <<" ms" <<std::endl;
//
//   x_newCPU = x_old;
//   t_newCPU = t;
//
//
//   if(compareArrays(x_newGPU.data(),x_newCPU.data(),MG,(GPUPrec)1e-4)){
//      cout << "GaussSeidel GPU/CPU identical!...." << endl;
//   }else{
//      cout << "GaussSeidel GPU/CPU NOT identical!...." << endl;
//   }
//#if WRITE_TEST_TO_FILE == 1
//   matlab_file << "t_newCPU=" << t_newCPU.transpose().format(Matlab)<<"';" <<std::endl;
//   matlab_file << "x_newCPU=" << x_newCPU.transpose().format(Matlab)<<"';" <<std::endl;
//
//   matlab_file.close();
//#endif
//}
//
//void  gaussSeidelGPU::gaussSeidelTestNoDivisionWithError(){
//   //srand ( time(NULL) );
//   cout << " Gauss-Seidel Test  NO Division Iteration with Error Check ======================================="<<std::endl;
//#if WRITE_TEST_TO_FILE == 1
//   std::ofstream matlab_file;
//   matlab_file.close();
//   matlab_file.open("GaussSeidelStep.m", ios::trunc | ios::out);
//   matlab_file.clear();
//   if (matlab_file.is_open())
//   {
//      cout << " File opened: " << "GaussSeidelStep.m"<<std::endl;
//   }
//   Eigen::IOFormat Matlab(Eigen::FullPrecision, 0, ", ", ";\n", "", "", "[", "]");
//#endif
//
//
//
//   int MG = 1024;
//   int NG = MG;
//
//   // Generate Matrices
//   Eigen::Matrix<GPUPrec,Eigen::Dynamic, Eigen::Dynamic> G(MG,NG);
//   G.setRandom();
//   G += (Eigen::Matrix<GPUPrec,Eigen::Dynamic,1>::Ones(MG) * 20).asDiagonal();
//   Eigen::Matrix<GPUPrec,Eigen::Dynamic, 1> c(MG);
//   c.setRandom();
//
//   Eigen::Matrix<GPUPrec,Eigen::Dynamic, Eigen::Dynamic> T(MG,NG);
//   Eigen::Matrix<GPUPrec,Eigen::Dynamic, 1> d(MG);
//   T = G.diagonal().asDiagonal().inverse() * G;
//   d = G.diagonal().asDiagonal().inverse() * c;
//
//
//   Eigen::Matrix<GPUPrec,Eigen::Dynamic, 1> t(MG);
//   Eigen::Matrix<GPUPrec,Eigen::Dynamic, 1> t_newCPU(MG);
//   Eigen::Matrix<GPUPrec,Eigen::Dynamic, 1> t_newGPU(MG);
//   t.setZero();
//
//
//   Eigen::Matrix<GPUPrec,Eigen::Dynamic, 1> x_old(MG);
//   Eigen::Matrix<GPUPrec,Eigen::Dynamic, 1> x_newCPU(MG);
//   Eigen::Matrix<GPUPrec,Eigen::Dynamic, 1> x_newGPU(MG);
//   x_old.setZero();
//
//
//#if WRITE_TEST_TO_FILE == 1
//   matlab_file << "G=" << G.format(Matlab)<<";" <<std::endl;
//   matlab_file << "c=" << c.transpose().format(Matlab)<<"';" <<std::endl;
//   matlab_file << "t=" << t.transpose().format(Matlab)<<"';" <<std::endl;
//   matlab_file << "x_old=" << x_old.transpose().format(Matlab)<<"';" <<std::endl;
//#endif
//
//   // GPU GAUSS SEIDEL ====================================================================================
//   // Load to GPU
//   Matrix<GPUPrec> T_dev,d_dev,t_dev, x_old_dev;
//
//
//
//
//   // Cuda Event Create ==============================================================
//   cudaEvent_t start, stop, startKernel, stopKernel, startCopy,stopCopy;
//   CHECK_CUDA(cudaEventCreate(&start));
//   CHECK_CUDA(cudaEventCreate(&stop));
//   CHECK_CUDA(cudaEventCreate(&startKernel));
//   CHECK_CUDA(cudaEventCreate(&stopKernel));
//   CHECK_CUDA(cudaEventCreate(&startCopy));
//   CHECK_CUDA(cudaEventCreate(&stopCopy));
//
//   CHECK_CUDA(cudaEventRecord(start,0));
//
//
//   //Copy Data
//   CHECK_CUDA(cudaEventRecord(startCopy,0));
//   mallocAndCopyMatrixToDevice(T_dev, T);
//   mallocAndCopyMatrixToDevice(d_dev, d);
//   mallocAndCopyMatrixToDevice(t_dev,t);
//   mallocAndCopyMatrixToDevice(x_old_dev,x_old);
//   bool *pConvergedFlag_dev;
//   bool convergedFlag;
//   cudaMalloc(&pConvergedFlag_dev,sizeof(bool));
//
//   CHECK_CUDA(cudaEventRecord(stopCopy,0));
//   CHECK_CUDA(cudaEventSynchronize(stopCopy));
//
//   float elapsedTimeCopy,elapsedKernelTime;
//   CHECK_CUDA( cudaEventElapsedTime(&elapsedTimeCopy,startCopy,stopCopy));
//   cout << "Copy time: " << elapsedTimeCopy <<" ms" << endl;
//
//
//   int m = 64;
//   int g = MG / m;
//
//
//   CHECK_CUDA(cudaEventRecord(startKernel,0));
//
//   int nIter = 0;
//   int nMaxIterations = 50000;
//   int nCheckConvergedFlag = 1; // Number of iterations to check converged flag
//   GPUPrec absTOL = 1e-8;
//   GPUPrec relTOL = 1e-10;
//   printf("Iterations started...\n");
//   for (nIter=0; nIter< nMaxIterations ; nIter++){
//      // Do one Gauss Seidel Step!
//
//      // First set the converged flag to 1
//      cudaMemset(pConvergedFlag_dev,1,sizeof(bool));
//      // Do one step
//      for(int j_g = 0 ; j_g < g; j_g++){
//
//         gaussSeidelGPU::blockGaussSeidelStepACorrectNoDivision_kernelWrap(T_dev,d_dev,t_dev,x_old_dev, j_g, pConvergedFlag_dev,    absTOL,relTOL);
//         gaussSeidelGPU::blockGaussSeidelStepB_kernelWrap(T_dev,d_dev,t_dev,x_old_dev, j_g);
//
//      }
//
//      // Check each nCheckConvergedFlag  the converged flag
//      if(nIter % nCheckConvergedFlag == 0){
//         // Download the flag from the GPU and check
//         cudaMemcpy(&convergedFlag,pConvergedFlag_dev,sizeof(bool),cudaMemcpyDeviceToHost);
//         if(convergedFlag == true){
//            // Converged
//            break;
//         }
//      }
//
//   }
//
//   CHECK_CUDA(cudaEventRecord(stopKernel,0));
//   CHECK_CUDA(cudaEventSynchronize(stopKernel));
//
//
//   CHECK_CUDA(cudaThreadSynchronize());
//   printf("Iterations finished!\n");
//   CHECK_CUDA( cudaEventElapsedTime(&elapsedKernelTime,startKernel,stopKernel));
//   cout << "GPU Iteration time: " << elapsedKernelTime <<" ms" << endl;
//   cout << "Converged Flag: " << convergedFlag <<std::endl;
//   cout << "nIterations: " << nIter <<std::endl;
//   if (nIter == nMaxIterations){
//      cout << "Not converged! Max. Iterations reached."<<std::endl;
//   }
//   // Copy results back
//   copyMatrixToHost(t_newGPU,t_dev);
//   copyMatrixToHost(x_newGPU,x_old_dev);
//
//#if WRITE_TEST_TO_FILE == 1
//   matlab_file << "t_newGPU=" << t_newGPU.transpose().format(Matlab)<<"';" <<std::endl;
//   matlab_file << "x_newGPU=" << x_newGPU.transpose().format(Matlab)<<"';" <<std::endl;
//#endif
//
//
//   freeMatrixDevice(T_dev);
//   freeMatrixDevice(t_dev);
//   freeMatrixDevice(d_dev);
//   freeMatrixDevice(x_old_dev);
//   // =============================================================================================================
//
//   // Compute on CPU Gauss Seidel =================================================================================
//
//   platformstl::performance_counter counter;
//   counter.start();
//   utilities::blockGaussSeidelCorrect(G,c,x_old,nIter+1,m);
//   counter.stop();
//   cout << "CPU  Iteration time: " << counter.get_milliseconds() <<" ms" <<std::endl;
//
//   x_newCPU = x_old;
//   t_newCPU = t;
//
//
//   if(compareArrays(x_newGPU.data(),x_newCPU.data(),MG,(GPUPrec)1e-4)){
//      cout << "GaussSeidel GPU/CPU identical!...." << endl;
//   }else{
//      cout << "GaussSeidel GPU/CPU NOT identical!...." << endl;
//   }
//#if WRITE_TEST_TO_FILE == 1
//   matlab_file << "t_newCPU=" << t_newCPU.transpose().format(Matlab)<<"';" <<std::endl;
//   matlab_file << "x_newCPU=" << x_newCPU.transpose().format(Matlab)<<"';" <<std::endl;
//
//   matlab_file.close();
//#endif
//}

