// ========================================================================================
//  CudaFramework
//  Copyright (C) 2014 by Gabriel Nützi <nuetzig (at) imes (d0t) mavt (d0t) ethz (d0t) ch>
//
//  This Source Code Form is subject to the terms of the GNU GPL 3.0 licence. 
//  If a copy of the GNU GPL 3.0 was not distributed with this
//  file, you can obtain one at http://opensource.org/licenses/GPL-3.0.
// ========================================================================================


#include "CudaFramework/Kernels/TestsGPU/TestsGPU.hpp"

#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <fstream>


#include <stdlib.h>
#include <time.h>


#include "CudaFramework/CudaModern/CudaError.hpp"
#include "CudaFramework/General/Utilities.hpp"

#include <cuda_runtime.h>



using namespace std;

void testsGPU::branchTest(){

   typedef double GPUPrec;

   GPUPrec a[2] = {0.0,0.0};

   cout << "a[0]:" << a[0] <<std::endl;
   cout << "a[1]:" << a[1] <<std::endl;
   cout << " Run Kernel" << endl;
   GPUPrec * a_dev;
   CHECK_CUDA(cudaMalloc((void**) &a_dev,  2* sizeof(GPUPrec)));

   CHECK_CUDA(cudaMemcpy(a_dev, a, 2  * sizeof(GPUPrec), cudaMemcpyHostToDevice));

   branchTest_kernelWrap(a_dev);
   CHECK_CUDA(cudaGetLastError());
   CHECK_CUDA(cudaThreadSynchronize());
   CHECK_CUDA(cudaMemcpy(a, a_dev, 2  * sizeof(GPUPrec), cudaMemcpyDeviceToHost));

   cout << "a[0]:" << a[0] <<std::endl;
   cout << "a[1]:" << a[1] <<std::endl;
}



void testsGPU::cudaStrangeBehaviour(){

   typedef float PREC ;
   ofstream m_matlab_file;
   m_matlab_file.close();
   m_matlab_file.open("StrangeCudaBehaviour.m", ios::trunc | ios::out);
   m_matlab_file.clear();
   if (m_matlab_file.is_open())
   {
     cout << " File opened: " << "StrangeCudaBehaviour.m"<<std::endl;
   }


   //PROBLEM
   PREC x[3] = {0.0,0.0,0.0};
   PREC t[3] = {0.0,0.0,0.0};

   PREC d[3] = {4,5,6};

   PREC mu[1] = {0.5};

   PREC y[3] = {-1,-1,-1}; //Get overwritten in kernel!

   m_matlab_file << setprecision(9) << "x= [" << x[0] << " , " << x[1] << " , " << x[2] << "]'" <<endl;
   m_matlab_file << setprecision(9) << "t= [" << t[0] << " , " << t[1] << " , " << t[2] << "]'" <<endl;
   m_matlab_file << setprecision(9) << "d= [" << d[0] << " , " << d[1] << " , " << d[2] << "]'" <<endl;
   m_matlab_file << setprecision(9) << "mu= [" << mu[0] << "]'" <<endl;

   PREC * t_dev, *x_dev, *d_dev, *mu_dev, *y_dev;

   cudaDeviceReset();
   CHECK_CUDA(cudaMalloc((void**) &t_dev,  3 * sizeof(PREC)));
   CHECK_CUDA(cudaMalloc((void**) &x_dev,  3 * sizeof(PREC)));
   CHECK_CUDA(cudaMalloc((void**) &d_dev,  3 * sizeof(PREC)));
   CHECK_CUDA(cudaMalloc((void**) &mu_dev, 1 * sizeof(PREC)));
   CHECK_CUDA(cudaMalloc((void**) &y_dev,  3 * sizeof(PREC)));

   CHECK_CUDA(cudaMemcpy(t_dev,  t,    3  * sizeof(PREC), cudaMemcpyHostToDevice));
   CHECK_CUDA(cudaMemcpy(x_dev,  x,    3  * sizeof(PREC), cudaMemcpyHostToDevice));
   CHECK_CUDA(cudaMemcpy(d_dev,  d,    3  * sizeof(PREC), cudaMemcpyHostToDevice));
   CHECK_CUDA(cudaMemcpy(mu_dev, mu,   1  * sizeof(PREC), cudaMemcpyHostToDevice));
   CHECK_CUDA(cudaMemcpy(y_dev,  y,    3  * sizeof(PREC), cudaMemcpyHostToDevice));


   strangeCudaBehaviour_wrap(y_dev,mu_dev,d_dev,t_dev,x_dev);

   CHECK_CUDA(cudaGetLastError());
   CHECK_CUDA(cudaThreadSynchronize());

   PREC t_after[3];
   PREC y_after[3];

   CHECK_CUDA(cudaMemcpy(t_after, t_dev, 3  * sizeof(PREC), cudaMemcpyDeviceToHost));
   CHECK_CUDA(cudaMemcpy(y_after, y_dev, 3  * sizeof(PREC), cudaMemcpyDeviceToHost));
   cout << "#RESULTS:" <<endl;
   m_matlab_file << setprecision(9) << "y_afterGPU= [" << y_after[0] << " , " << y_after[1] << " , " << y_after[2] << "]'" <<endl;
   m_matlab_file << setprecision(9) << "t_afterGPU= [" << t_after[0] << " , " << t_after[1] << " , " << t_after[2] << "]'" <<endl;


}


