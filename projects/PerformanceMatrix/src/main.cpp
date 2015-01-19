
// includes, system
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>


#include "CudaFramework/CudaModern/CudaUtilities.hpp"
#include "CudaFramework/Kernels/MatrixMultGPU/MatrixMultGPU.hpp"
#include "CudaFramework/CudaModern/CudaError.hpp"


using namespace std;

using namespace utilCuda;
using namespace matrixMultGPU;

#define LOG_TO_FILE 0
////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{

	cout << "[Main started!!]" <<std::endl;

	// Set up the log files =================================
	std::streambuf * buf;

	std::ofstream ofsLog;
	std::ofstream ofsData;



#if LOG_TO_FILE == 1
		ofsLog.open("Log.txt");
      std::ostream oLog(ofsLog.rdbuf());
#else
      std::ostream oLog(std::cout.rdbuf());
#endif



   ofsData.open("Data2.txt");
   std::ostream oData(ofsData.rdbuf());
   cudaDeviceProp props;
   CHECK_CUDA(cudaGetDeviceProperties(&props,0));
   writeCudaDeviceProbs(oData,props,0);
   std::cout.rdbuf(oData.rdbuf());
   randomMatrixMult(2); //cublas
   //randomMatrixMult(4); //row
   randomMatrixMult(3); // col
   randomMatrixMult(5); // col
   randomMatrixMult(6); // col optimized
   randomMatrixMult(7); // from internet
   ofsData.close();


   // Do GridDim/Thread Test for general global matrix mult
   ofsData.open("Data0.txt");
   oData.rdbuf(ofsData.rdbuf());
	CHECK_CUDA(cudaGetDeviceProperties(&props,0));
	writeCudaDeviceProbs(oData,props,0);
   performanceTestMatrixMult(oData,oLog,0);
   ofsData.close();

   // Do GridDim/Thread Test for shared global matrix mult
   ofsData.open("Data1.txt");
   oData.rdbuf(ofsData.rdbuf());
   CHECK_CUDA(cudaGetDeviceProperties(&props,0));
   writeCudaDeviceProbs(oData,props,0);
   performanceTestMatrixMult(oData,oLog,1);
   ofsData.close();





	cout << "[Main ended!!]" <<std::endl;
    system("pause");
}
