
// includes, system
#include <iostream>
#include <stdio.h>

////////////////////////////////////////////////////////////////////////////////
// Cuda extern C includes
#include "CudaFramework/CudaModern/CudaUtilities.hpp"
#include "CudaFramework/Kernels/MatrixMultGPU/MatrixMultGPU.hpp"
#include "CudaFramework/Kernels/MatrixVectorMultGPU/MatrixVectorMultGPU.hpp"
#include "CudaFramework/Kernels/VectorAddGPU/VectorAddGPU.hpp"
#include "CudaFramework/CudaModern/CudaError.hpp"

#include "CudaFramework/General/PlatformDefines.hpp"

using namespace std;

using namespace utilCuda;
using namespace matrixMultGPU;
using namespace matrixVectorMultGPU;
using namespace vectorAddGPU;
////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char** argv)
{
   cout << "[Main started!!]" <<std::endl;

   // Set up the log files =================================
   //std::streambuf * buf;

   std::ofstream ofsLog;
   std::ofstream ofsData;

   std::ostream oData(std::cout.rdbuf());

#if LOG_TO_FILE == 1
   ofsLog.open("Log.txt");
   std::ostream oLog(ofsLog.rdbuf());
#else
   std::ostream oLog(std::cout.rdbuf());
#endif


	printAllCudaDeviceSpecs();

    // Matrix Mult
	//randomMatrixMult(1);
	randomMatrixMult(2); //cublas
	///randomMatrixMult(4); //row
	//randomMatrixMult(5); // col
	//randomMatrixMult(6); // col optimized
	//randomMatrixMult(7); // from internet
	//randomVectorAdd(0);



    HOLD_SYSTEM
}
