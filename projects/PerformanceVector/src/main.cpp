
// includes, system
#include <stdio.h>
#include <iostream>
#include <fstream>

using namespace std;


////////////////////////////////////////////////////////////////////////////////
// Cuda extern C includes
#include "CudaFramework/CudaModern/CudaUtilities.hpp"
#include "CudaFramework/Kernels/VectorAddGPU/VectorAddGPU.hpp"
#include "CudaFramework/CudaModern/CudaError.hpp"


using namespace std;

using namespace utilCuda;
using namespace vectorAddGPU;


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
		buf = ofsLog.rdbuf();
#else
		buf = std::cout.rdbuf();
#endif
	std::ostream oLog(buf);
	ofsData.open("Data.txt");
	buf = ofsData.rdbuf();
	std::ostream oData(buf);
	// ======================================================

	cudaDeviceProp props;
	CHECK_CUDA(cudaGetDeviceProperties(&props,0));
	writeCudaDeviceProbs(oData,props,0);

	//randomVectorAdd();



	performanceTestVectorAdd(oData,oLog,0);

	cout << "[Main ended!!]" <<std::endl;
    system("pause");
}
