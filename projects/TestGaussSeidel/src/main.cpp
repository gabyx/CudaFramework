
// includes, system
#include <iostream>
#include <stdlib.h>


////////////////////////////////////////////////////////////////////////////////
// Cuda extern C includes
#include "CudaFramework/CudaModern/CudaUtilities.hpp"
#include "CudaFramework/PerformanceTest/PerformanceTest.hpp"
#include "CudaFramework/Kernels/GaussSeidelGPU/GaussSeidelTestVariant.hpp"

#include "CudaFramework/Kernels/TestsGPU/TestsGPU.hpp"

using namespace utilCuda;
using namespace gaussSeidelGPU;
using namespace testsGPU;

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

int
main(int argc, char** argv)
{
	printAllCudaDeviceSpecs();


    typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         GaussSeidelBlockTestVariant<
            GaussSeidelBlockSettings<
               double,
               3,
               64, // Inactive always 64
               false,
               10,
               false,
               1,
               GaussSeidelBlockPerformanceTestSettings<64,150,2>,
               GaussSeidelBlockGPUVariantSettings<1,true>
            >
         >
    > test1;
    PerformanceTest<test1> A("test1");
    A.run();


   //cudaStrangeBehaviour();

   //branchTest();

    system("pause");
}
