


#include <iostream>
#include <vector>
#include <stdio.h>
#include <chrono>
#include <iostream>
#include "CudaFramework/CudaModern/CudaContext.hpp"
#include "CudaFramework/CudaModern/CudaError.hpp"
#include "CudaFramework/Kernels/JORProxVel/ReductionKernel/ReductionTestVariant.hpp"
#include "CudaFramework/PerformanceTest/PerformanceTest.hpp"



//#include "CudaFramework/PerformanceTest/PerformanceTest.hpp"
////////////////////////////////////////////////////////////////////////////////
// Cuda extern C includes

#include <boost/variant.hpp>



int main()
{


    typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         ReductionTestVariant<
            ReductionSettings<
                double,
                ReductionTestRangeSettings<1000000,1000000,1,1>,
                ReductionGPUVariantSettings<1>
            >
         >
    > test1;

    std::cout<<"Performance Test Start"<<std::endl;
    PerformanceTest<test1> A("test1");
    A.run();
    std::cout<<"PErformance Test Stop"<<std::endl;


   {
        typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         ReductionTestVariant<
            ReductionSettings<
                double,
                ReductionTestRangeSettings<1000,100000,1000,20>,
                ReductionGPUVariantSettings<1>
            >
         >
    > RED_d_1k_100k_1k_20_v1;

    PerformanceTest<RED_d_1k_100k_1k_20_v1> A("RED_d_1k_100k_1k_20_v1");
    A.run();

    }

       {
        typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         ReductionTestVariant<
            ReductionSettings<
                double,
                ReductionTestRangeSettings<6000,600000,6000,20>,
                ReductionGPUVariantSettings<1>
            >
         >
    > RED_d_6k_600k_6k_20_v1;

    PerformanceTest<RED_d_6k_600k_6k_20_v1> A("RED_d_6k_600k_6k_20_v1");
    A.run();

    }
       {
        typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         ReductionTestVariant<
            ReductionSettings<
                double,
                ReductionTestRangeSettings<60000,6000000,60000,20>,
                ReductionGPUVariantSettings<1>
            >
         >
    > RED_d_60k_6000k_60k_20_v1;

    PerformanceTest<RED_d_60k_6000k_60k_20_v1> A("RED_d_60k_6000k_60k_20_v1");
    A.run();

    }


}



