


#include <iostream>
#include <vector>
#include <stdio.h>


#include <chrono>
#include <iostream>

#include "CudaFramework/Kernels/JORProxVel/JORProxVelKernel/JORProxVel.hpp"


#include "CudaFramework/CudaModern/CudaMatrixUtilities.hpp"


#include "CudaFramework/PerformanceTest/PerformanceTest.hpp"



//#include "CudaFramework/PerformanceTest/PerformanceTest.hpp"
////////////////////////////////////////////////////////////////////////////////
// Cuda extern C includes



int main()
{

    typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         JORProxVelTestVariant<
            JORProxVelSettings<
                double,
                JORProxVelTestRangeSettings<1000000,1000000,1,1>,
                JORProxVelGPUVariantSettings<1>
            >
         >
    > test1;

    std::cout<<"blabla"<<std::endl;
    PerformanceTest<test1> A("test1");
    A.run();
    std::cout<<"blabla End"<<std::endl;



    /*
    {
        typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         JORProxVelTestVariant<
            JORProxVelSettings<
                double,
                JORProxVelTestRangeSettings<1000,100000,1000,20>,
                JORProxVelGPUVariantSettings<1>
            >
         >
    > JPV_d_1k_100k_1k_20_v1;

    PerformanceTest<JPV_d_1k_100k_1k_20_v1> A("JPV_d_1k_100k_1k_20_v1");
    A.run();

    }
*/


/*
    {
        typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         JORProxVelTestVariant<
            JORProxVelSettings<
                double,
                JORProxVelTestRangeSettings<1000,100000,1000,20>,
                JORProxVelGPUVariantSettings<2>
            >
         >
    > JPV_d_1k_100k_1k_20_v2;

    PerformanceTest<JPV_d_1k_100k_1k_20_v2> A("JPV_d_1k_100k_1k_20_v2");
    A.run();

    }
*/


/*
    {
        typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         JORProxVelTestVariant<
            JORProxVelSettings<
                double,
                JORProxVelTestRangeSettings<1000,100000,1000,20>,
                JORProxVelGPUVariantSettings<3>
            >
         >
    > JPV_d_1k_100k_1k_20_v3;

    PerformanceTest<JPV_d_1k_100k_1k_20_v3> A("JPV_d_1k_100k_1k_20_v3");
    A.run();

    }
*/


/*
    {
        typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         JORProxVelTestVariant<
            JORProxVelSettings<
                double,
                JORProxVelTestRangeSettings<1000,100000,1000,20>,
                JORProxVelGPUVariantSettings<4>
            >
         >
    > JPV_d_1k_100k_1k_20_v4;

    PerformanceTest<JPV_d_1k_100k_1k_20_v4> A("JPV_d_1k_100k_1k_20_v4");
    A.run();

    }
*/


/*
    {
        typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         JORProxVelTestVariant<
            JORProxVelSettings<
                double,
                JORProxVelTestRangeSettings<1000,100000,1000,20>,
                JORProxVelGPUVariantSettings<5>
            >
         >
    > JPV_d_1k_100k_1k_20_v5;

    PerformanceTest<JPV_d_1k_100k_1k_20_v5> A("JPV_d_1k_100k_1k_20_v5");
    A.run();

    }
*/


/*
    {
        typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         JORProxVelTestVariant<
            JORProxVelSettings<
                double,
                JORProxVelTestRangeSettings<1000,100000,1000,20>,
                JORProxVelGPUVariantSettings<6>
            >
         >
    > JPV_d_1k_100k_1k_20_v6;

    PerformanceTest<JPV_d_1k_100k_1k_20_v6> A("JPV_d_1k_100k_1k_20_v6");
    A.run();

    }
*/


/*
    {
        typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         JORProxVelTestVariant<
            JORProxVelSettings<
                double,
                JORProxVelTestRangeSettings<1000,100000,1000,20>,
                JORProxVelGPUVariantSettings<7>
            >
         >
    > JPV_d_1k_100k_1k_20_v7;

    PerformanceTest<JPV_d_1k_100k_1k_20_v7> A("JPV_d_1k_100k_1k_20_v7");
    A.run();

    }
*/


}



