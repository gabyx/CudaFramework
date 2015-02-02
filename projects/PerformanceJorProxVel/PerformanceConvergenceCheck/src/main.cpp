


#include <iostream>
#include <vector>
#include <stdio.h>


#include <chrono>
#include <iostream>

#include "CudaFramework/Kernels/JORProxVel/ConvergenceCheckKernel/ConvergenceCheck.hpp"
//#include "CudaFramework/Kernels/JORProxVel/ContactIterationKernel/ContactIteration.hpp"

#include "CudaFramework/PerformanceTest/PerformanceTest.hpp"



//#include "CudaFramework/PerformanceTest/PerformanceTest.hpp"
////////////////////////////////////////////////////////////////////////////////
// Cuda extern C includes

#include <boost/variant.hpp>



int main()
{
    typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         ConvCheckTestVariant<
            ConvCheckSettings<
                double,
                ConvCheckTestRangeSettings<3,3,1,1>,
                ConvCheckGPUVariantSettings<1>
            >
         >
    > test1;

    std::cout<<"blabla"<<std::endl;
    PerformanceTest<test1> A("test1");
    A.run();
    std::cout<<"blabla"<<std::endl;


        {
        typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         ConvCheckTestVariant<
            ConvCheckSettings<
                double,
                ConvCheckTestRangeSettings<1000,100000,1000,20>,
                ConvCheckGPUVariantSettings<8>
            >
         >
    > CCK_d_1k_100k_1k_20_v8;

    PerformanceTest<CCK_d_1k_100k_1k_20_v8> A("CCK_d_1k_100k_1k_20_v8");
    A.run();

    }

/*
    {
        typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         ConvCheckTestVariant<
            ConvCheckSettings<
                double,
                ConvCheckTestRangeSettings<1000,100000,1000,20>,
                ConvCheckGPUVariantSettings<1>
            >
         >
    > CCK_d_1k_100k_1k_20_v1;

    PerformanceTest<CCK_d_1k_100k_1k_20_v1> A("CCK_d_1k_100k_1k_20_v1");
    A.run();

    }




    {
        typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         ConvCheckTestVariant<
            ConvCheckSettings<
                double,
                ConvCheckTestRangeSettings<1000,100000,1000,20>,
                ConvCheckGPUVariantSettings<2>
            >
         >
    > CCK_d_1k_100k_1k_20_v2;

    PerformanceTest<CCK_d_1k_100k_1k_20_v2> A("CCK_d_1k_100k_1k_20_v2");
    A.run();

    }




    {
        typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         ConvCheckTestVariant<
            ConvCheckSettings<
                double,
                ConvCheckTestRangeSettings<1000,100000,1000,20>,
                ConvCheckGPUVariantSettings<3>
            >
         >
    > CCK_d_1k_100k_1k_20_v3;

    PerformanceTest<CCK_d_1k_100k_1k_20_v3> A("CCK_d_1k_100k_1k_20_v3");
    A.run();

    }




    {
        typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         ConvCheckTestVariant<
            ConvCheckSettings<
                double,
                ConvCheckTestRangeSettings<1000,100000,1000,20>,
                ConvCheckGPUVariantSettings<4>
            >
         >
    > CCK_d_1k_100k_1k_20_v4;

    PerformanceTest<CCK_d_1k_100k_1k_20_v4> A("CCK_d_1k_100k_1k_20_v4");
    A.run();

    }




    {
        typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         ConvCheckTestVariant<
            ConvCheckSettings<
                double,
                ConvCheckTestRangeSettings<1000,100000,1000,20>,
                ConvCheckGPUVariantSettings<5>
            >
         >
    > CCK_d_1k_100k_1k_20_v5;

    PerformanceTest<CCK_d_1k_100k_1k_20_v5> A("CCK_d_1k_100k_1k_20_v5");
    A.run();

    }




    {
        typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         ConvCheckTestVariant<
            ConvCheckSettings<
                double,
                ConvCheckTestRangeSettings<1000,100000,1000,20>,
                ConvCheckGPUVariantSettings<6>
            >
         >
    > CCK_d_1k_100k_1k_20_v6;

    PerformanceTest<CCK_d_1k_100k_1k_20_v6> A("CCK_d_1k_100k_1k_20_v6");
    A.run();

    }




    {
        typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         ConvCheckTestVariant<
            ConvCheckSettings<
                double,
                ConvCheckTestRangeSettings<1000,100000,1000,20>,
                ConvCheckGPUVariantSettings<7>
            >
         >
    > CCK_d_1k_100k_1k_20_v7;

    PerformanceTest<CCK_d_1k_100k_1k_20_v7> A("CCK_d_1k_100k_1k_20_v7");
    A.run();

    }
*/


}



