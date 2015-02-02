


#include <iostream>
#include <vector>
#include <stdio.h>


#include <chrono>
#include <iostream>

#include "CudaFramework/CudaModern/CudaError.hpp"
#include "CudaFramework/Kernels/JORProxVel/ContactIterationKernel/ContactIterationKernelWrap.hpp"
#include "CudaFramework/Kernels/JORProxVel/ContactIterationKernel/ContactIteration.hpp"


#include "CudaFramework/CudaModern/CudaMatrixUtilities.hpp"


#include "CudaFramework/PerformanceTest/PerformanceTest.hpp"



//#include "CudaFramework/PerformanceTest/PerformanceTest.hpp"
////////////////////////////////////////////////////////////////////////////////
// Cuda extern C includes

#include <boost/variant.hpp>



int main()
{

    typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         ContIterTestVariant<
            ContIterSettings<
                double,
                ContIterTestRangeSettings<50000,50000,1,40>,
                ContIterGPUVariantSettings<8>
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
         ContIterTestVariant<
            ContIterSettings<
                double,
                ContIterTestRangeSettings<1000,100000,1000,20>,
                ContIterGPUVariantSettings<8>
            >
         >
    > CIT_d_1k_100k_1k_20_v8;

    PerformanceTest<CIT_d_1k_100k_1k_20_v8> A("CIT_d_1k_100k_1k_20_v8");
    A.run();

    }
/*
    {
        typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         ContIterTestVariant<
            ContIterSettings<
                double,
                ContIterTestRangeSettings<1000,100000,1000,20>,
                ContIterGPUVariantSettings<1>
            >
         >
    > CIT_d_1k_100k_1k_20_v1;

    PerformanceTest<CIT_d_1k_100k_1k_20_v1> A("CIT_d_1k_100k_1k_20_v1");
    A.run();

    }




    {
        typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         ContIterTestVariant<
            ContIterSettings<
                double,
                ContIterTestRangeSettings<1000,100000,1000,20>,
                ContIterGPUVariantSettings<2>
            >
         >
    > CIT_d_1k_100k_1k_20_v2;

    PerformanceTest<CIT_d_1k_100k_1k_20_v2> A("CIT_d_1k_100k_1k_20_v2");
    A.run();

    }

    {
        typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         ContIterTestVariant<
            ContIterSettings<
                double,
                ContIterTestRangeSettings<1000,100000,1000,20>,
                ContIterGPUVariantSettings<3>
            >
         >
    > CIT_d_1k_100k_1k_20_v3;

    PerformanceTest<CIT_d_1k_100k_1k_20_v3> A("CIT_d_1k_100k_1k_20_v3");
    A.run();

    }

    {
        typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         ContIterTestVariant<
            ContIterSettings<
                double,
                ContIterTestRangeSettings<1000,100000,1000,20>,
                ContIterGPUVariantSettings<4>
            >
         >
    > CIT_d_1k_100k_1k_20_v4;

    PerformanceTest<CIT_d_1k_100k_1k_20_v4> A("CIT_d_1k_100k_1k_20_v4");
    A.run();

    }

    {
        typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         ContIterTestVariant<
            ContIterSettings<
                double,
                ContIterTestRangeSettings<1000,100000,1000,20>,
                ContIterGPUVariantSettings<5>
            >
         >
    > CIT_d_1k_100k_1k_20_v5;

    PerformanceTest<CIT_d_1k_100k_1k_20_v5> A("CIT_d_1k_100k_1k_20_v5");
    A.run();

    }

    {
        typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         ContIterTestVariant<
            ContIterSettings<
                double,
                ContIterTestRangeSettings<1000,100000,1000,20>,
                ContIterGPUVariantSettings<6>
            >
         >
    > CIT_d_1k_100k_1k_20_v6;

    PerformanceTest<CIT_d_1k_100k_1k_20_v6> A("CIT_d_1k_100k_1k_20_v6");
    A.run();

    }

    {
        typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         ContIterTestVariant<
            ContIterSettings<
                double,
                ContIterTestRangeSettings<1000,100000,1000,20>,
                ContIterGPUVariantSettings<7>
            >
         >
    > CIT_d_1k_100k_1k_20_v7;

    PerformanceTest<CIT_d_1k_100k_1k_20_v7> A("CIT_d_1k_100k_1k_20_v7");
    A.run();

    }

    {
        typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         ContIterTestVariant<
            ContIterSettings<
                double,
                ContIterTestRangeSettings<10000,1000000,10000,20>,
                ContIterGPUVariantSettings<1>
            >
         >
    > CIT_d_10k_1000k_10k_20_v1;

    PerformanceTest<CIT_d_10k_1000k_10k_20_v1> A("CIT_d_10k_1000k_10k_20_v1");
    A.run();

    }




    {
        typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         ContIterTestVariant<
            ContIterSettings<
                double,
                ContIterTestRangeSettings<10000,1000000,10000,20>,
                ContIterGPUVariantSettings<2>
            >
         >
    > CIT_d_10k_1000k_10k_20_v2;

    PerformanceTest<CIT_d_10k_1000k_10k_20_v2> A("CIT_d_10k_1000k_10k_20_v2");
    A.run();

    }

    {
        typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         ContIterTestVariant<
            ContIterSettings<
                double,
                ContIterTestRangeSettings<10000,1000000,10000,20>,
                ContIterGPUVariantSettings<3>
            >
         >
    > CIT_d_10k_1000k_10k_20_v3;

    PerformanceTest<CIT_d_10k_1000k_10k_20_v3> A("CIT_d_10k_1000k_10k_20_v3");
    A.run();

    }

    {
        typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         ContIterTestVariant<
            ContIterSettings<
                double,
                ContIterTestRangeSettings<10000,1000000,10000,20>,
                ContIterGPUVariantSettings<4>
            >
         >
    > CIT_d_10k_1000k_10k_20_v4;

    PerformanceTest<CIT_d_10k_1000k_10k_20_v4> A("CIT_d_10k_1000k_10k_20_v4");
    A.run();

    }

    {
        typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         ContIterTestVariant<
            ContIterSettings<
                double,
                ContIterTestRangeSettings<10000,1000000,10000,20>,
                ContIterGPUVariantSettings<5>
            >
         >
    > CIT_d_10k_1000k_10k_20_v5;

    PerformanceTest<CIT_d_10k_1000k_10k_20_v5> A("CIT_d_10k_1000k_10k_20_v5");
    A.run();

    }

    {
        typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         ContIterTestVariant<
            ContIterSettings<
                double,
                ContIterTestRangeSettings<10000,1000000,10000,20>,
                ContIterGPUVariantSettings<6>
            >
         >
    > CIT_d_10k_1000k_10k_20_v6;

    PerformanceTest<CIT_d_10k_1000k_10k_20_v6> A("CIT_d_10k_1000k_10k_20_v6");
    A.run();

    }

    {
        typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         ContIterTestVariant<
            ContIterSettings<
                double,
                ContIterTestRangeSettings<10000,1000000,10000,20>,
                ContIterGPUVariantSettings<7>
            >
         >
    > CIT_d_10k_1000k_10k_20_v7;

    PerformanceTest<CIT_d_10k_1000k_10k_20_v7> A("CIT_d_10k_1000k_10k_20_v7");
    A.run();

    }

*/

}



