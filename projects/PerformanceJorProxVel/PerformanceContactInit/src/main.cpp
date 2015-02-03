


#include <iostream>
#include <vector>
#include <stdio.h>
#include <chrono>
#include <iostream>
#include "CudaFramework/CudaModern/CudaContext.hpp"
#include "CudaFramework/CudaModern/CudaError.hpp"
#include "CudaFramework/Kernels/JORProxVel/ContactInitKernel/ContactInit.hpp"
#include "CudaFramework/PerformanceTest/PerformanceTest.hpp"



//#include "CudaFramework/PerformanceTest/PerformanceTest.hpp"
////////////////////////////////////////////////////////////////////////////////
// Cuda extern C includes

#include <boost/variant.hpp>



int main()
{

    typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         ContactInitTestVariant<
            ContactInitSettings<
                double,
                ContactInitTestRangeSettings<500000,500000,1,1>,
                ContactInitGPUVariantSettings<8>
            >
         >
    > test1;

    std::cout<<"blabla"<<std::endl;
    PerformanceTest<test1> A("test1");
    A.run();
    std::cout<<"blabla"<<std::endl;


/*
    typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         ContactInitTestVariant<
            ContactInitSettings<
                double,
                ContactInitTestRangeSettings<2,2,1,1>,
                ContactInitGPUVariantSettings<1>
            >
         >
    > test1;

    std::cout<<"blabla"<<std::endl;
    PerformanceTest<test1> A("test1");
    A.run();
    std::cout<<"blabla"<<std::endl;
    */
{
        typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         ContactInitTestVariant<
            ContactInitSettings<
                double,
                ContactInitTestRangeSettings<1000,100000,1000,20>,
                ContactInitGPUVariantSettings<8>
            >
         >
    > CIN_d_1k_100k_1k_20_v8;

    PerformanceTest<CIN_d_1k_100k_1k_20_v8> A("CIN_d_1k_100k_1k_20_v8");
    A.run();

    }
/*
    {
        typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         ContactInitTestVariant<
            ContactInitSettings<
                double,
                ContactInitTestRangeSettings<1000,100000,1000,20>,
                ContactInitGPUVariantSettings<1>
            >
         >
    > CIN_d_1k_100k_1k_20_v1;

    PerformanceTest<CIN_d_1k_100k_1k_20_v1> A("CIN_d_1k_100k_1k_20_v1");
    A.run();

    }
        {
        typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         ContactInitTestVariant<
            ContactInitSettings<
                double,
                ContactInitTestRangeSettings<1000,100000,1000,20>,
                ContactInitGPUVariantSettings<2>
            >
         >
    > CIN_d_1k_100k_1k_20_v2;

    PerformanceTest<CIN_d_1k_100k_1k_20_v2> A("CIN_d_1k_100k_1k_20_v2");
    A.run();

    }
        {
        typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         ContactInitTestVariant<
            ContactInitSettings<
                double,
                ContactInitTestRangeSettings<1000,100000,1000,20>,
                ContactInitGPUVariantSettings<3>
            >
         >
    > CIN_d_1k_100k_1k_20_v3;

    PerformanceTest<CIN_d_1k_100k_1k_20_v3> A("CIN_d_1k_100k_1k_20_v3");
    A.run();

    }
        {
        typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         ContactInitTestVariant<
            ContactInitSettings<
                double,
                ContactInitTestRangeSettings<1000,100000,1000,20>,
                ContactInitGPUVariantSettings<4>
            >
         >
    > CIN_d_1k_100k_1k_20_v4;

    PerformanceTest<CIN_d_1k_100k_1k_20_v4> A("CIN_d_1k_100k_1k_20_v4");
    A.run();

    }
        {
        typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         ContactInitTestVariant<
            ContactInitSettings<
                double,
                ContactInitTestRangeSettings<1000,100000,1000,20>,
                ContactInitGPUVariantSettings<5>
            >
         >
    > CIN_d_1k_100k_1k_20_v5;

    PerformanceTest<CIN_d_1k_100k_1k_20_v5> A("CIN_d_1k_100k_1k_20_v5");
    A.run();

    }

*/
/*
    {
        typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         ContactInitTestVariant<
            ContactInitSettings<
                double,
                ContactInitTestRangeSettings<10000,1000000,10000,20>,
                ContactInitGPUVariantSettings<1>
            >
         >
    > CIN_d_10k_1000k_10k_20_v1;

    PerformanceTest<CIN_d_10k_1000k_10k_20_v1> A("CIN_d_10k_1000k_10k_20_v1");
    A.run();

    }
        {
        typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         ContactInitTestVariant<
            ContactInitSettings<
                double,
                ContactInitTestRangeSettings<10000,1000000,10000,20>,
                ContactInitGPUVariantSettings<2>
            >
         >
    > CIN_d_10k_1000k_10k_20_v2;

    PerformanceTest<CIN_d_10k_1000k_10k_20_v2> A("CIN_d_10k_1000k_10k_20_v2");
    A.run();

    }
        {
        typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         ContactInitTestVariant<
            ContactInitSettings<
                double,
                ContactInitTestRangeSettings<10000,1000000,10000,20>,
                ContactInitGPUVariantSettings<3>
            >
         >
    > CIN_d_10k_1000k_10k_20_v3;

    PerformanceTest<CIN_d_10k_1000k_10k_20_v3> A("CIN_d_10k_1000k_10k_20_v3");
    A.run();

    }
        {
        typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         ContactInitTestVariant<
            ContactInitSettings<
                double,
                ContactInitTestRangeSettings<10000,1000000,10000,20>,
                ContactInitGPUVariantSettings<4>
            >
         >
    > CIN_d_10k_1000k_10k_20_v4;

    PerformanceTest<CIN_d_10k_1000k_10k_20_v4> A("CIN_d_10k_1000k_10k_20_v4");
    A.run();

    }
        {
        typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         ContactInitTestVariant<
            ContactInitSettings<
                double,
                ContactInitTestRangeSettings<10000,1000000,10000,20>,
                ContactInitGPUVariantSettings<5>
            >
         >
    > CIN_d_10k_1000k_10k_20_v5;

    PerformanceTest<CIN_d_10k_1000k_10k_20_v5> A("CIN_d_10k_1000k_10k_20_v5");
    A.run();

    }
    */
}



