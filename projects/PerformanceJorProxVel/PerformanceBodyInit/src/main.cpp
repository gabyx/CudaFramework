


#include <iostream>
#include <vector>
#include <stdio.h>


#include <chrono>
#include <iostream>

#include "CudaFramework/CudaModern/CudaError.hpp"
#include "CudaFramework/Kernels/JORProxVel/BodyInitKernel/BodyInitKernelWrap.hpp"
#include "CudaFramework/Kernels/JORProxVel/BodyInitKernel/BodyInit.hpp"
#include "CudaFramework/CudaModern/CudaMatrixUtilities.hpp"


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
         BodyInitTestVariant<
            BodyInitSettings<
                double,
                BodyInitTestRangeSettings<5000000,5000000,1,3>,
                BodyInitGPUVariantSettings<18>
            >
         >
    > test;

    PerformanceTest<test> A("test");
    A.run();

    std::cout<<"blabla"<<std::endl;



    /*
    typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         BodyInitTestVariant<
            BodyInitSettings<
                double,
                BodyInitTestRangeSettings<1000,1000,1,20>,
                BodyInitGPUVariantSettings<11>
            >
         >
    > TEST;

    PerformanceTest<TEST> A("TEST");
    A.run();
*/
/*
{



    typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         BodyInitTestVariant<
            BodyInitSettings<
                double,
                BodyInitTestRangeSettings<1000,100000,1000,20>,
                BodyInitGPUVariantSettings<11>
            >
         >
    > BIS_d_1k_100k_1k_20_v11;

    PerformanceTest<BIS_d_1k_100k_1k_20_v11> A("BIS_d_1k_100k_1k_20_v11");
    A.run();

    std::cout<<"blabla"<<std::endl;

           typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         BodyInitTestVariant<
            BodyInitSettings<
                double,
                BodyInitTestRangeSettings<1000,100000,1000,20>,
                BodyInitGPUVariantSettings<21>
            >
         >
    > BIS_d_1k_100k_1k_20_v21;

    PerformanceTest<BIS_d_1k_100k_1k_20_v21> B("BIS_d_1k_100k_1k_20_v21");
    B.run();

    typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         BodyInitTestVariant<
            BodyInitSettings<
                double,
                BodyInitTestRangeSettings<1000,100000,1000,20>,
                BodyInitGPUVariantSettings<31>
            >
         >
    > BIS_d_1k_100k_1k_20_v31;

    PerformanceTest<BIS_d_1k_100k_1k_20_v31> A_31("BIS_d_1k_100k_1k_20_v31");
    A_31.run();
    std::cout<<"blabla"<<std::endl;

}
{
    typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         BodyInitTestVariant<
            BodyInitSettings<
                double,
                BodyInitTestRangeSettings<1000,100000,1000,20>,
                BodyInitGPUVariantSettings<12>
            >
         >
    > BIS_d_1k_100k_1k_20_v12;

    PerformanceTest<BIS_d_1k_100k_1k_20_v12> A_1("BIS_d_1k_100k_1k_20_v12");
    A_1.run();
    std::cout<<"blabla"<<std::endl;

    typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         BodyInitTestVariant<
            BodyInitSettings<
                double,
                BodyInitTestRangeSettings<1000,100000,1000,20>,
                BodyInitGPUVariantSettings<22>
            >
         >
    > BIS_d_1k_100k_1k_20_v22;

    PerformanceTest<BIS_d_1k_100k_1k_20_v22> A_2("BIS_d_1k_100k_1k_20_v22");
    A_2.run();

    std::cout<<"blabla"<<std::endl;

           typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         BodyInitTestVariant<
            BodyInitSettings<
                double,
                BodyInitTestRangeSettings<1000,100000,1000,20>,
                BodyInitGPUVariantSettings<32>
            >
         >
    > BIS_d_1k_100k_1k_20_v32;
    PerformanceTest<BIS_d_1k_100k_1k_20_v32> A_3("BIS_d_1k_100k_1k_20_v32");
    A_3.run();

}

{


    typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         BodyInitTestVariant<
            BodyInitSettings<
                double,
                BodyInitTestRangeSettings<1000,100000,1000,20>,
                BodyInitGPUVariantSettings<13>
            >
         >
    > BIS_d_1k_100k_1k_20_v13;

    PerformanceTest<BIS_d_1k_100k_1k_20_v13> A_1("BIS_d_1k_100k_1k_20_v13");
    A_1.run();
    std::cout<<"blabla"<<std::endl;

    typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         BodyInitTestVariant<
            BodyInitSettings<
                double,
                BodyInitTestRangeSettings<1000,100000,1000,20>,
                BodyInitGPUVariantSettings<23>
            >
         >
    > BIS_d_1k_100k_1k_20_v23;

    PerformanceTest<BIS_d_1k_100k_1k_20_v23> A_2("BIS_d_1k_100k_1k_20_v23");
    A_2.run();

    std::cout<<"blabla"<<std::endl;

           typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         BodyInitTestVariant<
            BodyInitSettings<
                double,
                BodyInitTestRangeSettings<1000,100000,1000,20>,
                BodyInitGPUVariantSettings<33>
            >
         >
    > BIS_d_1k_100k_1k_20_v33;
    PerformanceTest<BIS_d_1k_100k_1k_20_v33> A_3("BIS_d_1k_100k_1k_20_v33");
    A_3.run();


}
{
    typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         BodyInitTestVariant<
            BodyInitSettings<
                double,
                BodyInitTestRangeSettings<1000,100000,1000,20>,
                BodyInitGPUVariantSettings<14>
            >
         >
    > BIS_d_1k_100k_1k_20_v14;

    PerformanceTest<BIS_d_1k_100k_1k_20_v14> A_1("BIS_d_1k_100k_1k_20_v14");
    A_1.run();
    std::cout<<"blabla"<<std::endl;

    typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         BodyInitTestVariant<
            BodyInitSettings<
                double,
                BodyInitTestRangeSettings<1000,100000,1000,20>,
                BodyInitGPUVariantSettings<24>
            >
         >
    > BIS_d_1k_100k_1k_20_v24;

    PerformanceTest<BIS_d_1k_100k_1k_20_v24> A_2("BIS_d_1k_100k_1k_20_v24");
    A_2.run();

    std::cout<<"blabla"<<std::endl;

           typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         BodyInitTestVariant<
            BodyInitSettings<
                double,
                BodyInitTestRangeSettings<1000,100000,1000,20>,
                BodyInitGPUVariantSettings<34>
            >
         >
    > BIS_d_1k_100k_1k_20_v34;
    PerformanceTest<BIS_d_1k_100k_1k_20_v34> A_3("BIS_d_1k_100k_1k_20_v34");
    A_3.run();
}

{
    typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         BodyInitTestVariant<
            BodyInitSettings<
                double,
                BodyInitTestRangeSettings<1000,100000,1000,20>,
                BodyInitGPUVariantSettings<15>
            >
         >
    > BIS_d_1k_100k_1k_20_v15;

    PerformanceTest<BIS_d_1k_100k_1k_20_v15> A_1("BIS_d_1k_100k_1k_20_v15");
    A_1.run();
    std::cout<<"blabla"<<std::endl;

    typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         BodyInitTestVariant<
            BodyInitSettings<
                double,
                BodyInitTestRangeSettings<1000,100000,1000,20>,
                BodyInitGPUVariantSettings<25>
            >
         >
    > BIS_d_1k_100k_1k_20_v25;

    PerformanceTest<BIS_d_1k_100k_1k_20_v25> A_2("BIS_d_1k_100k_1k_20_v25");
    A_2.run();

    std::cout<<"blabla"<<std::endl;

           typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         BodyInitTestVariant<
            BodyInitSettings<
                double,
                BodyInitTestRangeSettings<1000,100000,1000,20>,
                BodyInitGPUVariantSettings<35>
            >
         >
    > BIS_d_1k_100k_1k_20_v35;
    PerformanceTest<BIS_d_1k_100k_1k_20_v35> A_3("BIS_d_1k_100k_1k_20_v35");
    A_3.run();
}

{
    typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         BodyInitTestVariant<
            BodyInitSettings<
                double,
                BodyInitTestRangeSettings<1000,100000,1000,20>,
                BodyInitGPUVariantSettings<16>
            >
         >
    > BIS_d_1k_100k_1k_20_v16;

    PerformanceTest<BIS_d_1k_100k_1k_20_v16> A_1("BIS_d_1k_100k_1k_20_v16");
    A_1.run();
    std::cout<<"blabla"<<std::endl;

    typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         BodyInitTestVariant<
            BodyInitSettings<
                double,
                BodyInitTestRangeSettings<1000,100000,1000,20>,
                BodyInitGPUVariantSettings<26>
            >
         >
    > BIS_d_1k_100k_1k_20_v26;

    PerformanceTest<BIS_d_1k_100k_1k_20_v26> A_2("BIS_d_1k_100k_1k_20_v26");
    A_2.run();

    std::cout<<"blabla"<<std::endl;

           typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         BodyInitTestVariant<
            BodyInitSettings<
                double,
                BodyInitTestRangeSettings<1000,100000,1000,20>,
                BodyInitGPUVariantSettings<36>
            >
         >
    > BIS_d_1k_100k_1k_20_v36;
    PerformanceTest<BIS_d_1k_100k_1k_20_v36> A_3("BIS_d_1k_100k_1k_20_v36");
    A_3.run();
}


{
    typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         BodyInitTestVariant<
            BodyInitSettings<
                double,
                BodyInitTestRangeSettings<1000,100000,1000,20>,
                BodyInitGPUVariantSettings<17>
            >
         >
    > BIS_d_1k_100k_1k_20_v17;

    PerformanceTest<BIS_d_1k_100k_1k_20_v17> A_1("BIS_d_1k_100k_1k_20_v17");
    A_1.run();
    std::cout<<"blabla"<<std::endl;

    typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         BodyInitTestVariant<
            BodyInitSettings<
                double,
                BodyInitTestRangeSettings<1000,100000,1000,20>,
                BodyInitGPUVariantSettings<27>
            >
         >
    > BIS_d_1k_100k_1k_20_v27;

    PerformanceTest<BIS_d_1k_100k_1k_20_v27> A_2("BIS_d_1k_100k_1k_20_v27");
    A_2.run();

    std::cout<<"blabla"<<std::endl;

           typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         BodyInitTestVariant<
            BodyInitSettings<
                double,
                BodyInitTestRangeSettings<1000,100000,1000,20>,
                BodyInitGPUVariantSettings<37>
            >
         >
    > BIS_d_1k_100k_1k_20_v37;
    PerformanceTest<BIS_d_1k_100k_1k_20_v37> A_3("BIS_d_1k_100k_1k_20_v37");
    A_3.run();
}
*/
/*
{
    typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         BodyInitTestVariant<
            BodyInitSettings<
                double,
                BodyInitTestRangeSettings<10000,1000000,10000,20>,
                BodyInitGPUVariantSettings<17>
            >
         >
    > BIS_d_10k_1000k_10k_20_v17;

    PerformanceTest<BIS_d_10k_1000k_10k_20_v17> A_1("BIS_d_10k_1000k_10k_20_v17");
    A_1.run();
    std::cout<<"blabla"<<std::endl;

    typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         BodyInitTestVariant<
            BodyInitSettings<
                double,
                BodyInitTestRangeSettings<10000,1000000,10000,20>,
                BodyInitGPUVariantSettings<27>
            >
         >
    > BIS_d_10k_1000k_10k_20_v27;

    PerformanceTest<BIS_d_10k_1000k_10k_20_v27> A_2("BIS_d_10k_1000k_10k_20_v27");
    A_2.run();

    std::cout<<"blabla"<<std::endl;

           typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         BodyInitTestVariant<
            BodyInitSettings<
                double,
                BodyInitTestRangeSettings<10000,1000000,10000,20>,
                BodyInitGPUVariantSettings<37>
            >
         >
    > BIS_d_10k_1000k_10k_20_v37;
    PerformanceTest<BIS_d_10k_1000k_10k_20_v37> A_3("BIS_d_10k_1000k_10k_20_v37");
    A_3.run();


               typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         BodyInitTestVariant<
            BodyInitSettings<
                double,
                BodyInitTestRangeSettings<10000,1000000,10000,20>,
                BodyInitGPUVariantSettings<34>
            >
         >
    > BIS_d_10k_1000k_10k_20_v34;
    PerformanceTest<BIS_d_10k_1000k_10k_20_v34> A_4("BIS_d_10k_1000k_10k_20_v34");
    A_4.run();
}
*/
}








