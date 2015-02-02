


#include <iostream>
#include <vector>
#include <stdio.h>


#include <chrono>
#include <iostream>

#include "ConvergenceCheck.hpp"
#include "ContactIteration.hpp"
#include "ContactInit.hpp"
#include "ReductionTestVariant.hpp"
#include "BodyInit.hpp"

#include "PerformanceTest.hpp"



//#include "PerformanceTest.hpp"
////////////////////////////////////////////////////////////////////////////////
// Cuda extern C includes

#include <boost/variant.hpp>



int main()
{

   {

 typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         BodyInitTestVariant<
            BodyInitSettings<
                double,
                BodyInitTestRangeSettings<10000,1000000,10000,20>,
                BodyInitGPUVariantSettings<18>
            >
         >
    > BIS_d_10k_1000k_10k_20_v18;

    PerformanceTest<BIS_d_10k_1000k_10k_20_v18> A("BIS_d_10k_1000k_10k_20_v18");
    A.run();
   }
   {
    std::cout<<"blabla"<<std::endl;

    typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         BodyInitTestVariant<
            BodyInitSettings<
                double,
                BodyInitTestRangeSettings<10000,1000000,10000,20>,
                BodyInitGPUVariantSettings<11>
            >
         >
    > BIS_d_10k_1000k_10k_20_v11;

    PerformanceTest<BIS_d_10k_1000k_10k_20_v11> A("BIS_d_10k_1000k_10k_20_v11");
    A.run();

    std::cout<<"blabla"<<std::endl;

           typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         BodyInitTestVariant<
            BodyInitSettings<
                double,
                BodyInitTestRangeSettings<10000,1000000,10000,20>,
                BodyInitGPUVariantSettings<21>
            >
         >
    > BIS_d_10k_1000k_10k_20_v21;

    PerformanceTest<BIS_d_10k_1000k_10k_20_v21> B("BIS_d_10k_1000k_10k_20_v21");
    B.run();

    typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         BodyInitTestVariant<
            BodyInitSettings<
                double,
                BodyInitTestRangeSettings<10000,1000000,10000,20>,
                BodyInitGPUVariantSettings<31>
            >
         >
    > BIS_d_10k_1000k_10k_20_v31;

    PerformanceTest<BIS_d_10k_1000k_10k_20_v31> A_31("BIS_d_10k_1000k_10k_20_v31");
    A_31.run();
    std::cout<<"blabla"<<std::endl;

}
{
    typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         BodyInitTestVariant<
            BodyInitSettings<
                double,
                BodyInitTestRangeSettings<10000,1000000,10000,20>,
                BodyInitGPUVariantSettings<12>
            >
         >
    > BIS_d_10k_1000k_10k_20_v12;

    PerformanceTest<BIS_d_10k_1000k_10k_20_v12> A_1("BIS_d_10k_1000k_10k_20_v12");
    A_1.run();
    std::cout<<"blabla"<<std::endl;

    typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         BodyInitTestVariant<
            BodyInitSettings<
                double,
                BodyInitTestRangeSettings<10000,1000000,10000,20>,
                BodyInitGPUVariantSettings<22>
            >
         >
    > BIS_d_10k_1000k_10k_20_v22;

    PerformanceTest<BIS_d_10k_1000k_10k_20_v22> A_2("BIS_d_10k_1000k_10k_20_v22");
    A_2.run();

    std::cout<<"blabla"<<std::endl;

           typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         BodyInitTestVariant<
            BodyInitSettings<
                double,
                BodyInitTestRangeSettings<10000,1000000,10000,20>,
                BodyInitGPUVariantSettings<32>
            >
         >
    > BIS_d_10k_1000k_10k_20_v32;
    PerformanceTest<BIS_d_10k_1000k_10k_20_v32> A_3("BIS_d_10k_1000k_10k_20_v32");
    A_3.run();

}

{


    typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         BodyInitTestVariant<
            BodyInitSettings<
                double,
                BodyInitTestRangeSettings<10000,1000000,10000,20>,
                BodyInitGPUVariantSettings<13>
            >
         >
    > BIS_d_10k_1000k_10k_20_v13;

    PerformanceTest<BIS_d_10k_1000k_10k_20_v13> A_1("BIS_d_10k_1000k_10k_20_v13");
    A_1.run();
    std::cout<<"blabla"<<std::endl;

    typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         BodyInitTestVariant<
            BodyInitSettings<
                double,
                BodyInitTestRangeSettings<10000,1000000,10000,20>,
                BodyInitGPUVariantSettings<23>
            >
         >
    > BIS_d_10k_1000k_10k_20_v23;

    PerformanceTest<BIS_d_10k_1000k_10k_20_v23> A_2("BIS_d_10k_1000k_10k_20_v23");
    A_2.run();

    std::cout<<"blabla"<<std::endl;

           typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         BodyInitTestVariant<
            BodyInitSettings<
                double,
                BodyInitTestRangeSettings<10000,1000000,10000,20>,
                BodyInitGPUVariantSettings<33>
            >
         >
    > BIS_d_10k_1000k_10k_20_v33;
    PerformanceTest<BIS_d_10k_1000k_10k_20_v33> A_3("BIS_d_10k_1000k_10k_20_v33");
    A_3.run();


}
{
    typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         BodyInitTestVariant<
            BodyInitSettings<
                double,
                BodyInitTestRangeSettings<10000,1000000,10000,20>,
                BodyInitGPUVariantSettings<14>
            >
         >
    > BIS_d_10k_1000k_10k_20_v14;

    PerformanceTest<BIS_d_10k_1000k_10k_20_v14> A_1("BIS_d_10k_1000k_10k_20_v14");
    A_1.run();
    std::cout<<"blabla"<<std::endl;

    typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         BodyInitTestVariant<
            BodyInitSettings<
                double,
                BodyInitTestRangeSettings<10000,1000000,10000,20>,
                BodyInitGPUVariantSettings<24>
            >
         >
    > BIS_d_10k_1000k_10k_20_v24;

    PerformanceTest<BIS_d_10k_1000k_10k_20_v24> A_2("BIS_d_10k_1000k_10k_20_v24");
    A_2.run();

    std::cout<<"blabla"<<std::endl;

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
    PerformanceTest<BIS_d_10k_1000k_10k_20_v34> A_3("BIS_d_10k_1000k_10k_20_v34");
    A_3.run();
}

{
    typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         BodyInitTestVariant<
            BodyInitSettings<
                double,
                BodyInitTestRangeSettings<10000,1000000,10000,20>,
                BodyInitGPUVariantSettings<15>
            >
         >
    > BIS_d_10k_1000k_10k_20_v15;

    PerformanceTest<BIS_d_10k_1000k_10k_20_v15> A_1("BIS_d_10k_1000k_10k_20_v15");
    A_1.run();
    std::cout<<"blabla"<<std::endl;

    typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         BodyInitTestVariant<
            BodyInitSettings<
                double,
                BodyInitTestRangeSettings<10000,1000000,10000,20>,
                BodyInitGPUVariantSettings<25>
            >
         >
    > BIS_d_10k_1000k_10k_20_v25;

    PerformanceTest<BIS_d_10k_1000k_10k_20_v25> A_2("BIS_d_10k_1000k_10k_20_v25");
    A_2.run();

    std::cout<<"blabla"<<std::endl;

           typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         BodyInitTestVariant<
            BodyInitSettings<
                double,
                BodyInitTestRangeSettings<10000,1000000,10000,20>,
                BodyInitGPUVariantSettings<35>
            >
         >
    > BIS_d_1k_100k_1k_20_v35;
    PerformanceTest<BIS_d_1k_100k_1k_20_v35> A_3("BIS_d_10k_1000k_10k_20_v35");
    A_3.run();
}

{
    typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         BodyInitTestVariant<
            BodyInitSettings<
                double,
                BodyInitTestRangeSettings<10000,1000000,10000,20>,
                BodyInitGPUVariantSettings<16>
            >
         >
    > BIS_d_10k_1000k_10k_20_v16;

    PerformanceTest<BIS_d_10k_1000k_10k_20_v16> A_1("BIS_d_10k_1000k_10k_20_v16");
    A_1.run();
    std::cout<<"blabla"<<std::endl;

    typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         BodyInitTestVariant<
            BodyInitSettings<
                double,
                BodyInitTestRangeSettings<10000,1000000,10000,20>,
                BodyInitGPUVariantSettings<26>
            >
         >
    > BIS_d_10k_1000k_10k_20_v26;

    PerformanceTest<BIS_d_10k_1000k_10k_20_v26> A_2("BIS_d_10k_1000k_10k_20_v26");
    A_2.run();

    std::cout<<"blabla"<<std::endl;

           typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         BodyInitTestVariant<
            BodyInitSettings<
                double,
                BodyInitTestRangeSettings<10000,1000000,10000,20>,
                BodyInitGPUVariantSettings<36>
            >
         >
    > BIS_d_10k_1000k_10k_20_v36;
    PerformanceTest<BIS_d_10k_1000k_10k_20_v36> A_3("BIS_d_10k_1000k_10k_20_v36");
    A_3.run();
}


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
}

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

        {
        typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         ContactInitTestVariant<
            ContactInitSettings<
                double,
                ContactInitTestRangeSettings<10000,1000000,10000,20>,
                ContactInitGPUVariantSettings<8>
            >
         >
    > CIN_d_10k_1000k_10k_20_v8;

    PerformanceTest<CIN_d_10k_1000k_10k_20_v8> A("CIN_d_10k_1000k_10k_20_v8");
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

        {
        typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         ContIterTestVariant<
            ContIterSettings<
                double,
                ContIterTestRangeSettings<10000,1000000,10000,20>,
                ContIterGPUVariantSettings<8>
            >
         >
    > CIT_d_10k_1000k_10k_20_v8;

    PerformanceTest<CIT_d_10k_1000k_10k_20_v8> A("CIT_d_10k_1000k_10k_20_v8");
    A.run();

    }


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
        {
        typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         ConvCheckTestVariant<
            ConvCheckSettings<
                double,
                ConvCheckTestRangeSettings<10000,1000000,10000,20>,
                ConvCheckGPUVariantSettings<8>
            >
         >
    > CCK_d_10k_1000k_10k_20_v8;

    PerformanceTest<CCK_d_10k_1000k_10k_20_v8> A("CCK_d_10k_1000k_10k_20_v8");
    A.run();

    }


    {
        typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         ConvCheckTestVariant<
            ConvCheckSettings<
                double,
                ConvCheckTestRangeSettings<10000,1000000,10000,20>,
                ConvCheckGPUVariantSettings<1>
            >
         >
    > CCK_d_10k_1000k_10k_20_v1;

    PerformanceTest<CCK_d_10k_1000k_10k_20_v1> A("CCK_d_10k_1000k_10k_20_v1");
    A.run();

    }




    {
        typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         ConvCheckTestVariant<
            ConvCheckSettings<
                double,
                ConvCheckTestRangeSettings<10000,1000000,10000,20>,
                ConvCheckGPUVariantSettings<2>
            >
         >
    > CCK_d_10k_1000k_10k_20_v2;

    PerformanceTest<CCK_d_10k_1000k_10k_20_v2> A("CCK_d_10k_1000k_10k_20_v2");
    A.run();

    }




    {
        typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         ConvCheckTestVariant<
            ConvCheckSettings<
                double,
                ConvCheckTestRangeSettings<10000,1000000,10000,20>,
                ConvCheckGPUVariantSettings<3>
            >
         >
    > CCK_d_10k_1000k_10k_20_v3;

    PerformanceTest<CCK_d_10k_1000k_10k_20_v3> A("CCK_d_10k_1000k_10k_20_v3");
    A.run();

    }




    {
        typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         ConvCheckTestVariant<
            ConvCheckSettings<
                double,
                ConvCheckTestRangeSettings<10000,1000000,10000,20>,
                ConvCheckGPUVariantSettings<4>
            >
         >
    > CCK_d_10k_1000k_10k_20_v4;

    PerformanceTest<CCK_d_10k_1000k_10k_20_v4> A("CCK_d_10k_1000k_10k_20_v4");
    A.run();

    }




    {
        typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         ConvCheckTestVariant<
            ConvCheckSettings<
                double,
                ConvCheckTestRangeSettings<10000,1000000,10000,20>,
                ConvCheckGPUVariantSettings<5>
            >
         >
    > CCK_d_10k_1000k_10k_20_v5;

    PerformanceTest<CCK_d_10k_1000k_10k_20_v5> A("CCK_d_10k_1000k_10k_20_v5");
    A.run();

    }




    {
        typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         ConvCheckTestVariant<
            ConvCheckSettings<
                double,
                ConvCheckTestRangeSettings<10000,1000000,10000,20>,
                ConvCheckGPUVariantSettings<6>
            >
         >
    > CCK_d_10k_1000k_10k_20_v6;

    PerformanceTest<CCK_d_10k_1000k_10k_20_v6> A("CCK_d_10k_1000k_10k_20_v6");
    A.run();

    }




    {
        typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         ConvCheckTestVariant<
            ConvCheckSettings<
                double,
                ConvCheckTestRangeSettings<10000,1000000,10000,20>,
                ConvCheckGPUVariantSettings<7>
            >
         >
    > CCK_d_10k_1000k_10k_20_v7;

    PerformanceTest<CCK_d_10k_1000k_10k_20_v7> A("CCK_d_10k_1000k_10k_20_v7");
    A.run();

    }

}



