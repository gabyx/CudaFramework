
// includes, system
#include <iostream>
#include <stdlib.h>


////////////////////////////////////////////////////////////////////////////////
// Cuda extern C includes
#include "CudaFramework/CudaModern/CudaUtilities.hpp"
#include "CudaFramework/CudaModern/CudaError.hpp"
#include "CudaFramework/Kernels/ProxGPU/ProxGPU.hpp"
#include "CudaFramework/PerformanceTest/PerformanceTest.hpp"
#include "CudaFramework/Kernels/MatrixVectorMultGPU/MatrixVectorMultTestVariant.hpp"
#include "CudaFramework/Kernels/ProxGPU/ProxTestVariant.hpp"
#include "CudaFramework/General/PlatformDefines.hpp"
#include "CudaFramework/Kernels/ProxGPU/ProxKernelSettings.hpp"

using namespace std;
using namespace utilCuda;
using namespace proxGPU;
////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

#define LOG_TO_FILE 0
#define GPU_TO_USE 0

#define COMPILE_ALL_TESTS 1

#if COMPILE_ALL_TESTS == 1

template<int V>
struct RunAllJor{
   static void run(){
      const bool matchGPUCPU = true;

        // Performance Tests JOR, ALL TESTS
        {
           typedef KernelTestMethod<
               KernelTestMethodSettings<false,true,GPU_TO_USE> ,
               ProxTestVariant<
                  ProxSettings<
                     double,
                     SeedGenerator::Fixed<5>,
                     false,
                     10,
                     false,
                     10,
                     matchGPUCPU,
                     ProxPerformanceTestSettings<0,20,4000,1>,
                     JorProxGPUVariantSettings<V,ConvexSets::RPlusAndDisk,true>
                  >
               >
          > test1;
           std::stringstream s;
           s << "JorVariant" << V << "Da";
           PerformanceTest<test1> A(s.str());
          A.run();
        }

         {
           typedef KernelTestMethod<
               KernelTestMethodSettings<false,true,GPU_TO_USE> ,
               ProxTestVariant<
                  ProxSettings<
                     float,
                     SeedGenerator::Fixed<5>,
                     false,
                     10,
                     false,
                     10,
                     matchGPUCPU,
                     ProxPerformanceTestSettings<0,20,4000,1>,
                     JorProxGPUVariantSettings<V,ConvexSets::RPlusAndDisk,true>
                  >
               >
          > test1;
          std::stringstream s;
           s << "JorVariant" << V << "Fa";
           PerformanceTest<test1> A(s.str());
          A.run();
        }

      // run recursive till 0
      RunAllJor<V-1>::run();
      return;
   }
};
template<>
struct RunAllJor<0>{
    static void run(){
       return;
    }
};




template<int V>
struct RunAllSor{
   static void run(){
      const bool matchGPUCPU = true;

        // Performance Tests JOR, ALL TESTS
        {
           typedef KernelTestMethod<
               KernelTestMethodSettings<false,true,GPU_TO_USE> ,
               ProxTestVariant<
                  ProxSettings<
                     double,
                     SeedGenerator::Fixed<5>,
                     false,
                     10,
                     false,
                     10,
                     matchGPUCPU,
                     ProxPerformanceTestSettings<0,20,4000,1>,
                     SorProxGPUVariantSettings<V,ConvexSets::RPlusAndDisk,true>
                  >
               >
          > test1;
           std::stringstream s;
           s << "SorVariant" << V << "Da";
           PerformanceTest<test1> A(s.str());
          A.run();
        }

         {
           typedef KernelTestMethod<
               KernelTestMethodSettings<false,true,GPU_TO_USE> ,
               ProxTestVariant<
                  ProxSettings<
                     float,
                     SeedGenerator::Fixed<5>,
                     false,
                     10,
                     false,
                     10,
                     matchGPUCPU,
                     ProxPerformanceTestSettings<0,20,4000,1>,
                     SorProxGPUVariantSettings<V,ConvexSets::RPlusAndDisk,true>
                  >
               >
          > test1;
          std::stringstream s;
           s << "SorVariant" << V << "Fa";
           PerformanceTest<test1> A(s.str());
          A.run();
        }

      // run recursive till 0
      RunAllSor<V-1>::run();
      return;
   }
};


template<int N>
struct RunAllRelaxedSor{
   static void run(){
      const bool matchGPUCPU = true;
      // RELAXED SOR PROX RECURSION IN ALL Variations
      {
      typedef KernelTestMethod<
            KernelTestMethodSettings<false,true,GPU_TO_USE> ,
            ProxTestVariant<
               ProxSettings<
                  float,
                  SeedGenerator::Fixed<5>,
                  false,
                  10,
                  false,
                  10,
                  matchGPUCPU, //Match CPU to GPU
                  ProxPerformanceTestSettings<0,20,4000,1>,
                  SorProxGPUVariantSettings<5,ConvexSets::RPlusAndDisk,true, RelaxedSorProxKernelSettings<N,ConvexSets::RPlusAndDisk> >
               >
            >
       > test1;
      std::stringstream s;
      s << "SorVariant5Fa" << N;
      PerformanceTest<test1> A(s.str());
      A.run();
      }

      {
      typedef KernelTestMethod<
            KernelTestMethodSettings<false,true,GPU_TO_USE> ,
            ProxTestVariant<
               ProxSettings<
                  double,
                  SeedGenerator::Fixed<5>,
                  false,
                  10,
                  false,
                  10,
                  matchGPUCPU, //Match CPU to GPU
                  ProxPerformanceTestSettings<0,20,4000,1>,
                  SorProxGPUVariantSettings<5,ConvexSets::RPlusAndDisk,true, RelaxedSorProxKernelSettings<N,ConvexSets::RPlusAndDisk> >
               >
            >
       > test1;
      std::stringstream s;
      s << "SorVariant5Da" << N;
      PerformanceTest<test1> A(s.str());
      A.run();
      }

      // run recursive till 0
      RunAllRelaxedSor<N/2>::run();
      return;
   }
};

template<>
struct RunAllRelaxedSor<0>{
    static void run(){
       return;
    }
};

template<>
struct RunAllSor<5>{
    static void run(){

      RunAllRelaxedSor<128>::run();

       // run recursive till 0
      RunAllSor<5-1>::run();
      return;
       return;
    }
};

template<>
struct RunAllSor<0>{
    static void run(){
       return;
    }
};




void performJor(){
   RunAllJor<6>::run();
}

void performSor(){
   RunAllSor<5>::run();
}

#endif

int main(int argc, char** argv)
{
   cout << "[Main started!!]" <<std::endl;

//   typedef KernelTestMethod<
//         KernelTestMethodSettings<false,true,GPU_TO_USE> ,
//         MatrixVectorMultTestVariant<
//            MatrixVectorMultSettings<
//               double,
//               4000,
//               false,
//               1,
//               MatrixVectorMultPerformanceTestSettings<4000,20,4000,1>,
//               MatrixVectorMultGPUVariantSettings<2,true>
//            >
//         >
//    > test1;

//   PerformanceTest<test1> A("test1");
//    A.run();

//{
//  typedef KernelTestMethod<
//         KernelTestMethodSettings<false,true,GPU_TO_USE> ,
//         ProxTestVariant<
//            ProxSettings<
//               double,
//               SeedGenerator::Fixed<5>,
//               false,
//               100,
//               false,
//               10,
//               true,
//               ProxPerformanceTestSettings<20,20,4000,1>,
//               JorProxGPUVariantSettings<5,ConvexSets::RPlusAndDisk,true>
//            >
//         >
//    > test1;
//    PerformanceTest<test1> A("JorProxVariant7D");
//    A.run();
//}
  {
  typedef KernelTestMethod<
         KernelTestMethodSettings<false,true,0> ,
         ProxTestVariant<
            ProxSettings<
               double,
               SeedGenerator::Fixed<5>,
               false,
               1,
               false,
               10,
               true,
               ProxPerformanceTestSettings<10,20,10,1>,
               SorProxGPUVariantSettings<1,ConvexSets::RPlusAndDisk,true>
            >
         >
    > test1;
    PerformanceTest<test1> A("SorProxVariant1D");
    A.run();
   }



   //{
   //typedef KernelTestMethod<
   //      KernelTestMethodSettings<false,true,GPU_TO_USE> ,
   //      ProxTestVariant<
   //         ProxSettings<
   //            double,
   //            SeedGenerator::Fixed<5>,
   //            false,
   //            10,
   //            false,
   //            1,
   //            true, //Match CPU to GPU
   //            ProxPerformanceTestSettings<300,20,320,1>,
   //            SorProxGPUVariantSettings<5,ConvexSets::RPlusAndDisk,true>
   //         >
   //      >
   // > test1;

   // PerformanceTest<test1> A("SorVariant5Da1");
   // A.run();
   //}

//   {
//   typedef KernelTestMethod<
//         KernelTestMethodSettings<false,true,GPU_TO_USE> ,
//         ProxTestVariant<
//            ProxSettings<
//               double,
//               SeedGenerator::Fixed<5>,
//               false,
//               5000,
//               true,
//               1,
//               true, //Match CPU to GPU
//               ProxPerformanceTestSettings<10,4000,4010,1>,
//               JorProxGPUVariantSettings<1,ConvexSets::RPlusAndDisk,true>
//            >
//         >
//    > test1;
//
//    PerformanceTest<test1> A("SorVariant5Fa1");
//    A.run();
//   }

//   {
//      const bool matchGPUCPU = true;
//      const unsigned int N = 32;
//      typedef KernelTestMethod<
//            KernelTestMethodSettings<false,true,GPU_TO_USE> ,
//            ProxTestVariant<
//               ProxSettings<
//                  double,
//                  SeedGenerator::Fixed<5>,
//                  false,
//                  10,
//                  false,
//                  10,
//                  matchGPUCPU, //Match CPU to GPU
//                  ProxPerformanceTestSettings<0,20,4000,1>,
//                  SorProxGPUVariantSettings<5,ConvexSets::RPlusAndDisk,true, RelaxedSorProxKernelSettings<N,ConvexSets::RPlusAndDisk> >
//               >
//            >
//       > test1;
//      std::stringstream s;
//      s << "SorVariant5Da" << N;
//      PerformanceTest<test1> A(s.str());
//      A.run();
//    }

// {
//  typedef KernelTestMethod<
//         KernelTestMethodSettings<false,true,0> ,
//         ProxTestVariant<
//            ProxSettings<
//               double,
//               SeedGenerator::Fixed<5>,
//               false,
//               50,
//               false,
//               10,
//               true,
//               ProxPerformanceTestSettings<4000,20,4000,1>,
//               SorProxGPUVariantSettings<3,ConvexSets::RPlusAndDisk,true>
//            >
//         >
//    > test1;
//    PerformanceTest<test1> A("SorProxVariant5F");
//    A.run();
//   }



//  performJor();
//  performSor();



   HOLD_SYSTEM

}


