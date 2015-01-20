// ========================================================================================
//  CudaFramework
//  Copyright (C) 2014 by Gabriel Nützi <nuetzig (at) imes (d0t) mavt (d0t) ethz (d0t) ch>
//
//  This Source Code Form is subject to the terms of the GNU GPL 3.0 licence.
//  If a copy of the GNU GPL 3.0 was not distributed with this
//  file, you can obtain one at http://opensource.org/licenses/GPL-3.0.
// ========================================================================================

#ifndef CudaFramework_Kernels_ProxGPU_JorProxGPUVariant_hpp
#define CudaFramework_Kernels_ProxGPU_JorProxGPUVariant_hpp

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include <algorithm>

#include <Eigen/Dense>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <tinyformat/TinyFormatInclude.hpp>

#include "CudaFramework/General/CPUTimer.hpp"
#include "CudaFramework/General/StaticAssert.hpp"
#include "CudaFramework/General/TypeTraitsHelper.hpp"
#include "CudaFramework/CudaModern/CudaError.hpp"
#include "CudaFramework/General/Utilities.hpp"
#include "CudaFramework/CudaModern/CudaMatrix.hpp"
#include "CudaFramework/CudaModern/CudaMatrixUtilities.hpp"


#include "CudaFramework/Kernels/ProxGPU/ProxSettings.hpp"
#include "CudaFramework/Kernels/ProxGPU/ProxGPU.hpp"
#include "CudaFramework/Kernels/ProxGPU/ProxKernelSettings.hpp"


#include "ConvexSets.hpp"

/**
* @addtogroup ProxTestVariant
* @defgroup JorProxGPUVariant Jor Prox GPUVariants
* @detailed VariantId specifies which variant is launched:
* Here the different variants have been included in one class!
* To be more flexible we can also completely reimplement the whole class for another GPUVariant!
* @{
*/

using namespace TypeTraitsHelper;


template<typename TJorProxGPUVariantSettingsWrapper>
class JorProxGPUVariant<TJorProxGPUVariantSettingsWrapper,ConvexSets::RPlusAndDisk> {
public:

   DEFINE_JorProxGPUVariant_SettingsWrapper(TJorProxGPUVariantSettingsWrapper);

   JorProxGPUVariant(): m_nMaxIterations(nMaxIterations), pConvergedFlag_dev(NULL){

   }
   ~JorProxGPUVariant(){

   }

   void setSettings(unsigned int iterations, PREC absTol, PREC relTol){
      m_nMaxIterations = iterations;
      m_absTOL = absTol;
      m_relTOL = relTol;
   }


   typedef typename ManualOrDefault<IsEqual<VariantId,3>::result,TKernelSettings,JorProxKernelSettings<128,126,4,TConvexSet,6> >::TValue JorProxSettings3;

   typedef typename ManualOrDefault<IsEqual<VariantId,4>::result,TKernelSettings,JorProxKernelSettings<192,192,4,TConvexSet,6> >::TValue JorProxSettings4;

    /** Check settings at runtime, static settings are already checked at compile time*/
    bool checkSettings(int gpuID){
    switch(VariantId){
      case 1:
         return true;
      case 2:
         return true;
      case 3:
         return JorProxSettings3::checkSettings(gpuID);
      case 4:
         return JorProxSettings4::checkSettings(gpuID);
      case 5:
         return true;
      case 6:
        return true;
      case 7:
        return true;
      default:
         ERRORMSG("No settings check specified for variant: " << VariantId << std::endl)
         return false;
      }
   }

   static std::string getVariantName(){
       std::stringstream s;
      switch(VariantId){
      case 1:
         s << "JOR";
         break;
      case 2:
         s << "JOR";
          break;
      case 3:
         s << "JOR";
          break;
      case 4:
         s << "JOR";
          break;
      case 5:
         s << "JOR";
          break;
      case 6:
         s << "JOR";
          break;
      case 7:
         s << "CPU JOR";
          break;
      }
      s << ((bAbortIfConverged)? std::string("[Convergence Check]") : std::string(""));

      return s.str();
   }

   static std::string getVariantDescriptionShort(){
       std::stringstream s;
      switch(VariantId){
      case 1:
         s << "[cuBlas Multiplication][Addition][Prox]";
         break;
      case 2:
         s << "[cuBlas Multiplication)][Prox + Addition]";
          break;
      case 3:
         s<< "[Multiplication + Addition + Prox]";
          break;
      case 4:
         s<< "[Multiplication + Addition + Prox]";
          break;
      case 5:
         s<< "[Multiplication + Addition][Prox]";
          break;
      case 6:
         s<< "[Prox]";
          break;
      case 7:

      #if USE_GOTO_BLAS ==1
       s<< "[GotoBLAS Multiplication][Eigen Addition + Prox]";
      #else
      #if USE_INTEL_BLAS ==1
        s<< "[Intel MKL Multiplication][Eigen Addition + Prox]";
      #endif
      #endif
          break;
      }

      if(VariantId !=7){
        s << ((bAbortIfConverged)? std::string("[Convergence Check]") : std::string(""));
      }

      return s.str();
   }


   static std::string getVariantDescriptionLong(){
      std::stringstream s;
      switch(VariantId){
      case 1:
         s<< "[gemv routine][ThreadsPerBlock : 192][ThreadsPerBlock : 128]";
         break;
      case 2:
         s<< "[gemv routine][ThreadsPerBlock : 256]";
         break;
      case 3:
         s << JORPROXKERNEL_SETTINGS_STR(JorProxSettings3);
        break;
      case 4:
         s << JORPROXKERNEL_SETTINGS_STR(JorProxSettings4);
         break;
      case 5:
         s<< "[ThreadsPerBlock : 128, BlockDim : 128, XElementsPerThread : 4,  UnrollBlockDotProduct : 6][ThreadsPerBlock : 128]";
         break;
      case 6:
         s<< "[ThreadsPerBlock : 128]";
         break;
      case 7:
         s<< "[6 CPU Threads][]";
         break;
      }
      return s.str() + std::string(", Matrix T aligned: ") + ((alignMatrix)? std::string("on") : std::string("off"));
   }


   double getNOps(){
      if(VariantId<6 || VariantId==7){
       return evaluateProxTermSOR_FLOPS(T_dev.m_M,T_dev.m_N) + proxContactOrdered_RPlusAndDisk_1threads_kernel_FLOPS(T_dev.m_M / TConvexSet::Dimension);
      }
      else if(VariantId==6){
       return proxContactOrdered_RPlusAndDisk_1threads_kernel_FLOPS(T_dev.m_M / TConvexSet::Dimension);
      }
   }

   double getBytesReadWrite(){
      switch(VariantId){
      case 1 || 7:
         return /*A*x+b = y*/ T_dev.m_M*T_dev.m_N + 3*T_dev.m_M  /* Prox */ + T_dev.m_M +  T_dev.m_M + /* convergence */ + ((bAbortIfConverged)? 2*T_dev.m_M : 0);
      case 2:
         return /*A*x = y*/ T_dev.m_M*T_dev.m_N + 2*T_dev.m_M + /* Prox */ + T_dev.m_M +  T_dev.m_M +  T_dev.m_M +/* convergence */ + ((bAbortIfConverged)? 2*T_dev.m_M : 0);
      case 3:
         return /*A*x+b = y*/ T_dev.m_M*T_dev.m_N + 3*T_dev.m_M   + /* convergence */ + ((bAbortIfConverged)? 2*T_dev.m_M : 0);
      case 4:
         return /*A*x+b = y*/ T_dev.m_M*T_dev.m_N + 3*T_dev.m_M   + /* convergence */ + ((bAbortIfConverged)? 2*T_dev.m_M : 0);
      case 5:
         return /*A*x+b = y*/ T_dev.m_M*T_dev.m_N + 3*T_dev.m_M  /* Prox */ + T_dev.m_M +  T_dev.m_M + /* convergence */ + ((bAbortIfConverged)? 2*T_dev.m_M : 0);
      case 6:
         return /* Prox */ + T_dev.m_M +  T_dev.m_M ;
      }
   }

   unsigned int getTradeoff(){
      return 180;
   }

   void initialize(std::ostream* pLog, std::ofstream* pMatlab_file){
      m_pLog = pLog;
      m_pMatlab_file = pMatlab_file;
   }

   void finalize(){
   }

   void setDeviceToUse(int device){
      CHECK_CUDA(cudaSetDevice(device));
   }

   template<typename Derived1, typename Derived2>
   void initializeTestProblem( const Eigen::MatrixBase<Derived2> &T, const Eigen::MatrixBase<Derived1> & x_old,const Eigen::MatrixBase<Derived1> & d){

      CHECK_CUBLAS(cublasCreate(&m_cublasHandle));
      CHECK_CUDA(cudaEventCreate(&m_startKernel));
      CHECK_CUDA(cudaEventCreate(&m_stopKernel));
      CHECK_CUDA(cudaEventCreate(&m_startCopy));
      CHECK_CUDA(cudaEventCreate(&m_stopCopy));


      if(VariantId != 7){
         size_t freeMem;
         size_t totalMem;
         cudaError_t error;

         CHECK_CUDA(cudaMemGetInfo (&freeMem, &totalMem));

         size_t nGPUBytes = (2*x_old.rows() + d.rows() + x_old.rows()/TConvexSet::Dimension +  T.rows()*T.cols())*sizeof(PREC);
         *m_pLog << "Will try to allocate ("<<nGPUBytes<<"/"<<freeMem<<") = " << (double)nGPUBytes/freeMem * 100.0 <<" % of global memory on GPU"<<std::endl;
         if(nGPUBytes > freeMem){
            *m_pLog <<"Probably to little memory on GPU, try anway!..."<<std::endl;
         }


         CHECK_CUDA(utilCuda::releaseAndMallocMatrixDevice<alignMatrix>(T_dev, T));
         CHECK_CUDA(utilCuda::releaseAndMallocMatrixDevice<false>(d_dev, d));
         CHECK_CUDA(utilCuda::releaseAndMallocMatrixDevice<false>(x_old_dev,x_old));

         if(TypeTraitsHelper::IsSame<TConvexSet,ConvexSets::RPlusAndDisk>::result){
            ASSERTMSG(x_old.rows()%TConvexSet::Dimension==0,"Wrong Dimension!");
            CHECK_CUDA(utilCuda::releaseAndMallocMatrixDevice<false>(mu_dev, x_old.rows()/TConvexSet::Dimension, x_old.cols()));
         }
         else if(TypeTraitsHelper::IsSame<TConvexSet,ConvexSets::RPlusAndContensouEllipsoid>::result){
            //TODO malloc more! mu and R vector
         }

         utilCuda::releaseAndMallocMatrixDevice<false>(x_new_dev,x_old_dev.m_M,x_old_dev.m_N);

         if(pConvergedFlag_dev){
            CHECK_CUDA(cudaFree(pConvergedFlag_dev));
            pConvergedFlag_dev = NULL;
         }

         CHECK_CUDA(cudaMalloc(&pConvergedFlag_dev,sizeof(bool)));
      }
      else{

         // Use same utilCuda::CudaMatrix strut for the mklBlas stuff, but only make references
         // but already in runGPUProfile...

      }
   }

   template<typename Derived1, typename Derived2,typename Derived3>
   void runCPUEquivalentProfile(Eigen::MatrixBase<Derived1> & x_newCPU, const Eigen::MatrixBase<Derived2> &T, Eigen::MatrixBase<Derived1> & x_old, const Eigen::MatrixBase<Derived1> & d, const Eigen::MatrixBase<Derived3> & mu){


      // Do Jor Scheme!
      //TODO platformstl::performance_counter counter;
     START_TIMER(start);

     if(VariantId < 6 || VariantId == 7 ){
        // Prox- Iteration
         x_old.swap(x_newCPU.derived());

         bool m_bConverged = false;
         for(m_nIterCPU=0; m_nIterCPU < m_nMaxIterations ; m_nIterCPU++)
         {

            x_old.swap(x_newCPU.derived());

            x_newCPU.noalias() = T * x_old + d;
            Prox::ProxFunction<ConvexSets::RPlusAndDisk>::doProxMulti(mu,x_newCPU);

            if(bAbortIfConverged){
                  // Check each nCheckConvergedFlag  the converged flag
                  if(m_nIterCPU % nCheckConvergedFlag == 0){
                        //Calculate CancelCriteria
                        m_bConverged = Numerics::cancelCriteriaValue(x_newCPU,x_old, m_absTOL, m_relTOL);
                        if (m_bConverged == true)
                        {
                           break;
                        }
                  }
            }

         }
     }
     else if(VariantId == 6){
        // Prox- Iteration
         x_old.swap(x_newCPU.derived());

         bool m_bConverged = false;
         for(m_nIterCPU=0; m_nIterCPU < m_nMaxIterations ; m_nIterCPU++)
         {

            x_old.swap(x_newCPU.derived());

            Prox::ProxFunction<ConvexSets::RPlusAndDisk>::doProxMulti(mu,x_newCPU);

         }
     }

      STOP_TIMER_NANOSEC(count,start)


      m_cpuIterationTime = (count*1e-6 / m_nIterCPU);

      *m_pLog << " ---> CPU Sequential Iteration time: " <<  tinyformat::format("%1$8.6f ms",m_cpuIterationTime) <<std::endl;
      *m_pLog << " ---> nIterations: " << m_nIterCPU <<std::endl;
      if (m_nIterCPU == nMaxIterations){
         *m_pLog << " ---> Not converged! Max. Iterations reached."<<std::endl;
      }

   }

   template<typename Derived1, typename Derived2,typename Derived3>
   void runCPUEquivalentPlain(Eigen::MatrixBase<Derived1> & x_newCPU, const Eigen::MatrixBase<Derived2> &T, Eigen::MatrixBase<Derived1> & x_old, const Eigen::MatrixBase<Derived1> & d, const Eigen::MatrixBase<Derived3> & mu){

      // Do Jor Scheme!

      START_TIMER(start);

      // Prox- Iteration
      x_old.swap(x_newCPU.derived());

      bool m_bConverged = false;
      for (m_nIterCPU=0; m_nIterCPU < m_nMaxIterations ; m_nIterCPU++)
      {

         x_old.swap(x_newCPU.derived());

         x_newCPU.noalias() = T * x_old + d;
         Prox::ProxFunction<ConvexSets::RPlusAndDisk>::doProxMulti(mu,x_newCPU);

         if(bAbortIfConverged){
               // Check each nCheckConvergedFlag  the converged flag
               if(m_nIterCPU % nCheckConvergedFlag == 0){
                     //Calculate CancelCriteria
                     m_bConverged = Numerics::cancelCriteriaValue(x_newCPU,x_old, m_absTOL, m_relTOL);
                     if (m_bConverged == true)
                     {
                        break;
                     }
               }
         }

      }

      STOP_TIMER_NANOSEC(count,start)
      m_cpuIterationTime = (count*1e-6 / m_nIterCPU);

   }

   template<typename Derived1, typename Derived2, typename Derived3>
   void runGPUProfile(Eigen::MatrixBase<Derived1> & x_newGPU, const Eigen::MatrixBase<Derived2> &T, const Eigen::MatrixBase<Derived1> & x_old, const Eigen::MatrixBase<Derived1> & d, const Eigen::MatrixBase<Derived3> & mu){


      ASSERTMSG(x_newGPU.rows() == x_old.rows(), "Wrong Dimensions");
      ASSERTMSG(T.cols() == T.rows(), "Wrong Dimensions");
      ASSERTMSG(x_old.rows() == d.rows(), "Wrong Dimensions");
      ASSERTMSG(mu.rows() * TConvexSet::Dimension == x_old.rows(), mu.rows() * TConvexSet::Dimension << " , " << x_old.rows() << "Wrong Dimensions" );


      if(VariantId !=7){
         //Copy Data
         CHECK_CUDA(cudaEventRecord(m_startCopy,0));
         copyMatrixToDevice(T_dev, T);
         copyMatrixToDevice(d_dev, d);
         copyMatrixToDevice(x_old_dev,x_old);
         copyMatrixToDevice(mu_dev,mu);
         CHECK_CUDA(cudaEventRecord(m_stopCopy,0));
         CHECK_CUDA(cudaEventSynchronize(m_stopCopy));

         float time;
         CHECK_CUDA( cudaEventElapsedTime(&time,m_startCopy,m_stopCopy));
         m_elapsedTimeCopyToGPU = time;
         *m_pLog << " ---> Copy time to GPU:"<< tinyformat::format("%1$8.6f ms", time)<<std::endl;


         *m_pLog << " ---> Iterations started..."<<std::endl;

         m_absTOL = 1e-8;
         m_relTOL = 1e-10;

         CHECK_CUDA(cudaEventRecord(m_startKernel,0));

         runKernel();

         CHECK_CUDA(cudaEventRecord(m_stopKernel,0));
         CHECK_CUDA(cudaGetLastError());
         CHECK_CUDA(cudaEventSynchronize(m_stopKernel));


         CHECK_CUDA(cudaThreadSynchronize());
         *m_pLog<<" ---> Iterations finished" << std::endl;
         CHECK_CUDA( cudaEventElapsedTime(&time,m_startKernel,m_stopKernel));
         m_gpuIterationTime = (time/(double)m_nIterGPU);

         *m_pLog << " ---> GPU Iteration time :"<< tinyformat::format("%1$8.6f ms",m_gpuIterationTime) <<std::endl;
         *m_pLog << " ---> nIterations: " << m_nIterGPU <<std::endl;
         if (m_nIterGPU == nMaxIterations){
            *m_pLog << " ---> Max. Iterations reached."<<std::endl;
         }

         // Copy results back
         CHECK_CUDA(cudaEventRecord(m_startCopy,0));
         utilCuda::copyMatrixToHost(x_newGPU,x_new_dev);
         CHECK_CUDA(cudaEventRecord(m_stopCopy,0));
         CHECK_CUDA(cudaEventSynchronize(m_stopCopy));
         CHECK_CUDA( cudaEventElapsedTime(&time,m_startCopy,m_stopCopy));
         m_elapsedTimeCopyFromGPU = time;
         *m_pLog << " ---> Copy time from GPU:"<< tinyformat::format("%1$8.6f ms", time) <<std::endl;

         }
      else{
         // make reference to the eigen stuff
         Derived1 x_old2 = x_old; // Copy x_old!
         m_elapsedTimeCopyToGPU = 0;

          START_TIMER(start);
            runCPUParallel(x_newGPU,T,x_old2,d,mu);
          STOP_TIMER_NANOSEC(count,start)
          m_gpuIterationTime = ((count)*1e-6 / nMaxIterations);

          *m_pLog << " ---> CPU Parallel Iteration time :"<< tinyformat::format("%1$8.6f ms",m_gpuIterationTime) <<std::endl;
          *m_pLog << " ---> nIterations: " << m_nIterGPU <<std::endl;
           if (m_nIterGPU == nMaxIterations){
              *m_pLog << " ---> Max. Iterations reached."<<std::endl;
           }
      }
   }

   template<typename Derived1, typename Derived2, typename Derived3>
   bool runGPUPlain(Eigen::MatrixBase<Derived1> & x_newGPU, const Eigen::MatrixBase<Derived2> &T, const Eigen::MatrixBase<Derived1> & x_old, const Eigen::MatrixBase<Derived1> & d, const Eigen::MatrixBase<Derived3> & mu){


      ASSERTMSG(x_newGPU.rows() == x_old.rows(), "Wrong Dimensions");
      ASSERTMSG(T.cols() == T.rows(), "Wrong Dimensions");
      ASSERTMSG(x_old.rows() == d.rows(), "Wrong Dimensions");
      ASSERTMSG(mu.rows() * TConvexSet::Dimension == x_old.rows(), mu.rows() * TConvexSet::Dimension << " , " << x_old.rows() << "Wrong Dimensions" );



         //Malloc
         cudaError_t error;

         error = utilCuda::releaseAndMallocMatrixDevice<alignMatrix>(T_dev, T); CHECK_CUDA(error);
         error = utilCuda::releaseAndMallocMatrixDevice<false>(d_dev, d);  CHECK_CUDA(error);
         error = utilCuda::releaseAndMallocMatrixDevice<false>(x_old_dev,x_old); CHECK_CUDA(error);

         if(TypeTraitsHelper::IsSame<TConvexSet,ConvexSets::RPlusAndDisk>::result){
            ASSERTMSG(x_old.rows()%TConvexSet::Dimension==0,"Wrong Dimension!");
            error = utilCuda::releaseAndMallocMatrixDevice<false>(mu_dev, x_old.rows()/TConvexSet::Dimension, x_old.cols()); CHECK_CUDA(error);
         }
         else if(TypeTraitsHelper::IsSame<TConvexSet,ConvexSets::RPlusAndContensouEllipsoid>::result){
             ASSERTMSG(false, "NOT IMPLEMENTED!");
         }

         error = utilCuda::releaseAndMallocMatrixDevice<false>(x_new_dev,x_old_dev.m_M,x_old_dev.m_N); CHECK_CUDA(error);
         error = cudaMalloc(&pConvergedFlag_dev,sizeof(bool)); CHECK_CUDA(error);



      if(VariantId !=7){
         //Copy Data

         copyMatrixToDevice(T_dev, T);
         copyMatrixToDevice(d_dev, d);
         copyMatrixToDevice(x_old_dev,x_old);
         copyMatrixToDevice(mu_dev,mu);

         cudaEventCreate(&m_startKernel);
         cudaEventCreate(&m_stopKernel);


         cudaEventRecord(m_startKernel,0);
         runKernel();
         cudaEventRecord(m_stopKernel,0);

         cudaThreadSynchronize();

         float time;
         cudaEventElapsedTime(&time,m_startKernel,m_stopKernel);
         m_gpuIterationTime = (time/(double)m_nIterGPU);


         // Copy results back
         utilCuda::copyMatrixToHost(x_newGPU,x_new_dev);

         cudaEventDestroy(m_startKernel);
         cudaEventDestroy(m_stopKernel);

         }
      else{
         // make reference to the eigen stuff
         Derived1 x_old2 = x_old; // Copy x_old!

        runCPUParallel(x_newGPU,T,x_old2,d,mu);

      }

      return true;
   }


   // Partial Specialization of Member Function in template Class is not allowed.. Can do alternative struct wrapping, but this gets nasty!
   void runKernel(){

      if(VariantId == 1){

         // Swap pointers of old and new on the device
         std::swap(x_old_dev.m_pDevice, x_new_dev.m_pDevice);

         for (m_nIterGPU=0; m_nIterGPU< m_nMaxIterations ; m_nIterGPU++){

            // Swap pointers of old and new on the device
            std::swap(x_old_dev.m_pDevice, x_new_dev.m_pDevice);

            // Do one Jor Step
            matrixVectorMultGPU::cublasGemv(m_cublasHandle,1.0,T_dev,x_old_dev,0.0,x_new_dev);
            // Vector add temp_dev + d_dev --> x_dev
            dim3 threads(192);
            dim3 blocks((x_new_dev.m_M + (threads.x-1))/threads.x);
            vectorAddGPU::vectorAddShared_kernelWrap(x_new_dev.m_pDevice,x_new_dev.m_pDevice,d_dev.m_pDevice,x_new_dev.m_M,threads,blocks);

            proxGPU::proxContactOrdered_1threads_kernelWrap<TConvexSet>(mu_dev,x_new_dev);


            if(bAbortIfConverged){
               // Check each nCheckConvergedFlag  the converged flag
               if(m_nIterGPU % nCheckConvergedFlag == 0){
                   // First set the converged flag to 1
                   cudaMemset(pConvergedFlag_dev,1,sizeof(bool));

                  // Check convergence
                  proxGPU::convergedEach_kernelWrap(x_new_dev,x_old_dev,pConvergedFlag_dev,m_absTOL,m_relTOL);

                  // Download the flag from the GPU and check
                  cudaMemcpy(&m_bConvergedFlag,pConvergedFlag_dev,sizeof(bool),cudaMemcpyDeviceToHost);
                  if(m_bConvergedFlag == true){
                     // Converged
                     break;
                  }
               }
            }


         }
      }
      else if (VariantId == 2){
         // Swap pointers of old and new on the device
         std::swap(x_old_dev.m_pDevice, x_new_dev.m_pDevice);

        for (m_nIterGPU=0; m_nIterGPU< m_nMaxIterations ; m_nIterGPU++){

            // Swap pointers of old and new on the device
            std::swap(x_old_dev.m_pDevice, x_new_dev.m_pDevice);

            // Do one Jor Step
            matrixVectorMultGPU::cublasGemv(m_cublasHandle,1.0,T_dev,x_old_dev,0.0,x_new_dev);
            proxGPU::proxContactOrdered_1threads_kernelWrap<TConvexSet>(mu_dev,x_new_dev, d_dev);


            if(bAbortIfConverged){
               // Check each nCheckConvergedFlag  the converged flag
               if(m_nIterGPU % nCheckConvergedFlag == 0){
                  // First set the converged flag to 1
                   cudaMemset(pConvergedFlag_dev,1,sizeof(bool));
                  // Check convergence
                  proxGPU::convergedEach_kernelWrap(x_new_dev,x_old_dev,pConvergedFlag_dev,m_absTOL,m_relTOL);

                  // Download the flag from the GPU and check
                  cudaMemcpy(&m_bConvergedFlag,pConvergedFlag_dev,sizeof(bool),cudaMemcpyDeviceToHost);
                  if(m_bConvergedFlag == true){
                     // Converged
                     break;
                  }
               }
            }
         }
      }
      else if(VariantId == 3){

         // Swap pointers of old and new on the device
         std::swap(x_old_dev.m_pDevice, x_new_dev.m_pDevice);

        for (m_nIterGPU=0; m_nIterGPU< m_nMaxIterations ; m_nIterGPU++){

            // Swap pointers of old and new on the device
            std::swap(x_old_dev.m_pDevice, x_new_dev.m_pDevice);

            // Do one Jor Step
            proxGPU::jorProxContactOrdered_1threads_kernelWrap<JorProxSettings3>(mu_dev,x_new_dev,1.0,T_dev,x_old_dev,1.0,d_dev);

            if(bAbortIfConverged){
               // Check each nCheckConvergedFlag  the converged flag
               if(m_nIterGPU % nCheckConvergedFlag == 0){
                   // First set the converged flag to 1
                   cudaMemset(pConvergedFlag_dev,1,sizeof(bool));
                  // Check convergence
                  proxGPU::convergedEach_kernelWrap(x_new_dev,x_old_dev,pConvergedFlag_dev,m_absTOL,m_relTOL);

                  // Download the flag from the GPU and check
                  cudaMemcpy(&m_bConvergedFlag,pConvergedFlag_dev,sizeof(bool),cudaMemcpyDeviceToHost);
                  if(m_bConvergedFlag == true){
                     // Converged
                     break;
                  }
               }
            }
         }

      }
      else if(VariantId == 4){
         // Swap pointers of old and new on the device
         std::swap(x_old_dev.m_pDevice, x_new_dev.m_pDevice);

         for (m_nIterGPU=0; m_nIterGPU< m_nMaxIterations ; m_nIterGPU++){

            // Swap pointers of old and new on the device
            std::swap(x_old_dev.m_pDevice, x_new_dev.m_pDevice);

            // Do one Jor Step
            proxGPU::jorProxContactOrdered_1threads_kernelWrap<JorProxSettings4>(mu_dev,x_new_dev,1.0,T_dev,x_old_dev,1.0,d_dev);

            if(bAbortIfConverged){
               // Check each nCheckConvergedFlag  the converged flag
               if(m_nIterGPU % nCheckConvergedFlag == 0){
                  // First set the converged flag to 1
                  cudaMemset(pConvergedFlag_dev,1,sizeof(bool));
                  // Check convergence
                  proxGPU::convergedEach_kernelWrap(x_new_dev,x_old_dev,pConvergedFlag_dev,m_absTOL,m_relTOL);

                  // Download the flag from the GPU and check
                  cudaMemcpy(&m_bConvergedFlag,pConvergedFlag_dev,sizeof(bool),cudaMemcpyDeviceToHost);
                  if(m_bConvergedFlag == true){
                     // Converged
                     break;
                  }
               }
            }
         }

      }
      else if(VariantId == 5){

         // Swap pointers of old and new on the device
         std::swap(x_old_dev.m_pDevice, x_new_dev.m_pDevice);

         for (m_nIterGPU=0; m_nIterGPU< m_nMaxIterations ; m_nIterGPU++){

            // Swap pointers of old and new on the device
            std::swap(x_old_dev.m_pDevice, x_new_dev.m_pDevice);

            // Do one Jor Step
            matrixVectorMultGPU::matrixVectorMultiply_kernelWrap(x_new_dev,1,1.0,T_dev,x_old_dev,1,1.0,d_dev,1);
            proxGPU::proxContactOrdered_1threads_kernelWrap<TConvexSet>(mu_dev,x_new_dev);


            if(bAbortIfConverged){
               // Check each nCheckConvergedFlag  the converged flag
               if(m_nIterGPU % nCheckConvergedFlag == 0){
                  // First set the converged flag to 1
                   cudaMemset(pConvergedFlag_dev,1,sizeof(bool));
                  // Check convergence
                  proxGPU::convergedEach_kernelWrap(x_new_dev,x_old_dev,pConvergedFlag_dev,m_absTOL,m_relTOL);

                  // Download the flag from the GPU and check
                  cudaMemcpy(&m_bConvergedFlag,pConvergedFlag_dev,sizeof(bool),cudaMemcpyDeviceToHost);
                  if(m_bConvergedFlag == true){
                     // Converged
                     break;
                  }
               }
            }
         }
      }
      else if(VariantId == 6){

         // Swap pointers of old and new on the device
         std::swap(x_old_dev.m_pDevice, x_new_dev.m_pDevice);

         for (m_nIterGPU=0; m_nIterGPU< m_nMaxIterations ; m_nIterGPU++){

            // Swap pointers of old and new on the device
            std::swap(x_old_dev.m_pDevice, x_new_dev.m_pDevice);

            proxGPU::proxContactOrderedWORSTCASE_1threads_kernelWrap<TConvexSet>(mu_dev,x_new_dev);

         }
      }
   }

    template<typename Derived1, typename Derived2,typename Derived3>
   void runCPUParallel(Eigen::MatrixBase<Derived1> & x_newCPU, const Eigen::MatrixBase<Derived2> &T, Eigen::MatrixBase<Derived1> & x_old, const Eigen::MatrixBase<Derived1> & d, const Eigen::MatrixBase<Derived3> & mu){


      x_old.swap(x_newCPU.derived());

         bool m_bConverged = false;
         for (m_nIterGPU=0; m_nIterGPU< m_nMaxIterations ; m_nIterGPU++)
         {

            x_old.swap(x_newCPU.derived());

            matrixVectorMultGPU::blasGemv<PREC>::run(1.0,T,x_old,0.0,x_newCPU);

            x_newCPU.noalias() += d;

            Prox::ProxFunction<ConvexSets::RPlusAndDisk>::doProxMulti(mu,x_newCPU);

            if(bAbortIfConverged){
                  // Check each nCheckConvergedFlag  the converged flag
                  if(m_nIterGPU % nCheckConvergedFlag == 0){
                        //Calculate CancelCriteria
                        m_bConverged = Numerics::cancelCriteriaValue(x_newCPU,x_old, m_absTOL, m_relTOL);
                        if (m_bConverged == true)
                        {
                           break;
                        }
                  }
            }

         }

   }



   void cleanUpTestProblem(){
      if(VariantId!=7){
      CHECK_CUDA(freeMatrixDevice(T_dev));
      CHECK_CUDA(freeMatrixDevice(d_dev));
      CHECK_CUDA(freeMatrixDevice(x_old_dev));
      CHECK_CUDA(freeMatrixDevice(x_new_dev));
      CHECK_CUDA(freeMatrixDevice(mu_dev));
     if(pConvergedFlag_dev){
         CHECK_CUDA(cudaFree(pConvergedFlag_dev));
         pConvergedFlag_dev = NULL;
     }
     }else{

     }

      CHECK_CUBLAS(cublasDestroy(m_cublasHandle));
      CHECK_CUDA(cudaEventDestroy(m_startKernel));
      CHECK_CUDA(cudaEventDestroy(m_stopKernel));
      CHECK_CUDA(cudaEventDestroy(m_startCopy));
      CHECK_CUDA(cudaEventDestroy(m_stopCopy));

   }

   double m_gpuIterationTime;
   double m_elapsedTimeCopyToGPU;
   double m_elapsedTimeCopyFromGPU;
   int m_nIterGPU;

   double m_cpuIterationTime;
   int m_nIterCPU;

private:

   utilCuda::CudaMatrix<PREC> T_dev, d_dev, x_old_dev, x_new_dev, mu_dev;
   bool *pConvergedFlag_dev;
   bool m_bConvergedFlag;

   unsigned int m_nMaxIterations;
   PREC m_absTOL, m_relTOL;

   cublasHandle_t m_cublasHandle;
   cudaEvent_t m_startKernel, m_stopKernel, m_startCopy,m_stopCopy;

   std::ostream* m_pLog;
   std::ofstream* m_pMatlab_file;
};

/** @} */


#endif
