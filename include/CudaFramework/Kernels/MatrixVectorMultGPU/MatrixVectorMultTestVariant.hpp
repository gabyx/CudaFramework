// ========================================================================================
//  CudaFramework
//  Copyright (C) 2014 by Gabriel Nützi <nuetzig (at) imes (d0t) mavt (d0t) ethz (d0t) ch>
//
//  This Source Code Form is subject to the terms of the GNU GPL 3.0 licence.
//  If a copy of the GNU GPL 3.0 was not distributed with this
//  file, you can obtain one at http://opensource.org/licenses/GPL-3.0.
// ========================================================================================

#ifndef CudaFramework_Kernels_MatrixVectorMultGPU_MatrixVectorMultTestVariant_hpp
#define CudaFramework_Kernels_MatrixVectorMultGPU_MatrixVectorMultTestVariant_hpp

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <time.h>

#include "CudaFramework/General/CPUTimer.hpp"
#include "CudaFramework/CudaModern/CudaError.hpp"
#include "CudaFramework/General/Utilities.hpp"
#include "CudaFramework/CudaModern/CudaMatrix.hpp"
#include "CudaFramework/CudaModern/CudaMatrixUtilities.hpp"

#include "CudaFramework/General/FloatingPointType.hpp"

#include <Eigen/Dense>

#include <cuda_runtime.h>
#include "cublas_v2.h"


/**
* @addtogroup KernelTestMethod
* @{
*/



/**
* @addtogroup TestVariants
* @{
*/


/**
* @defgroup MatrixVectorMultTestVariant Matrix Vector Test Variant
* This provides a Test Variant for Matrix Vector multiplication.
*
*
* How to launch this Test Variant:
* @code

   #include "CudaFramework/PerformanceTest/PerformanceTest.hpp"
   #include "CudaFramework/Kernels/MatrixVectorMultGPU/MatrixVectorMultTestVariant.hpp"

   typedef KernelTestMethod<                                         // Use the KernelTestMethod!
         KernelTestMethodSettings<false,true> ,                      // Some settings for the KernelTestMethod!
         MatrixVectorMultTestVariant<                                // The Test Variant which should be tested with the KernelTestMethod
            MatrixVectorMultSettings<                                   // Some settings for the Test Variant
               double,                                                  // Use double
               3,                                                       // ContactSize = 3
               false,                                                   // Dont write to file!
               10,                                                      // max iterations for the multiplication
               MatrixVectorMultPerformanceTestSettings<0,20,2000,5>,    // Some Performance settings, how to iterate over all test problems, and over the random runs!
               MatrixVectorMultGPUVariantSettings<2,true>               // The actual GPUVariant 2 for this Test Variant, alignMatrix is true
            >
         >
    > test1;

   PerformanceTest<test1> A("test1");  // Define a Performance Test!
   A.run();                            // Run the Test!
* @endcode
* @{
*/

template<typename _PREC, int _nValuesPerContact, bool _bwriteToFile, int _nMaxIterations,  typename _TPerformanceTestSettings, typename _TGPUVariantSetting >
struct MatrixVectorMultSettings {
    typedef _PREC PREC;
    static const int nValuesPerContact = _nValuesPerContact;
    static const bool bWriteToFile = _bwriteToFile;
    static const int nMaxIterations = _nMaxIterations;
    typedef  _TPerformanceTestSettings TPerformanceTestSettings;
    typedef  typename _TGPUVariantSetting::template GPUVariant<PREC,nValuesPerContact,nMaxIterations>::TGPUVariant TGPUVariant;
};

#define DEFINE_MatrixVectorMult_Settings(_TSettings_) \
   typedef typename _TSettings_::PREC PREC; \
   static const int nValuesPerContact = _TSettings_::nValuesPerContact; \
   static const bool bWriteToFile = _TSettings_::bWriteToFile; \
   static const int nMaxIterations = _TSettings_::nMaxIterations; \
   typedef typename _TSettings_::TPerformanceTestSettings TPerformanceTestSettings; \
   typedef typename _TSettings_::TGPUVariant TGPUVariant;


template< int _minNContacts, int _stepNContacts, int _maxNContacts, int _maxNRandomRuns >
struct MatrixVectorMultPerformanceTestSettings {
    static const int minNContacts = _minNContacts;
    static const int stepNContacts = _stepNContacts;
    static const int maxNContacts = _maxNContacts;
    static const int maxNRandomRuns = _maxNRandomRuns;
};
#define DEFINE_MatrixVectorMultPerformanceTest_Settings(_TSettings_)  \
   static const int minNContacts = _TSettings_::minNContacts; \
   static const int  stepNContacts = _TSettings_::stepNContacts; \
   static const int  maxNContacts = _TSettings_::maxNContacts; \
   static const int  maxNRandomRuns = _TSettings_::maxNRandomRuns;



template<typename TMatrixVectorMultSettings>
class MatrixVectorMultTestVariant {
public:

    DEFINE_MatrixVectorMult_Settings(TMatrixVectorMultSettings);
    DEFINE_MatrixVectorMultPerformanceTest_Settings(TPerformanceTestSettings);

    static std::string getTestVariantDescription() {
        return "MatrixVectorMultiplication";
    }

    MatrixVectorMultTestVariant() {
        m_nContactCounter = 0;
        m_nRandomRunsCounter =0;
    }
    ~MatrixVectorMultTestVariant() {}

    void initialize(std::ostream * pLog, std::ostream * pData) {

        m_pData = pData;
        m_pLog = pLog;



        m_nContactCounter = minNContacts;
        m_nRandomRunsCounter =0;

        std::srand ( (unsigned int)time(NULL) );

        m_gpuVariant.initialize(m_pLog);
    }

    std::vector<std::pair<std::string,std::string> >
    getDescriptions() {
        std::vector<std::pair<std::string,std::string> > s;
        s.push_back( std::make_pair("Remarks" ,"This text should explain the description to remember" ));
        s.push_back( std::make_pair("VariantName" , TGPUVariant::getVariantName() ));
        s.push_back( std::make_pair("VariantDescription" , TGPUVariant::getVariantDescription() ));
        s.push_back( std::make_pair("TotalGaussSeidelBlockIterations" , std::to_string(nMaxIterations) ));
        s.push_back( std::make_pair("MaxRandomRuns" , std::to_string(maxNRandomRuns) ));
        s.push_back( std::make_pair("PREC" , typeid(PREC).name() ));

        return s;
    }

    std::vector<std::string>
    getDataColumHeader(){
        std::vector<std::string> s;
        s.push_back("nContacts");
        return s;
    }


    bool generateNextTestProblem() {


        if(m_nContactCounter>maxNContacts) {
            *m_pLog << "No more Test Problems to generate,---> exit ============="<<std::endl;
            return false;
        }

        // Set up next test problem
        if (m_nContactCounter <= 0) {
            m_nContacts = 1;
        } else {
            m_nContacts = m_nContactCounter;
        }

        *m_pLog << "Compute test for nContacts: "<< m_nContacts <<" ============="<<std::endl;
        int MG = m_nContacts*nValuesPerContact;
        int NG = MG;
        m_nOps = 2*MG*NG-MG + MG ;
        m_nBytesReadWrite = (double)( (MG *NG) + MG) * sizeof(PREC);


        m_T.resize(MG,NG);
        m_d.resize(MG);
        m_x_old.resize(MG);
        m_x_newGPU.resize(MG);


        //reset randomRun
        m_nRandomRunsCounter = 0;

        m_gpuVariant.initializeTestProblem(m_T,m_x_old,m_d);

        // Increment counter
        m_nContactCounter += stepNContacts;

        return true;
    }

    void cleanUpTestProblem() {
        m_gpuVariant.cleanUpTestProblem();
    }

    bool generateNextRandomRun() {

        if(m_nRandomRunsCounter < maxNRandomRuns) {
            m_nRandomRunsCounter++;
        } else {
            return false;
        }

        *m_pLog << "Random Run # : "<<m_nRandomRunsCounter<<std::endl;

        // Set Values! ==============================
        m_T.setRandom();
        m_d.setRandom();
        m_x_old.setRandom();
        // ==========================================

        return true;
    }

    void runOnGPU() {

        //Select variant
        m_gpuVariant.run(m_x_newGPU,m_T,m_x_old,m_d);

        m_elapsedTimeCopyToGPU = m_gpuVariant.m_elapsedTimeCopyToGPU;
        m_gpuIterationTime = m_gpuVariant.m_gpuIterationTime;
        m_elapsedTimeCopyFromGPU = m_gpuVariant.m_elapsedTimeCopyFromGPU;
        m_nIterGPU = m_gpuVariant.m_nIter;
    }

    void runOnCPU() {

        START_TIMER(start)
        for (m_nIterCPU=0; m_nIterCPU< nMaxIterations ; m_nIterCPU++) {
            m_x_newCPU.noalias() = m_T * m_x_old + m_d;
            m_x_newCPU.swap(m_x_old);
        }
        STOP_TIMER_NANOSEC(count,start)

        m_cpuIterationTime = count*1e-6 / nMaxIterations;

        *m_pLog << "---> CPU  Iteration time: " <<  tinyformat::format("%1$8.6f ms",m_cpuIterationTime) <<std::endl;
        *m_pLog << "---> nIterations: " << m_nIterCPU <<std::endl;

        m_x_newCPU = m_x_old;
    }

    void checkResults() {

        double relTolGPUCPU = 1e-5;
        unsigned int tolUlpGPUCPU = 20;

        std::pair<bool,bool> result = Utilities::compareArraysEachCombined(m_x_newGPU.data(),m_x_newCPU.data(),m_x_newGPU.rows(),(PREC)relTolGPUCPU,tolUlpGPUCPU,m_maxRelTol,m_avgRelTol,m_maxUlp,m_avgUlp,false);
        if(result.first == true && result.second == true) {
            *m_pLog << "---> GPU/CPU identical!...." << std::endl;
        } else {
            *m_pLog << "---> GPU/CPU NOT identical!...." << std::endl;
            *m_pLog << "---> Converged RelTol: "<<result.first <<" \t Identical Ulp: "<< result.second <<std::endl;
            ///std::exit(-1);
        }

        *m_pLog << "---> maxUlp :" << (double)m_maxUlp << std::endl;
        *m_pLog << "---> avgUlp :" << m_avgUlp << std::endl;
        *m_pLog << "---> maxRelTol :" << m_maxRelTol << std::endl;
        *m_pLog << "---> avgRelTol :" << m_avgRelTol << std::endl;

    }

    void writeData() {
      tinyformat::format(*m_pData,"%.9d\t",m_nContacts);
    }

    void finalize() {
        m_gpuVariant.finalize();
    }

    int m_nIterCPU;
    int m_nIterGPU;


    double m_nOps;
    double m_nBytesReadWrite;

    double m_elapsedTimeCopyToGPU;
    double m_elapsedTimeCopyFromGPU;
    double m_cpuIterationTime;
    double m_gpuIterationTime;

    double m_maxRelTol;
    double m_avgRelTol;
    typename TypeWithSize<sizeof(PREC)>::UInt m_maxUlp;
    double m_avgUlp;

private:

    int m_nContacts;
    int m_nContactCounter;
    int m_nRandomRunsCounter;

    std::ostream* m_pData;
    std::ostream* m_pLog;


    PREC m_alpha;
    PREC m_beta;
    Eigen::Matrix<PREC,Eigen::Dynamic, Eigen::Dynamic> m_T;
    Eigen::Matrix<PREC,Eigen::Dynamic, 1> m_d;
    Eigen::Matrix<PREC,Eigen::Dynamic, 1> m_x_old;
    Eigen::Matrix<PREC,Eigen::Dynamic, 1> m_x_newCPU;
    Eigen::Matrix<PREC,Eigen::Dynamic, 1> m_x_newGPU;

    typename TMatrixVectorMultSettings::TGPUVariant m_gpuVariant;
};













/**
* @defgroup MatrixVectorMultGPUVariant Matrix Vector Mult GPUVariant
* @detailed VariantId specifies which variant is launched:
* Here the different variants have been included in one class!
* To be more flexible we can also completely reimplement the whole class for another GPUVariant!
*
* VariantId = 1: Tests the gemv cuBlas
* VariantId = 2: Tests my implementation
* VariantId = 3: Tests the symv cuBlas
* @{
*/

#include "CudaFramework/Kernels/MatrixVectorMultGPU/MatrixVectorMultGPU.hpp"


template<typename _PREC, int _VariantId, int _nValuesPerContact, int _alignMatrix, int _nMaxIterations>
struct MatrixVectorMultVariantSettingsWrapper {
    typedef _PREC PREC;
    static const int VariantId = _VariantId;
    static const int nValuesPerContact = _nValuesPerContact;
    static const int alignMatrix = _alignMatrix;
    static const int nMaxIterations = _nMaxIterations;
};

#define DEFINE_MatrixVectorMultGPUVariant_Settings(_TSettings_) \
      typedef typename _TSettings_::PREC PREC; \
      static const int VariantId = _TSettings_::VariantId; \
      static const int nValuesPerContact = _TSettings_::nValuesPerContact; \
      static const int alignMatrix = _TSettings_::alignMatrix; \
      static const int nMaxIterations = _TSettings_::nMaxIterations;


template<typename TMatrixVectorMultVariantSettings> class MatrixVectorMultGPUVariant; // Prototype

template<int _VariantId, bool _alignMatrix> // Settings from below
struct MatrixVectorMultGPUVariantSettings {

    template <typename _PREC, int _nValuesPerContact, int _nMaxIterations> // Settings from above
    struct GPUVariant {
        typedef  MatrixVectorMultGPUVariant< MatrixVectorMultVariantSettingsWrapper<_PREC,_VariantId,_nValuesPerContact,_alignMatrix,_nMaxIterations> >  TGPUVariant;
    };

};

template<typename TMatrixVectorMultVariantSettings>
class MatrixVectorMultGPUVariant {
public:

    DEFINE_MatrixVectorMultGPUVariant_Settings(TMatrixVectorMultVariantSettings);

    static std::string getVariantName() {
        return getVariantNameInternal<VariantId>();
    }
    static std::string getVariantDescription() {
        return getVariantDescriptionInternal<VariantId>() + std::string(", aligned: ") + ((alignMatrix)? std::string("on") : std::string("off"));
    }

    template<int VariantId> static std::string getVariantNameInternal() {
        switch(VariantId) {
        case 1:
            return "[cuBlas (Multiplication + Addition)]" ;
        case 2:
            return "[Multiplication + Addition]";
        case 3:
            return "[cuBlas (Multiplication + Addition) symetric]";
        }
    }
    template<int VariantId> static std::string getVariantDescriptionInternal() {
        switch(VariantId) {
        case 1:
            return  std::string("gemv routine") + std::string(", Results are not identical to CPU, due to gemv!") ;
        case 2:
            return "ThreadsPerBlock : 128, BlockDim : 128, XElementsPerThread : 4,  UnrollBlockDotProduct : 6";
        case 3:
            return "symv routine";
        }
    }

    void initialize(std::ostream* pLog) {
        m_pLog = pLog;

        CHECK_CUBLAS(cublasCreate(&m_cublasHandle));
        CHECK_CUDA(cudaEventCreate(&m_start));
        CHECK_CUDA(cudaEventCreate(&m_stop));
        CHECK_CUDA(cudaEventCreate(&m_startKernel));
        CHECK_CUDA(cudaEventCreate(&m_stopKernel));
        CHECK_CUDA(cudaEventCreate(&m_startCopy));
        CHECK_CUDA(cudaEventCreate(&m_stopCopy));

    }

    void finalize() {
        CHECK_CUBLAS(cublasDestroy(m_cublasHandle));
        CHECK_CUDA(cudaEventDestroy(m_start));
        CHECK_CUDA(cudaEventDestroy(m_stop));
        CHECK_CUDA(cudaEventDestroy(m_startKernel));
        CHECK_CUDA(cudaEventDestroy(m_stopKernel));
        CHECK_CUDA(cudaEventDestroy(m_startCopy));
        CHECK_CUDA(cudaEventDestroy(m_stopCopy));

    }

    template<typename Derived1, typename Derived2>
    void initializeTestProblem( const Eigen::MatrixBase<Derived2> &T, const Eigen::MatrixBase<Derived1> & x_old, const Eigen::MatrixBase<Derived1> & d) {

        size_t freeMem;
        size_t totalMem;

        CHECK_CUDA(cudaMemGetInfo (&freeMem, &totalMem));

        int nGPUBytes = (int)(3*T.rows() + T.rows()*T.cols())*sizeof(PREC);
        *m_pLog << "Will try to allocate ("<<nGPUBytes<<"/"<<freeMem<<") = " << (double)nGPUBytes/freeMem * 100.0 <<" % of global memory on GPU"<<std::endl;
        if(nGPUBytes > freeMem) {
            *m_pLog <<"To little memory on GPU, exit!..."<<std::endl;
            return;
        }

        CHECK_CUDA(utilCuda::mallocMatrixDevice<alignMatrix>(T_dev, T));
        CHECK_CUDA(utilCuda::mallocMatrixDevice<false>(d_dev, d));
        CHECK_CUDA(utilCuda::mallocMatrixDevice<false>(x_old_dev,x_old));
        CHECK_CUDA(utilCuda::mallocMatrixDevice<false>(x_new_dev,x_old_dev.m_M,x_old_dev.m_N,false));
    }

    template<typename Derived1, typename Derived2>
    void run(Eigen::MatrixBase<Derived1> & x_newGPU, const Eigen::MatrixBase<Derived2> &T, const Eigen::MatrixBase<Derived1> & x_old, const Eigen::MatrixBase<Derived1> & d) {


        //Copy Data
        CHECK_CUDA(cudaEventRecord(m_startCopy,0));
        utilCuda::copyMatrixToDevice(T_dev, T);
        utilCuda::copyMatrixToDevice(d_dev, d);
        utilCuda::copyMatrixToDevice(x_old_dev,x_old);
        CHECK_CUDA(cudaEventRecord(m_stopCopy,0));
        CHECK_CUDA(cudaEventSynchronize(m_stopCopy));

        float time;
        CHECK_CUDA( cudaEventElapsedTime(&time,m_startCopy,m_stopCopy));
        m_elapsedTimeCopyToGPU = time;
        *m_pLog << "---> Copy time to GPU:"<< tinyformat::format("%8.6f ms", time) <<std::endl;

        m_nIter = 0;
        *m_pLog << "---> Iterations started..."<<std::endl;

        CHECK_CUDA(cudaEventRecord(m_startKernel,0));
        for (m_nIter=0; m_nIter< nMaxIterations ; m_nIter++) {
            runKernel<VariantId>();
            std::swap(x_old_dev.m_pDevice,x_new_dev.m_pDevice);
        }
        CHECK_CUDA(cudaEventRecord(m_stopKernel,0));
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventSynchronize(m_stopKernel));


        CHECK_CUDA(cudaThreadSynchronize());
        *m_pLog<<"---> Iterations finished" << std::endl;
        CHECK_CUDA( cudaEventElapsedTime(&time,m_startKernel,m_stopKernel));
        double average = (time/(double)nMaxIterations);
        m_gpuIterationTime = average;

        *m_pLog << "---> GPU Iteration time :"<< tinyformat::format("%8.6f ms",average) <<std::endl;
        *m_pLog << "---> nIterations: " << m_nIter <<std::endl;
        if (m_nIter == nMaxIterations) {
            *m_pLog << "---> Max. Iterations reached."<<std::endl;
        }

        // Copy results back
        CHECK_CUDA(cudaEventRecord(m_startCopy,0));
        utilCuda::copyMatrixToHost(x_newGPU,x_old_dev);
        CHECK_CUDA(cudaEventRecord(m_stopCopy,0));
        CHECK_CUDA(cudaEventSynchronize(m_stopCopy));
        CHECK_CUDA( cudaEventElapsedTime(&time,m_startCopy,m_stopCopy));
        m_elapsedTimeCopyFromGPU = time;
        *m_pLog << "---> Copy time from GPU:"<< tinyformat::format("%8.6f ms",time) <<std::endl;
    }

    template<int VariantId> inline void runKernel() {
        if(VariantId==1) {
            matrixVectorMultGPU::cublasGemv(m_cublasHandle,1.0,T_dev,x_old_dev,1.0,d_dev);
        } else if(VariantId==2) {
            matrixVectorMultGPU::matrixVectorMultiply_kernelWrap<PREC>(x_new_dev,1,1.0,T_dev,x_old_dev,1,1.0,d_dev,1);
        } else if(VariantId==3) {
            matrixVectorMultGPU::cublasSymv(m_cublasHandle,1.0,T_dev,x_old_dev,0.0,x_new_dev);
        }
    }


    void cleanUpTestProblem() {

        CHECK_CUDA(freeMatrixDevice(T_dev));
        CHECK_CUDA(freeMatrixDevice(d_dev));
        CHECK_CUDA(freeMatrixDevice(x_old_dev));
        CHECK_CUDA(freeMatrixDevice(x_new_dev));
    }

    double m_gpuIterationTime;
    double m_elapsedTimeCopyToGPU;
    double m_elapsedTimeCopyFromGPU;
    double m_nIter;

private:

    utilCuda::CudaMatrix<PREC> T_dev, d_dev, x_old_dev, x_new_dev;

    cublasHandle_t m_cublasHandle;
    cudaEvent_t m_start, m_stop, m_startKernel, m_stopKernel, m_startCopy,m_stopCopy;

    std::ostream* m_pLog;
};
/** @} */



/** @} */ // MatrixVectorMultTestVariant


/** @} */ // addtogroup TestVariants

/** @} */ // addtogroup KernelTestMethods

#endif
