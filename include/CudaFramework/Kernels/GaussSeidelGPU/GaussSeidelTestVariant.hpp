// ========================================================================================
//  CudaFramework
//  Copyright (C) 2014 by Gabriel Nützi <nuetzig (at) imes (d0t) mavt (d0t) ethz (d0t) ch>
//
//  This Source Code Form is subject to the terms of the GNU GPL 3.0 licence.
//  If a copy of the GNU GPL 3.0 was not distributed with this
//  file, you can obtain one at http://opensource.org/licenses/GPL-3.0.
// ========================================================================================

#ifndef CudaFramework_Kernels_GaussSeidelGPU_GaussSeidelTestVariant_hpp
#define CudaFramework_Kernels_GaussSeidelGPU_GaussSeidelTestVariant_hpp

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
#include "CudaFramework/General/FlopsCounting.hpp"

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
* @defgroup GaussSeidelBlockTestVariant GaussSeidelBlock Test Variant
* This provides a Test Variant for Jor or Sor GaussSeidelBlock.
*
* How to launch this Test Variant:
* @code

 typedef KernelTestMethod<
         KernelTestMethodSettings<false,true> ,
         GaussSeidelBlockTestVariant<
            GaussSeidelBlockSettings<
               double,
               3,
               false,
               10,
               false,
               10,
               GaussSeidelBlockPerformanceTestSettings<4000,20,4000,5>,
               JorGaussSeidelBlockGPUVariantSettings<5,true>
            >
         >
    > test1;
    PerformanceTest<test1> A("test1");
    A.run();
* @endcode

* @{
*/


template<typename _PREC, int _nValuesPerContact, int _nBlockSize, bool _bwriteToFile, int _nMaxIterations, bool _bAbortIfConverged, int _nCheckConvergedFlag,  typename _TPerformanceTestSettings, typename _TGPUVariantSetting >
struct GaussSeidelBlockSettings {
    typedef _PREC PREC;
    static const int nValuesPerContact = _nValuesPerContact;
    static const bool bWriteToFile = _bwriteToFile;
    static const int nMaxIterations = _nMaxIterations;
    static const int nBlockSize = 64;
    static const bool bAbortIfConverged = _bAbortIfConverged;
    static const int nCheckConvergedFlag = _nCheckConvergedFlag;
    typedef  _TPerformanceTestSettings TPerformanceTestSettings;
    typedef typename _TGPUVariantSetting::template GPUVariant<PREC,nValuesPerContact,nMaxIterations,nBlockSize,bAbortIfConverged,nCheckConvergedFlag>::TGPUVariant TGPUVariant;
};

#define DEFINE_GaussSeidelBlock_Settings(_TSettings_) \
   typedef typename _TSettings_::PREC PREC; \
   static const int nValuesPerContact = _TSettings_::nValuesPerContact; \
   static const bool bWriteToFile = _TSettings_::bWriteToFile; \
   static const int nMaxIterations = _TSettings_::nMaxIterations; \
   static const int nBlockSize = _TSettings_::nBlockSize; \
   static const bool bAbortIfConverged = _TSettings_::bAbortIfConverged; \
   static const int nCheckConvergedFlag = _TSettings_::nCheckConvergedFlag; \
   typedef typename _TSettings_::TPerformanceTestSettings TPerformanceTestSettings; \
   typedef typename _TSettings_::TGPUVariant TGPUVariant;


template< int _minNContacts, int _maxNContacts, int _maxNRandomRuns >
struct GaussSeidelBlockPerformanceTestSettings {
    static const int minNContacts = _minNContacts;
    static const int stepNContacts = 64;
    static const int maxNContacts = _maxNContacts;
    static const int maxNRandomRuns = _maxNRandomRuns;
};
#define DEFINE_GaussSeidelBlockPerformanceTestSettings_Settings(_TSettings_)  \
   static const int  minNContacts = _TSettings_::minNContacts; \
   static const int  stepNContacts = _TSettings_::stepNContacts; \
   static const int  maxNContacts = _TSettings_::maxNContacts; \
   static const int  maxNRandomRuns = _TSettings_::maxNRandomRuns;


template<typename TGaussSeidelBlockSettings>
class GaussSeidelBlockTestVariant {
public:

    DEFINE_GaussSeidelBlock_Settings(TGaussSeidelBlockSettings);
    DEFINE_GaussSeidelBlockPerformanceTestSettings_Settings(TPerformanceTestSettings);

    static std::string getTestVariantDescription() {
        return "GaussSeidelBlock";
    }

    GaussSeidelBlockTestVariant():
        Matlab(Eigen::FullPrecision, 0, ", ", ";\n", "", "", "[", "]") {
        m_nContactCounter = 0;
        m_nRandomRunsCounter =0;
    }
    ~GaussSeidelBlockTestVariant() {
        m_matlab_file.close();
    }

     bool checkSettings(unsigned int gpuID){
        return m_gpuVariant.checkSettings(gpuID);
    }


    void initialize(std::ostream * pLog, std::ostream * pData) {

        m_pData = pData;
        m_pLog = pLog;



        m_nContactCounter =  ((int)(minNContacts + (stepNContacts -1 ) ) / stepNContacts) * stepNContacts ;

        if(m_nContactCounter <=0) {
            m_nContactCounter = stepNContacts;
        }

        m_nRandomRunsCounter =0;

        std::srand ( (unsigned int)time(NULL) );

        if(bWriteToFile) {
            m_matlab_file.close();
            m_matlab_file.open("GaussSeidelBlockStep.m_M", std::ios::trunc | std::ios::out);
            m_matlab_file.clear();
            if (m_matlab_file.is_open()) {
                std::cout << " File opened: " << "GaussSeidelBlockStep.m_M"<<std::endl;
            }

        }


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
            *m_pLog << "No more Test Problems to generate, --->exit ============="<<std::endl;
            return false;
        }


        m_nContacts = m_nContactCounter;
        *m_pLog << "Compute test for nContacts: "<< m_nContacts <<" ============="<<std::endl;
        m_MG = m_nContacts*nValuesPerContact;
        m_NG = m_MG;
        m_nOps = proxContactOrdered_RPlusAndDisk_1threads_kernel_FLOPS(m_nContacts) + evaluateProxTermJOR_FLOPS(m_MG, m_NG) ;
        m_nBytesReadWrite = ( m_MG       +    (m_MG *m_NG) + m_MG) * (double)sizeof(PREC);



        m_G.resize(m_MG,m_NG);
        m_c.resize(m_MG);
        m_T.resize(m_MG,m_NG);
        m_d.resize(m_MG);

        m_t.resize(m_MG);
        m_t_newCPU.resize(m_MG);
        m_t_newGPU.resize(m_MG);

        m_x_old.resize(m_MG);
        m_x_newCPU.resize(m_MG);
        m_x_newGPU.resize(m_MG);


        //reset randomRun
        m_nRandomRunsCounter = 0;

        m_gpuVariant.initializeTestProblem(m_T,m_x_old,m_d,m_t);

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
        m_G.setRandom();
        m_G += (Eigen::Matrix<PREC,Eigen::Dynamic,1>::Ones(m_MG) * 20).asDiagonal();
        m_c.setRandom();

        m_T = m_G.diagonal().asDiagonal().inverse() * m_G;
        m_d = m_G.diagonal().asDiagonal().inverse() * m_c;

        m_t.setZero();
        m_x_old.setZero();
        // ==========================================


        if(bWriteToFile) {
            m_matlab_file << "G=" << m_G.format(Matlab)<<";" <<std::endl;
            m_matlab_file << "c=" << m_c.transpose().format(Matlab)<<"';" <<std::endl;
            m_matlab_file << "t=" << m_t.transpose().format(Matlab)<<"';" <<std::endl;
            m_matlab_file << "x_old=" << m_x_old.transpose().format(Matlab)<<"';" <<std::endl;
        }

        return true;
    }

    void runOnGPU() {

        //Select variant
        m_gpuVariant.run(m_x_newGPU,m_t_newGPU ,m_T,m_x_old,m_d,m_t);

        m_elapsedTimeCopyToGPU = m_gpuVariant.m_elapsedTimeCopyToGPU;
        m_gpuIterationTime = m_gpuVariant.m_gpuIterationTime;
        m_elapsedTimeCopyFromGPU = m_gpuVariant.m_elapsedTimeCopyFromGPU;
        m_nIterGPU = m_gpuVariant.m_nIter;


        if(bWriteToFile) {
            m_matlab_file << "t_newGPU=" << m_t_newGPU.transpose().format(Matlab)<<"';" <<std::endl;
            m_matlab_file << "x_newGPU=" << m_x_newGPU.transpose().format(Matlab)<<"';" <<std::endl;
        }
    }

    void runOnCPU() {

        START_TIMER(start)

        // GaussSeidelBlock- Iteration
        m_nIterCPU = m_nIterGPU;
        Utilities::gaussSeidelBlockCorrect(m_G,m_c,m_x_old,m_nIterCPU,nBlockSize);

        STOP_TIMER_NANOSEC(count,start)

        m_cpuIterationTime = count*1e-6 / nMaxIterations;

        *m_pLog << " ---> CPU  Iteration time: " <<  tinyformat::format("%1$8.6f ms", m_cpuIterationTime) <<std::endl;
        *m_pLog << " ---> nIterations: " << m_nIterCPU <<std::endl;
        if (m_nIterCPU == nMaxIterations) {
            *m_pLog << " ---> Not converged! Max. Iterations reached."<<std::endl;
        }

        m_x_newCPU = m_x_old;
        m_t_newCPU = m_t;


        if(bWriteToFile) {
            m_matlab_file << "t_newCPU=" << m_t_newCPU.transpose().format(Matlab)<<"';" <<std::endl;
            m_matlab_file << "x_newCPU=" << m_x_newCPU.transpose().format(Matlab)<<"';" <<std::endl;
            m_matlab_file.close();
        }
    }

    void checkResults() {

        double relTolGPUCPU = 1e-5;
        unsigned int tolUlpGPUCPU = 20;

        //TODO Eliminate warning???
        bool b1,b2,b3,b4;
        std::tie(b1,b2,b3,b4) = Utilities::compareArraysEachCombined(m_x_newGPU.data(),m_x_newCPU.data(),m_x_newGPU.rows(),(PREC)relTolGPUCPU,tolUlpGPUCPU,m_maxRelTol,m_avgRelTol, m_maxUlp,m_avgUlp,false);

        if(b1 && b2 && b3 && b4 ){
         *m_pLog << " ---> GPU/CPU identical!...." << std::endl;
        }else{
         *m_pLog << " ---> GPU/CPU NOT identical!...." << std::endl;
        }
        *m_pLog << " ---> Converged relTol: "<<b1  <<" \t Identical Ulp: "<< b2
                << "      CPU finite: "<<b3  <<" \t GPU finite: "<< b4 << std::endl;




        *m_pLog << " ---> maxUlp :" << (double)m_maxUlp << std::endl;
        *m_pLog << " ---> avgUlp :" << m_avgUlp << std::endl;
        *m_pLog << " ---> maxRelTol :" << m_maxRelTol << std::endl;
        *m_pLog << " ---> avgRelTol :" << m_avgRelTol << std::endl;

    }

    void writeData() {
        tinyformat::format(*m_pData,"%1$.9d\t", m_nContacts);
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
    int m_MG, m_NG;

    int m_nContactCounter;
    int m_nRandomRunsCounter;

    std::ostream* m_pData;
    std::ostream* m_pLog;

    std::ofstream m_matlab_file;
    Eigen::IOFormat Matlab;

    Eigen::Matrix<PREC,Eigen::Dynamic, Eigen::Dynamic> m_G;
    Eigen::Matrix<PREC,Eigen::Dynamic, 1> m_c;
    Eigen::Matrix<PREC,Eigen::Dynamic, 1> m_t;
    Eigen::Matrix<PREC,Eigen::Dynamic, 1> m_t_newCPU;
    Eigen::Matrix<PREC,Eigen::Dynamic, 1> m_t_newGPU;
    Eigen::Matrix<PREC,Eigen::Dynamic, Eigen::Dynamic> m_T;
    Eigen::Matrix<PREC,Eigen::Dynamic, 1> m_d;
    Eigen::Matrix<PREC,Eigen::Dynamic, 1> m_x_old;
    Eigen::Matrix<PREC,Eigen::Dynamic, 1> m_x_newCPU;
    Eigen::Matrix<PREC,Eigen::Dynamic, 1> m_x_newGPU;

    typename TGaussSeidelBlockSettings::TGPUVariant m_gpuVariant;
};



/**
* @defgroup JorGaussSeidelBlockGPUVariant Jor GaussSeidelBlock GPUVariants
* @detailed VariantId specifies which variant is launched:
* Here the different variants have been included in one class!
* To be more flexible we can also completely reimplement the whole class for another GPUVariant!
* @{
*/

#include "CudaFramework/Kernels/GaussSeidelGPU/GaussSeidelGPU.hpp"



template<typename _PREC, int _VariantId, int _nValuesPerContact, int _alignMatrix, int _nMaxIterations,int _nBlockSize, bool _bAbortIfConverged, int _nCheckConvergedFlag>
struct GaussSeidelBlockGPUVariantSettingsWrapper {
    typedef _PREC PREC;
    static const int VariantId = _VariantId;
    static const int alignMatrix = _alignMatrix;
    static const int nBlockSize = _nBlockSize;
    static const int nMaxIterations = _nMaxIterations;
    static const int nValuesPerContact = _nValuesPerContact;
    static const bool bAbortIfConverged = _bAbortIfConverged;
    static const int nCheckConvergedFlag = _nCheckConvergedFlag;
};

#define DEFINE_GaussSeidelBlockGPUVariant_Settings(_TSettings_) \
   typedef typename _TSettings_::PREC PREC; \
   static const int VariantId = _TSettings_::VariantId; \
   static const int alignMatrix = _TSettings_::alignMatrix; \
   static const int nMaxIterations = _TSettings_::nMaxIterations; \
   static const int nBlockSize = _TSettings_::nBlockSize; \
   static const int nValuesPerContact = _TSettings_::nValuesPerContact; \
   static const bool bAbortIfConverged = _TSettings_::bAbortIfConverged; \
   static const int nCheckConvergedFlag = _TSettings_::nCheckConvergedFlag;

template<typename TGaussSeidelBlockGPUVariantSettings> class GaussSeidelBlockGPUVariant; // Prototype

template<int _VariantId, bool _alignMatrix> // Settings from below
struct GaussSeidelBlockGPUVariantSettings {

    template <typename _PREC, int _nValuesPerContact, int _nMaxIterations, int _nBlockSize, bool _bAbortIfConverged, int _nCheckConvergedFlag> // Settings from above
    struct GPUVariant {
        typedef GaussSeidelBlockGPUVariant< GaussSeidelBlockGPUVariantSettingsWrapper<_PREC,_VariantId,_nValuesPerContact,_alignMatrix,_nMaxIterations,_nBlockSize,_bAbortIfConverged, _nCheckConvergedFlag> >  TGPUVariant;
    };

};

template<typename TGaussSeidelBlockGPUVariantSettings>
class GaussSeidelBlockGPUVariant {
public:

    DEFINE_GaussSeidelBlockGPUVariant_Settings(TGaussSeidelBlockGPUVariantSettings);

    static std::string getVariantName() {
        return getVariantNameInternal<VariantId>();
    }
    static std::string getVariantDescription() {
        return getVariantDescriptionInternal<VariantId>() + std::string(", aligned: ") + ((alignMatrix)? std::string("on") : std::string("off"));
    }


    bool checkSettings(unsigned int gpuID){
        return true;
    }


    template<int VariantId> static std::string getVariantNameInternal() {
        switch(VariantId) {
        case 1:
            return "[Gauss Seidel Kernel A][Gauss Seidel Kernel B]";
        case 2:
            return "[Gauss Seidel Kernel A with Conv. Check][Gauss Seidel Kernel B]";
        }
    }
    template<int VariantId> static std::string getVariantDescriptionInternal() {
        switch(VariantId) {
        case 1:
            return "[(64 x 64) Block with 64 Threads (Diagonal Dependencies)][(MG - 64) rows, in each 64 Threads, (Propagate)]";
        case 2:
            return "[(64 x 64) Block with 64 Threads (Diagonal Dependencies)][(MG - 64) rows, in each 64 Threads, (Propagate)]";
        }
    }




    void initialize(std::ostream* pLog) {
        m_pLog = pLog;
        CHECK_CUBLAS(cublasCreate(&m_cublasHandle));
        CHECK_CUDA(cudaEventCreate(&m_startKernel));
        CHECK_CUDA(cudaEventCreate(&m_stopKernel));
        CHECK_CUDA(cudaEventCreate(&m_startCopy));
        CHECK_CUDA(cudaEventCreate(&m_stopCopy));
    }

    void finalize() {
        CHECK_CUBLAS(cublasDestroy(m_cublasHandle));
        CHECK_CUDA(cudaEventDestroy(m_startKernel));
        CHECK_CUDA(cudaEventDestroy(m_stopKernel));
        CHECK_CUDA(cudaEventDestroy(m_startCopy));
        CHECK_CUDA(cudaEventDestroy(m_stopCopy));
    }

    template<typename Derived1, typename Derived2>
    void initializeTestProblem( const Eigen::MatrixBase<Derived2> &T, const Eigen::MatrixBase<Derived1> & x_old,const Eigen::MatrixBase<Derived1> & d, const Eigen::MatrixBase<Derived1> & t ) {

        size_t freeMem;
        size_t totalMem;

        CHECK_CUDA(cudaMemGetInfo (&freeMem, &totalMem));

        int nGPUBytes = (2*x_old.rows() + d.rows() + t.rows() +  T.rows()*T.cols())*sizeof(PREC);
        *m_pLog << "Will try to allocate ("<<nGPUBytes<<"/"<<freeMem<<") = " << (double)nGPUBytes/freeMem * 100.0 <<" % of global memory on GPU"<<std::endl;
        if(nGPUBytes > freeMem) {
            *m_pLog <<"To little memory on GPU, exit!..."<<std::endl;
            std::exit(-1);
        }

        CHECK_CUDA(utilCuda::mallocMatrixDevice<alignMatrix>(T_dev, T));
        CHECK_CUDA(utilCuda::mallocMatrixDevice<false>(d_dev, d));
        CHECK_CUDA(utilCuda::mallocMatrixDevice<false>(x_old_dev,x_old));
        CHECK_CUDA(utilCuda::mallocMatrixDevice<false>(t_dev,t));
        CHECK_CUDA(utilCuda::mallocMatrixDevice<false>(x_new_dev,x_old_dev.m_M,x_old_dev.m_N));
        CHECK_CUDA(cudaMalloc(&pConvergedFlag_dev,sizeof(bool)));
    }

    template<typename Derived1, typename Derived2>
    void run(Eigen::MatrixBase<Derived1> & x_newGPU, Eigen::MatrixBase<Derived1> & t_newGPU,  const Eigen::MatrixBase<Derived2> &T, const Eigen::MatrixBase<Derived1> & x_old, const Eigen::MatrixBase<Derived1> & d, const Eigen::MatrixBase<Derived1> & t) {

        //Copy Data
        CHECK_CUDA(cudaEventRecord(m_startCopy,0));
        copyMatrixToDevice(T_dev, T);
        copyMatrixToDevice(d_dev, d);
        copyMatrixToDevice(x_old_dev,x_old);
        copyMatrixToDevice(t_dev,t);
        CHECK_CUDA(cudaEventRecord(m_stopCopy,0));
        CHECK_CUDA(cudaEventSynchronize(m_stopCopy));

        float time;
        CHECK_CUDA( cudaEventElapsedTime(&time,m_startCopy,m_stopCopy));
        m_elapsedTimeCopyToGPU = time;
        *m_pLog << " ---> Copy time to GPU:"<< tinyformat::format("%1$8.6f ms",time) <<std::endl;


        *m_pLog << " ---> Iterations started..."<<std::endl;

        m_absTOL = 1e-8;
        m_relTOL = 1e-10;

        CHECK_CUDA(cudaEventRecord(m_startKernel,0));

        runKernel<VariantId>();

        CHECK_CUDA(cudaEventRecord(m_stopKernel,0));
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventSynchronize(m_stopKernel));


        CHECK_CUDA(cudaThreadSynchronize());
        *m_pLog<<" ---> Iterations finished" << std::endl;
        CHECK_CUDA( cudaEventElapsedTime(&time,m_startKernel,m_stopKernel));
        double average = (time/(double)nMaxIterations);
        m_gpuIterationTime = average;

        *m_pLog << " ---> GPU Iteration time :"<< tinyformat::format("%1$8.6f ms",average) <<std::endl;
        *m_pLog << " ---> nIterations: " << m_nIter <<std::endl;
        if (m_nIter == nMaxIterations) {
            *m_pLog << " ---> Max. Iterations reached."<<std::endl;
        }

        // Copy results back
        CHECK_CUDA(cudaEventRecord(m_startCopy,0));
        CHECK_CUDA(utilCuda::copyMatrixToHost(x_newGPU,x_old_dev));
        CHECK_CUDA(utilCuda::copyMatrixToHost(t_newGPU,t_dev));
        CHECK_CUDA(cudaEventRecord(m_stopCopy,0));
        CHECK_CUDA(cudaEventSynchronize(m_stopCopy));
        CHECK_CUDA( cudaEventElapsedTime(&time,m_startCopy,m_stopCopy));
        m_elapsedTimeCopyFromGPU = time;
        *m_pLog << " ---> Copy time from GPU:"<< tinyformat::format("%1$8.6f ms", time)<<std::endl;
    }

    template<int VariantId> inline void runKernel() {

        if(VariantId==1) {

            ASSERTMSG( T_dev.m_M % nBlockSize == 0, " Gauss Seidel Block works only with T as a multiple of nBlockSize" );
            int g = T_dev.m_M / nBlockSize;


            for (m_nIter=0; m_nIter< nMaxIterations ; m_nIter++) {
                // First set the converged flag to 1
                cudaMemset(pConvergedFlag_dev,1,sizeof(bool));

                // Do one Gauss Seidel Step!

                // First set the converged flag to 1
                cudaMemset(pConvergedFlag_dev,1,sizeof(bool));
                // Do one step
                for(int j_g = 0 ; j_g < g; j_g++) {
                    gaussSeidelGPU::blockGaussSeidelStepACorrectNoDivision_kernelWrap(T_dev,d_dev,t_dev,x_old_dev, j_g);
                    gaussSeidelGPU::blockGaussSeidelStepB_kernelWrap(T_dev,d_dev,t_dev,x_old_dev, j_g);
                }

            }
        } else if(VariantId==2) {

            ASSERTMSG( T_dev.m_M % nBlockSize == 0, " Gauss Seidel Block works only with T as a multiple of nBlockSize" );
            int g = T_dev.m_M / nBlockSize;

            for (m_nIter=0; m_nIter< nMaxIterations ; m_nIter++) {
                // First set the converged flag to 1
                cudaMemset(pConvergedFlag_dev,1,sizeof(bool));

                // Do one Gauss Seidel Step!

                // First set the converged flag to 1
                cudaMemset(pConvergedFlag_dev,1,sizeof(bool));
                // Do one step
                for(int j_g = 0 ; j_g < g; j_g++) {

                    gaussSeidelGPU::blockGaussSeidelStepACorrectNoDivision_kernelWrap(T_dev,d_dev,t_dev,x_old_dev, j_g, pConvergedFlag_dev,    (PREC)m_absTOL,(PREC)m_relTOL);
                    gaussSeidelGPU::blockGaussSeidelStepB_kernelWrap(T_dev,d_dev,t_dev,x_old_dev, j_g);

                }


                if(bAbortIfConverged) {
                    // Check each nCheckConvergedFlag  the converged flag
                    if(m_nIter % nCheckConvergedFlag == 0) {
                        // Download the flag from the GPU and check
                        cudaMemcpy(&m_bConvergedFlag,pConvergedFlag_dev,sizeof(bool),cudaMemcpyDeviceToHost);
                        if(m_bConvergedFlag == true) {
                            // Converged
                            break;
                        }
                    }
                }
            }
        }
    }


    void cleanUpTestProblem() {

        CHECK_CUDA(freeMatrixDevice(T_dev));
        CHECK_CUDA(freeMatrixDevice(d_dev));
        CHECK_CUDA(freeMatrixDevice(x_old_dev));
        CHECK_CUDA(freeMatrixDevice(x_new_dev));
        CHECK_CUDA(freeMatrixDevice(t_dev));
        CHECK_CUDA(cudaFree(pConvergedFlag_dev));
    }

    double m_gpuIterationTime;
    double m_elapsedTimeCopyToGPU;
    double m_elapsedTimeCopyFromGPU;
    int m_nIter;

private:

    utilCuda::CudaMatrix<PREC> T_dev, d_dev, x_old_dev, x_new_dev, t_dev;
    bool *pConvergedFlag_dev;
    bool m_bConvergedFlag;

    double m_absTOL, m_relTOL;


    cublasHandle_t m_cublasHandle;
    cudaEvent_t m_startKernel, m_stopKernel, m_startCopy,m_stopCopy;

    std::ostream* m_pLog;
};




/** @} */



/** @} */ // GaussSeidelBlockGPUVariant


/** @} */ // addtogroup TestVariants

/** @} */ // addtogroup KernelTestMethod

#endif
