// ========================================================================================
//  CudaFramework
//  Copyright (C) 2014 by Gabriel Nützi <nuetzig (at) imes (d0t) mavt (d0t) ethz (d0t) ch>
//
//  This Source Code Form is subject to the terms of the GNU GPL 3.0 licence.
//  If a copy of the GNU GPL 3.0 was not distributed with this
//  file, you can obtain one at http://opensource.org/licenses/GPL-3.0.
// ========================================================================================

#ifndef CudaFramework_Kernels_ProxGPU_ProxTestVariant_hpp
#define CudaFramework_Kernels_ProxGPU_ProxTestVariant_hpp

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include <algorithm>

#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "CudaFramework/General/StaticAssert.hpp"
#include "CudaFramework/General/TypeTraitsHelper.hpp"
#include "CudaFramework/CudaModern/CudaError.hpp"
#include "CudaFramework/General/Utilities.hpp"
#include "CudaFramework/CudaModern/CudaMatrix.hpp"
#include "CudaFramework/CudaModern/CudaMatrixUtilities.hpp"

#include "ConvexSets.hpp"

#include "CudaFramework/General/FloatingPointType.hpp"

#include <Eigen/Dense>

#include "CudaFramework/Kernels/ProxGPU/ProxSettings.hpp"

#include "CudaFramework/Kernels/ProxGPU/SorProxGPUVariant.hpp"
#include "CudaFramework/Kernels/ProxGPU/JorProxGPUVariant.hpp"


// SHIT WINDOWS DEFINES! FUCKIT
#ifdef min
#undef min
#endif

#ifdef max
#undef max
#endif

/**
* @addtogroup KernelTestMethod
* @{
*/


/**
* @addtogroup TestVariants
* @{
*/


/**
* @defgroup ProxTestVariant Prox Test Variant
* This provides a Test Variant for Jor or Sor Prox.
*
* How to launch this Test Variant:
* @code

typedef KernelTestMethod<
KernelTestMethodSettings<false,true> ,
ProxTestVariant<
ProxSettings<
double,
3,
false,
10,
false,
10,
ProxPerformanceTestSettings<4000,20,4000,5>,
JorProxGPUVariantSettings<5,true>
>
>
> test1;
PerformanceTest<test1> A("test1");
A.run();
* @endcode

* @{
*/





struct SeedGenerator {
    template<int _seed = 5>
    struct Fixed {
        static const int seed = _seed;
        static unsigned int getSeed() {
            return seed;
        };
    };
    struct Time {
        static unsigned int getSeed() {
            return (unsigned int)time(NULL);
        };
    };
};


template<typename _PREC,
         typename _TSeedGenerator,
         bool _bwriteToFile,
         int _nMaxIterations,
         bool _bAbortIfConverged,
         int _nCheckConvergedFlag,
         bool _bMatchCPUToGPU,
         typename _TPerformanceTestSettings,
         typename _TGPUVariantSetting >
struct ProxSettings {

    typedef  _PREC PREC;
    typedef   _TSeedGenerator TSeedGenerator;
    static const bool bWriteToFile = _bwriteToFile;
    static const int nMaxIterations = _nMaxIterations;
    static const bool bAbortIfConverged = _bAbortIfConverged;
    static const int nCheckConvergedFlag = _nCheckConvergedFlag;
    static const bool bMatchCPUToGPU = _bMatchCPUToGPU;
    typedef typename _TGPUVariantSetting::TProxIterationType TProxIterationType;    // Sor/Jor
    typedef typename _TGPUVariantSetting::TConvexSet TConvexSet;                    // Rplus/RPlusAndDisk/RPlusAndContensouEllipsoid
    typedef _TPerformanceTestSettings TPerformanceTestSettings;
    typedef typename _TGPUVariantSetting::template GPUVariant<PREC,nMaxIterations,bAbortIfConverged,nCheckConvergedFlag, bMatchCPUToGPU >::TGPUVariant TGPUVariant;
};


#define DEFINE_Prox_Settings(_TSettings_) \
   typedef typename _TSettings_::PREC PREC; \
   typedef typename _TSettings_::TSeedGenerator TSeedGenerator; \
   static const bool bWriteToFile = _TSettings_::bWriteToFile; \
   static const int nMaxIterations = _TSettings_::nMaxIterations; \
   static const bool bAbortIfConverged = _TSettings_::bAbortIfConverged; \
   static const int nCheckConvergedFlag = _TSettings_::nCheckConvergedFlag; \
   static const bool bMatchCPUToGPU = _TSettings_::bMatchCPUToGPU; \
   typedef typename _TSettings_::TProxIterationType TProxIterationType;  \
   typedef typename _TSettings_::TConvexSet TConvexSet; \
   typedef typename _TSettings_::TPerformanceTestSettings TPerformanceTestSettings; \
   typedef typename _TSettings_::TGPUVariant TGPUVariant;


template< int _minNContacts, int _stepNContacts, int _maxNContacts, int _maxNRandomRuns >
struct ProxPerformanceTestSettings {
    static const int minNContacts = _minNContacts;
    static const int stepNContacts = _stepNContacts;
    static const int maxNContacts = _maxNContacts;
    static const int maxNRandomRuns = _maxNRandomRuns;
};
#define DEFINE_ProxPerformanceTestSettings_Settings(_TSettings_)  \
   static const int  minNContacts = _TSettings_::minNContacts; \
   static const int  stepNContacts = _TSettings_::stepNContacts; \
   static const int  maxNContacts = _TSettings_::maxNContacts; \
   static const int  maxNRandomRuns = _TSettings_::maxNRandomRuns;



template<typename TProxSettings>
class ProxTestVariant {
public:


    DEFINE_Prox_Settings(TProxSettings);
    DEFINE_ProxPerformanceTestSettings_Settings(TPerformanceTestSettings);

    static std::string getTestVariantDescription() {
        return "Prox";
    }

    ProxTestVariant()
        :Matlab(Eigen::FullPrecision, 0, ", ", ";\n", "", "", "[", "]") {
        m_nContactCounter = 0;
        m_nRandomRunsCounter =0;
    }
    ~ProxTestVariant() {
        m_matlab_file.close();
    }

    void initialize(std::ostream * pLog, std::ostream * pData) {

        m_pData = pData;
        m_pLog = pLog;



        m_nContactCounter = minNContacts;
        m_nRandomRunsCounter =0;


        std::srand ( TSeedGenerator::getSeed() );


        if(bWriteToFile) {
            m_matlab_file.close();
            m_matlab_file.open("ProxTest.m_M", std::ios::trunc | std::ios::out);
            m_matlab_file.clear();
            if (m_matlab_file.is_open()) {
                std::cout << " File opened: " << "ProxTest.m_M"<<std::endl;
            }

        }

        m_relTolGPUCPU = 1e-5;
        m_tolUlpGPUCPU = 20;


        m_gpuVariant.initialize(m_pLog,&m_matlab_file);



    }

    std::vector<std::pair<std::string,std::string> >
    getDescriptions() {
        *m_pLog  << "Kernel used: \t\t"<< TGPUVariant::getVariantName()<<std::endl;

        std::vector<std::pair<std::string,std::string> > s;

        s.push_back( std::make_pair("Remarks" ,"This text should explain the description to remember" ));

        s.push_back( std::make_pair("ProxIterationType",Utilities::cutStringTillScope(typeid(TProxIterationType).name()) ));
        s.push_back( std::make_pair("VariantID",std::to_string(TGPUVariant::VariantId)));
        s.push_back( std::make_pair("VariantName", TGPUVariant::getVariantName() ));
        s.push_back( std::make_pair("VariantDescription", TGPUVariant::getVariantDescriptionShort()));
        s.push_back( std::make_pair("VariantSettings" , TGPUVariant::getVariantDescriptionLong()));
        s.push_back( std::make_pair("MatchCPUGPU" , ((bMatchCPUToGPU)? std::string("on") : std::string("off"))));

        s.push_back( std::make_pair("ConvexSet",  Utilities::cutStringTillScope(typeid(typename TGPUVariant::TConvexSet).name())));
        s.push_back( std::make_pair("TotalProxIterations", std::to_string(nMaxIterations)));
        s.push_back( std::make_pair("TotalTestProblemsPerContactSize", std::to_string(maxNRandomRuns)));
        s.push_back( std::make_pair("Precision", typeid(PREC).name()));

        return s;
    }


    std::vector<std::string>
    getDataColumHeader(){
        std::vector<std::string> s;
        s.push_back("nContacts");
        return s;
    }

    bool checkSettings(unsigned int gpuID){
        return m_gpuVariant.checkSettings(gpuID);
    }

    bool generateNextTestProblem() {



        if(m_nContactCounter>maxNContacts) {
            *m_pLog << "No more Test Problems to generate, --->exit ============="<<std::endl;
            return false;
        }

        // Set up next test problem
        if (m_nContactCounter <= 0) {
            m_nContacts = 1;
        } else {
            m_nContacts = m_nContactCounter;
        }

        *m_pLog << "Compute test for nContacts: "<< m_nContacts <<" ============="<<std::endl;
        m_MG = m_nContacts*TConvexSet::Dimension;
        m_NG = m_MG;




        m_G.resize(m_MG,m_NG);
        m_c.resize(m_MG);
        m_mu.resize(m_nContacts);
        m_R.resize(m_MG);
        m_T.resize(m_MG,m_NG);
        m_d.resize(m_MG);
        m_x_old.resize(m_MG);
        m_x_newCPU.resize(m_MG);
        m_x_newGPU.resize(m_MG);


        //reset randomRun
        m_nRandomRunsCounter = 0;

        m_gpuVariant.initializeTestProblem(m_T,m_x_old,m_d);

        // Increment counter
        m_nContactCounter += stepNContacts;


        m_nOps = m_gpuVariant.getNOps();
        m_nBytesReadWrite = m_gpuVariant.getBytesReadWrite();

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
        m_G.diagonal().array() += 4;
        m_c.setRandom();

        PREC r_T_i;
        PREC alpha = 0.5;
        for(int i=0; i<m_nContacts; i++) {
            m_R((2+1)*i) =  alpha / m_G((2+1)*i,(2+1)*i);
            r_T_i = alpha / m_G.diagonal().template segment<2>((2+1)*i+1).maxCoeff();
            m_R((2+1)*i+1) = r_T_i;
            m_R((2+1)*i+2) = r_T_i;
        }

        m_mu.setConstant(0);
        m_T = (  Eigen::Matrix<PREC,Eigen::Dynamic, Eigen::Dynamic>::Identity(m_MG,m_NG) - m_R.asDiagonal()* m_G );
        m_T_RM = m_T;
        m_d = m_R.asDiagonal() * m_c * -1;

        if(TypeTraitsHelper::IsSame<TProxIterationType,ProxIterationType::JOR>::result) {
            m_x_old.setRandom();
        } else {
            m_x_old.setZero();
            //m_x_old.setConstant(0.5);
        }
        // ==========================================


        if(bWriteToFile) {
            m_matlab_file << "T=" << m_T.format(Matlab)<<";" <<std::endl;
            m_matlab_file << "d=" << m_d.transpose().format(Matlab)<<"';" <<std::endl;
            m_matlab_file << "x_old=" << m_x_old.transpose().format(Matlab)<<"';" <<std::endl;
            m_matlab_file << "mu=" << m_mu.transpose().format(Matlab)<<"';" <<std::endl;
        }


        return true;
    }

    void runOnGPU() {

        //Select variant
        // TODO doo here a DISPATCHING (with dispatching helper class, which is specialized for the convex set type), according to the convexSet type we need to call a different function, the
        // m_gpuVariant has then only the run method with the correct number of arguments, and this one also gets called!, this ProxTestVariant class provides all data for all convexset types, but the underlying
        // GPUVariant is different
        m_gpuVariant.runGPUProfile(m_x_newGPU,m_T,m_x_old,m_d,m_mu);

        m_elapsedTimeCopyToGPU     = m_gpuVariant.m_elapsedTimeCopyToGPU;
        m_gpuIterationTime         = m_gpuVariant.m_gpuIterationTime;
        m_elapsedTimeCopyFromGPU   = m_gpuVariant.m_elapsedTimeCopyFromGPU;
        m_nIterGPU                 = m_gpuVariant.m_nIterGPU;

        if(bWriteToFile) {
            m_matlab_file << "x_newGPU=" << m_x_newGPU.transpose().format(Matlab)<<"';" <<std::endl;
        }

    }

    void runOnCPU() {

        // Do Jor Scheme!

        if(TypeTraitsHelper::IsSame<TProxIterationType,ProxIterationType::JOR>::result) {
            m_gpuVariant.runCPUEquivalentProfile(m_x_newCPU,m_T,m_x_old,m_d,m_mu);
        } else if(TypeTraitsHelper::IsSame<TProxIterationType,ProxIterationType::SOR>::result) {
            m_gpuVariant.runCPUEquivalentProfile(m_x_newCPU,m_T_RM,m_x_old,m_d,m_mu);
        }


        m_cpuIterationTime         = m_gpuVariant.m_cpuIterationTime;
        m_nIterCPU                 = m_gpuVariant.m_nIterCPU;

        if(bWriteToFile) {
            m_matlab_file << "x_newCPU=" << m_x_newCPU.transpose().format(Matlab)<<"';" <<std::endl;
        }

    }

    void checkResults() {
        bool b1,b2,b3,b4;
        std::tie(b1,b2,b3,b4) = Utilities::compareArraysEachCombined(m_x_newGPU.data(),
                                m_x_newCPU.data(),
                                m_x_newGPU.rows(),
                                (PREC)m_relTolGPUCPU,
                                m_tolUlpGPUCPU,
                                m_maxRelTol,
                                m_avgRelTol,
                                m_maxUlp,
                                m_avgUlp);
        if(b1 && b2 && b3 && b4 ) {
            *m_pLog << " ---> GPU/CPU identical!...." << std::endl;
        } else {
            *m_pLog << " ---> GPU/CPU NOT identical!...." << std::endl;
        }
        *m_pLog << " ---> Converged relTol: "<<b1  <<" \t Identical Ulp: "<< b2
                << "      CPU finite: "<<b3  <<" \t GPU finite: "<< b4 << std::endl;

        //ASSERTMSG(result.first == true,"ABORT");

        *m_pLog << " ---> maxUlp :" << (double)m_maxUlp << std::endl;
        *m_pLog << " ---> avgUlp :" << m_avgUlp << std::endl;
        *m_pLog << " ---> maxRelTol :" << m_maxRelTol << std::endl;
        *m_pLog << " ---> avgRelTol :" << m_avgRelTol << std::endl;

    }

    void writeData() {
      tinyformat::format(*m_pData ,"%1$.9d\t", m_nContacts);
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

    // GPU / CPU Check
    double m_relTolGPUCPU;
    unsigned int m_tolUlpGPUCPU;

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

    PREC m_alpha;
    PREC m_beta;
    Eigen::Matrix<PREC,Eigen::Dynamic, Eigen::Dynamic> m_G;
    Eigen::Matrix<PREC,Eigen::Dynamic, 1> m_R;
    Eigen::Matrix<PREC,Eigen::Dynamic, 1> m_c;
    Eigen::Matrix<PREC,Eigen::Dynamic, 1> m_mu;
    Eigen::Matrix<PREC,Eigen::Dynamic, Eigen::Dynamic> m_T;
    Eigen::Matrix<PREC,Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> m_T_RM;  // This is only used to not fuck up the cach during CPU Prox iteration!
    Eigen::Matrix<PREC,Eigen::Dynamic, 1> m_d;
    Eigen::Matrix<PREC,Eigen::Dynamic, 1> m_x_old;
    Eigen::Matrix<PREC,Eigen::Dynamic, 1> m_x_newCPU;
    Eigen::Matrix<PREC,Eigen::Dynamic, 1> m_x_newGPU;

    typename TProxSettings::TGPUVariant m_gpuVariant;
};




/** @} */ // PrxTestVariant


/** @} */ // addtogroup TestVariants

/** @} */ // addtogroup KernelTestMethod

#endif
