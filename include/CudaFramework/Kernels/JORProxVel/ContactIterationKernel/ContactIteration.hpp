// ========================================================================================
//  CudaFramework
//  Copyright (C) 2014 by Gabriel Nützi <nuetzig (at) imes (d0t) mavt (d0t) ethz (d0t) ch>
//
//  This Source Code Form is subject to the terms of the GNU GPL 3.0 licence. 
//  If a copy of the GNU GPL 3.0 was not distributed with this
//  file, you can obtain one at http://opensource.org/licenses/GPL-3.0.
// ========================================================================================

#ifndef CudaFramework_Kernels_JORProxVel_ContactIterationKernel_ContactIteration_hpp
#define CudaFramework_Kernels_JORProxVel_ContactIterationKernel_ContactIteration_hpp

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <random>

#include <boost/format.hpp>
#include "CudaFramework/CudaModern/CudaMatrixUtilities.hpp"


#include "CudaFramework/General/AssertionDebug.hpp"
#include "CudaFramework/CudaModern/CudaError.hpp"
#include <Eigen/Dense>
#include "CudaFramework/Kernels/JORProxVel/ContactIterationKernel/ContactIterationKernelWrap.hpp"
#include "CudaFramework/General/FloatingPointType.hpp"
#include "CudaFramework/CudaModern/CudaMatrix.hpp"
#include "CudaFramework/Kernels/JORProxVel/GPUBufferOffsets.hpp"
#include "CudaFramework/Kernels/JORProxVel/VariantLaunchSettings.hpp"
#include "CudaFramework/Kernels/JORProxVel/GenRandomContactGraphClass.hpp"





//Prototyp
template<typename PREC, unsigned int TVariantId> class ContIterGPUVariant;

/**
* @brief Test range settings
*/
template<unsigned int _VariantId>
struct ContIterGPUVariantSettings {

    template <typename _PREC> // Settings from above
    struct GPUVariant {
        typedef ContIterGPUVariant< _PREC,_VariantId  >  GPUVariantType;
    };

};


/**
* @brief Test range settings
*/
template< unsigned int _minNContacts,
          unsigned int _maxNContacts,
          unsigned int _stepNContacts,
          unsigned int _maxNRandomRuns >
struct ContIterTestRangeSettings {
    static const int minNContacts = _minNContacts;
    static const int stepNContacts = _stepNContacts;
    static const int maxNContacts = _maxNContacts;
    static const int maxNRandomRuns = _maxNRandomRuns;
};
#define DEFINE_ContIterTestRangeSettings( _TSettings_ )  \
   static const int  minNContacts = _TSettings_::minNContacts; \
   static const int  stepNContacts = _TSettings_::stepNContacts; \
   static const int  maxNContacts = _TSettings_::maxNContacts; \
   static const int  maxNRandomRuns = _TSettings_::maxNRandomRuns;


/**
* @brief Settings for ContIterTestVariant
*/
template<typename _PREC, typename TTestRangeSettings, typename TGPUVariantSettings >
struct ContIterSettings {
    typedef _PREC PREC;
    typedef TTestRangeSettings TestRangeSettingsType;
    typedef TGPUVariantSettings GPUVariantSettingsType;

    typedef typename GPUVariantSettingsType::template GPUVariant<PREC>::GPUVariantType GPUVariantType;  // ContIterGPUVariant<double,1>
};

#define DEFINE_ContIterSettings( __Settings__ ) \
    typedef typename __Settings__::PREC PREC; \
    typedef typename __Settings__::TestRangeSettingsType TestRangeSettingsType; \
    typedef typename __Settings__::GPUVariantSettingsType GPUVariantSettingsType; \
    typedef typename __Settings__::GPUVariantType GPUVariantType; \
    DEFINE_ContIterTestRangeSettings( TestRangeSettingsType )



/**
* @brief TestVariant which computes a contact frame for all contacts.
*/
template<typename TSettings >
class ContIterTestVariant {

public:

    typedef TSettings SettingsType;

    DEFINE_ContIterSettings( TSettings )

    typename SettingsType::GPUVariantType m_gpuVariant;

    /// typename PREC

    /// typename TestRangeSettingsType
        /// static const int  minNContacts
        /// static const int  stepNContacts
        /// static const int  maxNContacts
        /// static const int  maxNRandomRuns

    /// typename GPUVariantSettingsType
        /// typedef ContIterGPUVariant< _PREC,_VariantId  >  GPUVariantType


    ContIterTestVariant();

private:

    using  RandomGeneratorType = std::mt19937;  // the Mersenne Twister with a popular choice of parameters
    using  DistributionType = std::uniform_real_distribution<PREC>;
    const uint32_t m_seed =  30;


    const double Tolerance=0.000001;

    typedef Eigen::Matrix<PREC,Eigen::Dynamic, Eigen::Dynamic,Eigen::ColMajor> MatrixType;
    typedef Eigen::Matrix<unsigned int,Eigen::Dynamic, Eigen::Dynamic,Eigen::ColMajor>  MatrixUIntType;
    typedef std::vector<PREC> VectorType;
    typedef std::vector<int> VectorIntType;
    typedef GenRndContactGraph<VectorIntType,MatrixType,MatrixUIntType> RndGraph;

    MatrixType m_inputMatrixGPU;
    MatrixType m_inputMatrixCPU;


    ///================///

    MatrixType m_bodyBufferGPU;
    MatrixType m_contBufferGPU;
    MatrixUIntType m_globalBufferGPU;
    VectorType m_reductionBufferGPU;
    MatrixUIntType m_indexSetGPU;

    MatrixType m_bodyBufferCPU;
    MatrixType m_contBufferCPU;
    MatrixUIntType m_globalBufferCPU;
    VectorType m_reductionBufferCPU;
    MatrixUIntType m_indexSetCPU;

    ///================///


    MatrixType m_outputMatrixCPU;
    MatrixType m_outputMatrixGPU;

    VectorIntType m_csrGPUDummy;

    const unsigned int m_implementedLengthInput=62;



    ///================///

    const unsigned int m_bodyBufferLength       = JORProxVelGPU::GPUBufferOffsets::BodyBufferOffsets::length;
    const unsigned int m_contactBufferLength    = JORProxVelGPU::GPUBufferOffsets::ContBufferOffsets::length;
    const unsigned int m_redBufferLength        = JORProxVelGPU::GPUBufferOffsets::ReductionBufferOffsets::length;
    const unsigned int m_globalBufferLength = JORProxVelGPU::GPUBufferOffsets::GlobalBufferOffsets::length;
    const unsigned int m_indexSetLength = JORProxVelGPU::GPUBufferOffsets::IndexBufferOffsets::length;

    ///================///

    const unsigned int m_implementedLengthOutput=15;

    unsigned int m_numberOfContacts;
    unsigned int m_numberOfBodies;
    unsigned int m_rowsInGlobalBuffer=1;
    unsigned int m_numberOfReductions=1;

    int m_nContacts;
    int m_nContactCounter;
    int m_nRandomRunsCounter;

    std::ostream* m_pData;
    std::ostream* m_pLog;

public:

    double m_nBytesReadWrite;
    double m_nOps;
    double m_elapsedTimeCopyToGPU;
    double m_elapsedTimeCopyFromGPU;
    double m_cpuIterationTime;
    double m_gpuIterationTime;

    double m_maxRelTol=5E-6;
    double m_avgRelTol=5E-6;
    typename TypeWithSize<sizeof(PREC)>::UInt m_maxUlp;
    double m_avgUlp;



    static std::string getTestVariantDescription()
    {
        return "ContactIteration";
    }


public:

    void initialize(std::ostream * pLog, std::ostream * pData);
    /** Check settings at runtime, static settings are already checked at compile time*/
    bool checkSettings(int gpuID){WARNINGMSG(false,"checkSettings not correctly implemented!"); return true;}
    bool generateNextTestProblem();
    bool generateNextRandomRun();

    void runOnGPU();
    void runOnCPU();
    void checkResults();
    void cleanUpTestProblem();
    void writeData();
    void finalize() {}
    bool compareOutput();
    bool isEqual(PREC,PREC);
   /// void contactiterationCPU(std::vector<Data> &dataList);

    std::vector<std::pair<std::string,std::string> >
    getDescriptions() {
        *m_pLog  << "Kernel used: \t\t"<< GPUVariantType::getVariantName()<<std::endl;

        std::vector<std::pair<std::string,std::string> > s;

        s.push_back( std::make_pair("Remarks" , "Kernel iterating over all the contacts, doing the prox " ));
        s.push_back( std::make_pair("VariantTagShort","CIT"));
        s.push_back( std::make_pair("VariantTagLong","ContactIteration"));

        s.push_back( std::make_pair("VariantID",std::to_string(GPUVariantType::VariantId2)));
        s.push_back( std::make_pair("VariantName", GPUVariantType::getVariantName() ));
        s.push_back( std::make_pair("VariantDescription", GPUVariantType::getVariantDescriptionShort()));
        s.push_back( std::make_pair("VariantSettings" , GPUVariantType::getVariantDescriptionLong()));
        s.push_back( std::make_pair("Precision", typeid(PREC).name()));

        return s;
    }
std::vector<std::string>
    getDataColumHeader(){

        std::vector<std::string> s;
        s.push_back("nBodies");

        return s;
    }




};

#include "CudaFramework/Kernels/JORProxVel/ContactIterationKernel/ContactIteration.icc"


/**
* @brief GPUTestVariant which computes a contact frame for all contacts.
*/
template<typename PREC, unsigned int TVariantId>
class ContIterGPUVariant {
public:

    typedef Eigen::Matrix<PREC,Eigen::Dynamic, Eigen::Dynamic,Eigen::ColMajor>  MatrixType;
    typedef Eigen::Matrix<unsigned int,Eigen::Dynamic, Eigen::Dynamic,Eigen::ColMajor>  MatrixUIntType;
    typedef std::vector<PREC> VectorType;


    unsigned int redLength=1;

    double m_nops;
         VariantLaunchSettings variantSettings;
             const static unsigned int VariantId2=TVariantId;



   void setSettings() {

        switch(TVariantId) {
        case 1:
            variantSettings.numberOfBlocks=128;
            variantSettings.numberOfThreads=32;
            break;
        case 2:
            variantSettings.numberOfBlocks=128;
            variantSettings.numberOfThreads=64;
             break;
        case 3:
            variantSettings.numberOfBlocks=128;
            variantSettings.numberOfThreads=128;
             break;
        case 4:
            variantSettings.numberOfBlocks=128;
            variantSettings.numberOfThreads=256;
             break;
          case 5:
            variantSettings.numberOfBlocks=256;
            variantSettings.numberOfThreads=256;
             break;
          case 6:
            variantSettings.numberOfBlocks=256;
            variantSettings.numberOfThreads=512;
             break;
        case 7:
            variantSettings.numberOfBlocks=512;
            variantSettings.numberOfThreads=512;
             break;
        default:
            variantSettings.numberOfBlocks=30;
            variantSettings.numberOfThreads=256;
             break;
        }
    }

    static std::string getVariantName() {

          return "Contact Iteration Kernel";
    }

    static std::string getVariantDescriptionShort() {
          return "Does the Prox and calculates the delta u's";
    }

    static std::string getVariantDescriptionLong() {
        switch(TVariantId) {
        case 1:
            return "(Block Dim: 32, Grid Dim: 128)";
            break;
        case 2:
            return "(Block Dim: 64, Grid Dim: 128)";
            break;
        case 3:
            return "(Block Dim: 128, Grid Dim: 128)";
            break;
        case 4:
            return "(Block Dim: 256, Grid Dim: 128)";
            break;
        case 5:
            return "(Block Dim: 256, Grid Dim: 256)";
            break;
        case 6:
            return "(Block Dim: 256, Grid Dim: 512)";
            break;
        case 7:
            return "(Block Dim: 512, Grid Dim: 512)";
            break;
        default:
            return "(Block Dim: 128, Grid Dim: 128)";
            break;
        }
    }



    void initialize(std::ostream* pLog) {

        m_pLog = pLog;
    }

    void finalize() {

    }

    void initializeTestProblem( unsigned int numberOfContacts,
                                MatrixType & bodyBufferGPU,
                                MatrixType & contBufferGPU,
                                MatrixUIntType & globalBufferGPU,
                                MatrixUIntType & indexSetGPU,
                                MatrixType & outputMatrixGPU) {


        redLength=12*numberOfContacts;

        size_t freeMem;
        size_t totalMem;

        CHECK_CUDA(cudaMemGetInfo (&freeMem, &totalMem));

        int bodyMatrixSize=bodyBufferGPU.cols()*bodyBufferGPU.rows();
        int globalMatrixSize=globalBufferGPU.cols()*globalBufferGPU.rows();
        int outputMatrixSize=outputMatrixGPU.cols()*outputMatrixGPU.rows();

        int nGPUBytes = (outputMatrixSize+bodyMatrixSize+redLength)*(sizeof(PREC))+(sizeof(unsigned int))*(globalMatrixSize);
        //ERRORMSG("HOW MANY BYTES ARE WE GONNA USE ON THE GPU");

        *m_pLog << "Will try to allocate ("<<nGPUBytes<<"/"<<freeMem<<") = " << (double)nGPUBytes/freeMem * 100.0 <<" % of global memory on GPU"<<std::endl;
        if(nGPUBytes > freeMem) {
            *m_pLog <<"To little memory on GPU, exit!..."<<std::endl;
            std::exit(-1);
        }


        CHECK_CUDA(utilCuda::mallocMatrixDevice<true>(m_bodyDev,bodyBufferGPU));
        CHECK_CUDA(utilCuda::mallocMatrixDevice<true>(m_contactDev,contBufferGPU));
        CHECK_CUDA(utilCuda::mallocMatrixDevice<true>(m_indexSetDev,indexSetGPU));

        CHECK_CUDA(cudaMalloc((void**)&m_reductionDev,redLength*sizeof(PREC)));
        CHECK_CUDA(utilCuda::mallocMatrixDevice<true>(m_globalDev,globalBufferGPU));

        CHECK_CUDA(utilCuda::mallocMatrixDevice<true>(m_outDev,outputMatrixGPU));

    }
    void run(unsigned int numberOfContacts,
             MatrixType & bodyBufferGPU,
             MatrixType & contBufferGPU,
             MatrixUIntType & globalBufferGPU,
             VectorType & reductionBufferGPU,

             MatrixUIntType & indexSetGPU,
             MatrixType & outputMatrixGPU) {

        *m_pLog <<"Entered GPU run"<<std::endl;

        float time[1];

        cudaEvent_t start,stop;

        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        CHECK_CUDA(cudaEventRecord(start,0));


        CHECK_CUDA(utilCuda::copyMatrixToDevice(m_indexSetDev,indexSetGPU));
        CHECK_CUDA(utilCuda::copyMatrixToDevice(m_contactDev,contBufferGPU));
        CHECK_CUDA(utilCuda::copyMatrixToDevice(m_bodyDev,bodyBufferGPU));
        CHECK_CUDA(cudaMemcpy(m_reductionDev,&(reductionBufferGPU[0]),redLength*sizeof(PREC),cudaMemcpyHostToDevice));
        CHECK_CUDA(utilCuda::copyMatrixToDevice(m_globalDev,globalBufferGPU));

        CHECK_CUDA(cudaEventRecord(stop,0));
        CHECK_CUDA(cudaEventSynchronize(stop));

        //
        CHECK_CUDA(cudaEventElapsedTime(time,start,stop));

        m_elapsedTimeCopyToGPU=double(time[0]);

        *m_pLog <<"Copy Time To Device is: "<< time[0] <<" ms"<< std::endl;



        /// allocate memory

        /// execute Kernel

        runKernel<TVariantId>(numberOfContacts);



        CHECK_CUDA(cudaEventRecord(start,0));
        CHECK_CUDA(utilCuda::copyMatrixToHost(outputMatrixGPU,m_outDev));
        CHECK_CUDA(cudaEventRecord(stop,0));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(time,start,stop));
        *m_pLog <<"Copy Time To Host is: "<< time[0] <<" ms"<< std::endl;

        m_elapsedTimeCopyFromGPU=double(time[0]);

        m_nops=176;

        /// Delete Events
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));


        *m_pLog << " ---> Iteration Time GPU:"<< m_gpuIterationTime <<std::endl;
        *m_pLog << " ---> Copy time from GPU:"<< m_elapsedTimeCopyFromGPU <<std::endl;
        *m_pLog << " ---> Copy time to GPU:"<<  m_elapsedTimeCopyToGPU <<std::endl;
        *m_pLog << " ---> FLOPS GPU:"<<  double(m_nops*numberOfContacts/m_gpuIterationTime*1000) <<std::endl;
    }

    template<int VariantId> inline void runKernel(unsigned int numberOfContacts) {


     float time[1];


         PREC deltaTime=0.1;

        setSettings();

        *m_pLog<<"number of blocks"<<variantSettings.numberOfBlocks <<std::endl;
        *m_pLog<<"number of threads"<<variantSettings.numberOfThreads <<std::endl;


        cudaEvent_t start,stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        unsigned int totalRedNumber=2*numberOfContacts;
        CHECK_CUDA(cudaEventRecord(start,0));

        ContIter::contIterKernelWrap<true,true>(  m_bodyDev,
                                        m_contactDev,
                                        m_globalDev,
                                        m_reductionDev,
                                        m_indexSetDev,
                                        m_outDev,
                                        numberOfContacts,
                                        variantSettings,
                                        totalRedNumber,
                                        0,
                                        0);
        CHECK_CUDA(cudaEventRecord(stop,0));
        CHECK_CUDA(cudaEventSynchronize(stop));

        //
        CHECK_CUDA(cudaEventElapsedTime(time,start,stop));
        m_gpuIterationTime = double(time[0]);

    }


    void cleanUpTestProblem() {

        CHECK_CUDA(utilCuda::freeMatrixDevice(m_outDev));
        CHECK_CUDA(utilCuda::freeMatrixDevice(m_bodyDev));
        CHECK_CUDA(utilCuda::freeMatrixDevice(m_contactDev));
        CHECK_CUDA(utilCuda::freeMatrixDevice(m_globalDev));
        CHECK_CUDA(cudaFree(m_reductionDev));
        CHECK_CUDA(utilCuda::freeMatrixDevice(m_indexSetDev));
    }

    double m_gpuIterationTime;
    double m_elapsedTimeCopyToGPU;
    double m_elapsedTimeCopyFromGPU;
    int m_nIter;

        utilCuda::CudaMatrix<PREC> m_outDev;

        utilCuda::CudaMatrix<PREC> m_bodyDev;
        utilCuda::CudaMatrix<PREC> m_contactDev;
        utilCuda::CudaMatrix<unsigned int> m_globalDev;
        PREC* m_reductionDev;
        utilCuda::CudaMatrix<unsigned int> m_indexSetDev;

private:


    std::ostream* m_pLog;

};


#endif
