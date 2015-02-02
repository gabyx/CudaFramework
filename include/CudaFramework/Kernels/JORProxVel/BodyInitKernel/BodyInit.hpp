#ifndef BodyInit_hpp
#define BodyInit_hpp

#include <cmath>
#include <cstdlib>
#include <chrono>
#include <iostream>

//#include <boost/format.hpp>

#include "CudaFramework/CudaModern/CudaUtilities.hpp"


#include "CudaFramework/General/AssertionDebug.hpp"
#include "CudaFramework/CudaModern/CudaError.hpp"
#include <Eigen/Dense>
#include "CudaFramework/Kernels/JORProxVel/BodyInitKernel/BodyInitKernelWrap.hpp"
#include "CudaFramework/General/FloatingPointType.hpp"
#include "CudaFramework/CudaModern/CudaMatrix.hpp"
#include "CudaFramework/CudaModern/CudaMatrixUtilities.hpp"
#include "CudaFramework/Kernels/JORProxVel/GPUBufferOffsets.hpp"
#include "CudaFramework/Kernels/JORProxVel/VariantLaunchSettings.hpp"

#include <cuda_runtime.h>





//Prototyp
template<typename PREC, unsigned int TVariantId> class BodyInitGPUVariant;

/**
* @brief Test range settings
*/
template<unsigned int _VariantId>
struct BodyInitGPUVariantSettings {

    template <typename _PREC> // Settings from above
    struct GPUVariant {
        typedef BodyInitGPUVariant< _PREC,_VariantId  >  GPUVariantType;
    };

};


/**
* @brief Test range settings
*/
template< unsigned int _minNContacts,
          unsigned int _maxNContacts,
          unsigned int _stepNContacts,
          unsigned int _maxNRandomRuns >
struct BodyInitTestRangeSettings {
    static const int minNContacts = _minNContacts;
    static const int stepNContacts = _stepNContacts;
    static const int maxNContacts = _maxNContacts;
    static const int maxNRandomRuns = _maxNRandomRuns;
};
#define DEFINE_BodyInitTestRangeSettings( _TSettings_ )  \
   static const int  minNContacts = _TSettings_::minNContacts; \
   static const int  stepNContacts = _TSettings_::stepNContacts; \
   static const int  maxNContacts = _TSettings_::maxNContacts; \
   static const int  maxNRandomRuns = _TSettings_::maxNRandomRuns;


/**
* @brief Settings for ContactInitTestVariant
*/
template<typename _PREC, typename TTestRangeSettings, typename TGPUVariantSettings >
struct BodyInitSettings {
    typedef _PREC PREC;
    typedef TTestRangeSettings TestRangeSettingsType;
    typedef TGPUVariantSettings GPUVariantSettingsType;

    typedef typename GPUVariantSettingsType::template GPUVariant<PREC>::GPUVariantType GPUVariantType;  // ContactInitGPUVariant<double,1>
};

#define DEFINE_BodyInitSettings( __Settings__ ) \
    typedef typename __Settings__::PREC PREC; \
    typedef typename __Settings__::TestRangeSettingsType TestRangeSettingsType; \
    typedef typename __Settings__::GPUVariantSettingsType GPUVariantSettingsType; \
    typedef typename __Settings__::GPUVariantType GPUVariantType; \
    DEFINE_BodyInitTestRangeSettings( TestRangeSettingsType )



/**
* @brief TestVariant which computes a contact frame for all contacts.
*/
template<typename TSettings >
class BodyInitTestVariant {

public:

    typedef TSettings SettingsType;

    DEFINE_BodyInitSettings( TSettings )

    typename SettingsType::GPUVariantType m_gpuVariant;

    /// typename PREC

    /// typename TestRangeSettingsType
        /// static const int  minNContacts
        /// static const int  stepNContacts
        /// static const int  maxNContacts
        /// static const int  maxNRandomRuns

    /// typename GPUVariantSettingsType
        /// typedef ContactInitGPUVariant< _PREC,_VariantId  >  GPUVariantType


    BodyInitTestVariant();

//    ~GetBaseClass();
private:
    using  RandomGeneratorType = std::mt19937;  // the Mersenne Twister with a popular choice of parameters
    using  DistributionType = std::uniform_real_distribution<PREC>;
    using  DistributionTypeuInt = std::uniform_int_distribution<unsigned int>;


    const double Tolerance=0.0000001;

    typedef Eigen::Matrix<PREC,Eigen::Dynamic, Eigen::Dynamic,Eigen::ColMajor> MatrixType;
    typedef Eigen::Matrix<unsigned int,Eigen::Dynamic, Eigen::Dynamic,Eigen::ColMajor> MatrixUIntType;


    MatrixType m_bodyBufferGPU;
    MatrixType m_contBufferGPU;
    MatrixUIntType m_globalBufferGPU;
    MatrixType m_reductionBufferGPU;

    MatrixType m_bodyBufferCPU;
    MatrixType m_contBufferCPU;
    MatrixType m_reductionBufferCPU;
    MatrixUIntType m_globalBufferCPU;

    MatrixType m_outputMatrixCPU;
    MatrixType m_outputMatrixGPU;

    const unsigned int m_implementedLengthInput=JORProxVelGPU::GPUBufferOffsets::BodyBufferOffsets::length;   ///

    const unsigned int bodyBufferLength       = JORProxVelGPU::GPUBufferOffsets::BodyBufferOffsets::length;
    const unsigned int contactBufferLength    = JORProxVelGPU::GPUBufferOffsets::ContBufferOffsets::length;
    const unsigned int redBufferLength        = JORProxVelGPU::GPUBufferOffsets::ReductionBufferOffsets::length;
    const unsigned int GlobalBufferOffsetsLength = JORProxVelGPU::GPUBufferOffsets::GlobalBufferOffsets::length;

    const unsigned int m_implementedLengthOutput=6;

    unsigned int m_numberOfBodies;
    unsigned int m_numberOfContacts;
    unsigned int m_rowsInGlobalBuffer=1;
    unsigned int m_numberOfReductions;


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


        static std::string getTestVariantDescription()
    {
        return "BodyInit";
    }

        std::vector<std::pair<std::string,std::string> >
    getDescriptions() {
        *m_pLog  << "Kernel used: \t\t"<< GPUVariantType::getVariantName()<<std::endl;

        std::vector<std::pair<std::string,std::string> > s;

        s.push_back( std::make_pair("Remarks" , "Initialising Kernel iterating over all the bodies" ));
        s.push_back( std::make_pair("VariantTagShort","BIS"));
        s.push_back( std::make_pair("VariantTagLong","BodyInit"));

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

#include "CudaFramework/Kernels/JORProxVel/BodyInitKernel/BodyInit.icc"


/**
* @brief GPUTestVariant which computes a contact frame for all contacts.
*/
template<typename PREC, unsigned int TVariantId>
class BodyInitGPUVariant {
public:

    const unsigned int VariantId=TVariantId;

    VariantLaunchSettings variantSettings;


    void setSettings() {

        switch(TVariantId/10){
         case 1:

            variantSettings.var=1;
            break;

        case 2:
            variantSettings.var=2;
             break;
        case 3:
            variantSettings.var=3;
        default:

            variantSettings.var=1;

             break;
        }

        switch(TVariantId%10) {
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
            variantSettings.numberOfBlocks=256;
            variantSettings.numberOfThreads=256;

             break;
        }
    }

    const static unsigned int VariantId2=TVariantId;
    double m_nops;
    static std::string getVariantName() {

          return "Body Initialization Kernel";
    }

    static std::string getVariantDescriptionShort() {
          return "Calculates the uZero from uStart";
    }

    static std::string getVariantDescriptionLong() {
        switch(TVariantId%10) {
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
            return "(Block Dim: 555, Grid Dim: 128)";
            break;
        }
    }

    void initialize(std::ostream* pLog) {

        m_pLog = pLog;
    }

    void finalize() {

    }

    template<typename MatrixType,typename MatrixUIntType>
    void initializeTestProblem( MatrixType & bodyBufferGPU,
                                MatrixUIntType & globalBufferGPU,
                                MatrixType & outputMatrixGPU) {

        size_t freeMem;
        size_t totalMem;

        CHECK_CUDA(cudaMemGetInfo (&freeMem, &totalMem));

        int nGPUBytes = ((bodyBufferGPU.rows()*bodyBufferGPU.cols())+(outputMatrixGPU.cols()*outputMatrixGPU.rows())+globalBufferGPU.cols()*globalBufferGPU.rows())*(sizeof(PREC));

        *m_pLog << "Will try to allocate ("<<nGPUBytes<<"/"<<freeMem<<") = " << (double)nGPUBytes/freeMem * 100.0 <<" % of global memory on GPU"<<std::endl;
        if(nGPUBytes > freeMem) {
            *m_pLog <<"To little memory on GPU, exit!..."<<std::endl;
            std::exit(-1);
        }

        CHECK_CUDA(utilCuda::mallocMatrixDevice<true>(m_outDev,outputMatrixGPU));
        CHECK_CUDA(utilCuda::mallocMatrixDevice<true>(m_bodyDev,bodyBufferGPU));
        CHECK_CUDA(utilCuda::mallocMatrixDevice<true>(m_globalDev,globalBufferGPU));


    }
    template<typename MatrixType,typename MatrixUIntType>
    void run(unsigned int numberOfBodies,
             MatrixType & bodyBufferGPU,
             MatrixUIntType & globalBufferGPU,
             MatrixType & outputMatrixGPU){

         *m_pLog << "Entered GPU run"<<std::endl;

        float time[1];

        cudaEvent_t start,stop;

        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start,0));

        CHECK_CUDA(utilCuda::copyMatrixToDevice(m_bodyDev,bodyBufferGPU));
        CHECK_CUDA(utilCuda::copyMatrixToDevice(m_globalDev,globalBufferGPU));

        CHECK_CUDA(cudaEventRecord(stop,0));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(time,start,stop));

        *m_pLog << "Copy Time To Device is: "<< time[0] <<" ms"<< std::endl;
        m_elapsedTimeCopyToGPU=double(time[0]);


         *m_pLog << "KernelStart"<<std::endl;

        runKernel<TVariantId>(numberOfBodies);

        CHECK_CUDA(cudaEventRecord(start,0));
        CHECK_CUDA(utilCuda::copyMatrixToHost(outputMatrixGPU,m_outDev));
        CHECK_CUDA(cudaEventRecord(stop,0));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(time,start,stop));

        *m_pLog << "Copy Time To Host is: "<< time[0] <<" ms"<< std::endl;

        m_elapsedTimeCopyFromGPU=double(time[0]);

        /// Delete Events
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));

        m_nops=18;

        *m_pLog << " ---> Iteration Time GPU: "<< m_gpuIterationTime <<std::endl;
        *m_pLog << " ---> Copy time from GPU: "<< m_elapsedTimeCopyFromGPU <<std::endl;
        *m_pLog << " ---> Copy time to GPU: "<<  m_elapsedTimeCopyToGPU <<std::endl;
        *m_pLog << " ---> FLOPS GPU: "<<  double(m_nops*numberOfBodies/m_gpuIterationTime*1000) <<std::endl;
    }


    template<int VariantId> inline void runKernel(unsigned int numberOfBodies) {


        float time[1];

        setSettings();

        *m_pLog<<"number of blocks"<<variantSettings.numberOfBlocks <<std::endl;
        *m_pLog<<"number of threads"<<variantSettings.numberOfThreads <<std::endl;

        PREC deltaTime=0.1; ///< only to test the functionality


        cudaEvent_t start,stop;

        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start,0));
        BodyInit::bodyInitKernelWrap<true,PREC>(  m_bodyDev,
                                        m_globalDev,
                                        m_outDev,
                                        numberOfBodies,
                                        variantSettings,
                                        deltaTime);

        CHECK_CUDA(cudaEventRecord(stop,0));
        CHECK_CUDA(cudaEventSynchronize(stop));

        CHECK_CUDA(cudaEventElapsedTime(time,start,stop));
        m_gpuIterationTime = double(time[0]);

    }

    void cleanUpTestProblem() {

        CHECK_CUDA(utilCuda::freeMatrixDevice(m_outDev));
        CHECK_CUDA(utilCuda::freeMatrixDevice(m_bodyDev));
        CHECK_CUDA(utilCuda::freeMatrixDevice(m_globalDev));

    }

    double m_gpuIterationTime;
    double m_elapsedTimeCopyToGPU;
    double m_elapsedTimeCopyFromGPU;
    int m_nIter;

    utilCuda::CudaMatrix<PREC> m_outDev;
    utilCuda::CudaMatrix<PREC> m_bodyDev;
    utilCuda::CudaMatrix<unsigned int> m_globalDev;


private:

    std::ostream* m_pLog;

};


#endif


