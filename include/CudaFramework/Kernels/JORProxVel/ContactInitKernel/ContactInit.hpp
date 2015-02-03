#ifndef ContactInit_hpp
#define ContactInit_hpp

#include <cmath>
#include <cstdlib>
#include <chrono>
#include <iostream>

#include <tinyformat/TinyFormatInclude.hpp>
#include "CudaFramework/CudaModern/CudaUtilities.hpp"


#include "CudaFramework/General/AssertionDebug.hpp"
#include "CudaFramework/CudaModern/CudaError.hpp"
#include <Eigen/Dense>
#include "CudaFramework/Kernels/JORProxVel/ContactInitKernel/ContactInitKernelWrap.hpp"
#include "CudaFramework/General/FloatingPointType.hpp"
#include "CudaFramework/CudaModern/CudaMatrix.hpp"
#include "CudaFramework/CudaModern/CudaMatrixUtilities.hpp"
#include "CudaFramework/Kernels/JORProxVel/VariantLaunchSettings.hpp"
#include "CudaFramework/Kernels/JORProxVel/GPUBufferOffsets.hpp"
#include "CudaFramework/Kernels/JORProxVel/GenRandomContactGraphClass.hpp"


#include <cuda_runtime.h>


//Prototyp
template<typename PREC, unsigned int TVariantId> class ContactInitGPUVariant;

/**
* @brief Test range settings
*/
template<unsigned int _VariantId>
struct ContactInitGPUVariantSettings {

    template <typename _PREC> // Settings from above
    struct GPUVariant {
        typedef ContactInitGPUVariant< _PREC,_VariantId  >  GPUVariantType;
    };

};


/**
* @brief Test range settings
*/
template< unsigned int _minNContacts,
          unsigned int _maxNContacts,
          unsigned int _stepNContacts,
          unsigned int _maxNRandomRuns >
struct ContactInitTestRangeSettings {
    static const int minNContacts = _minNContacts;
    static const int stepNContacts = _stepNContacts;
    static const int maxNContacts = _maxNContacts;
    static const int maxNRandomRuns = _maxNRandomRuns;
};
#define DEFINE_ContactInitTestRangeSettings( _TSettings_ )  \
   static const int  minNContacts = _TSettings_::minNContacts; \
   static const int  stepNContacts = _TSettings_::stepNContacts; \
   static const int  maxNContacts = _TSettings_::maxNContacts; \
   static const int  maxNRandomRuns = _TSettings_::maxNRandomRuns;


/**
* @brief Settings for ContactInitTestVariant
*/
template<typename _PREC, typename TTestRangeSettings, typename TGPUVariantSettings >
struct ContactInitSettings {
    typedef _PREC PREC;
    typedef TTestRangeSettings TestRangeSettingsType;
    typedef TGPUVariantSettings GPUVariantSettingsType;

    typedef typename GPUVariantSettingsType::template GPUVariant<PREC>::GPUVariantType GPUVariantType;  // ContactInitGPUVariant<double,1>
};

#define DEFINE_ContactInitSettings( __Settings__ ) \
    typedef typename __Settings__::PREC PREC; \
    typedef typename __Settings__::TestRangeSettingsType TestRangeSettingsType; \
    typedef typename __Settings__::GPUVariantSettingsType GPUVariantSettingsType; \
    typedef typename __Settings__::GPUVariantType GPUVariantType; \
    DEFINE_ContactInitTestRangeSettings( TestRangeSettingsType )



/**
* @brief TestVariant which computes a contact frame for all contacts.
*/
template<typename TSettings >
class ContactInitTestVariant {

public:

    typedef TSettings SettingsType;

    DEFINE_ContactInitSettings( TSettings )

    typename SettingsType::GPUVariantType m_gpuVariant;

    /// typename PREC

    /// typename TestRangeSettingsType
        /// static const int  minNContacts
        /// static const int  stepNContacts
        /// static const int  maxNContacts
        /// static const int  maxNRandomRuns

    /// typename GPUVariantSettingsType
        /// typedef ContactInitGPUVariant< _PREC,_VariantId  >  GPUVariantType


    ContactInitTestVariant();

private:

    using  RandomGeneratorType = std::mt19937;  // the Mersenne Twister with a popular choice of parameters
    using  DistributionType = std::uniform_real_distribution<PREC>;
    using  DistributionTypeuInt = std::uniform_int_distribution<unsigned int>;
    const uint32_t m_seed =  30;

    const double Tolerance=1E-5;

    ///typedef Eigen::Matrix<PREC,Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor> MatrixType;

    typedef Eigen::Matrix<PREC,Eigen::Dynamic, Eigen::Dynamic,Eigen::ColMajor> MatrixType;
    typedef Eigen::Matrix<unsigned int,Eigen::Dynamic, Eigen::Dynamic,Eigen::ColMajor> MatrixUIntType;
    typedef std::vector<PREC> VectorType;
    typedef std::vector<int> VectorIntType;

    typedef GenRndContactGraph<VectorIntType,MatrixType,MatrixUIntType> RndGraph;



    ///================///

    MatrixType m_bodyBufferGPU;
    MatrixType m_contBufferGPU;
    MatrixUIntType m_globalBufferGPU;
    MatrixType m_reductionBufferGPU;
    MatrixUIntType m_indexSetGPU;

    MatrixType m_bodyBufferCPU;
    MatrixType m_contBufferCPU;
    MatrixUIntType m_globalBufferCPU;
    MatrixType m_reductionBufferCPU;
    MatrixUIntType m_indexSetCPU;

    ///================///

    MatrixType m_outputMatrixCPU;
    MatrixType m_outputMatrixGPU;

    VectorIntType m_csrGPUDummy;


    ///================///


    const unsigned int m_implementedLengthInput = JORProxVelGPU::GPUBufferOffsets::ContBufferOffsets::length;
    const unsigned int m_bodyBufferLength       = JORProxVelGPU::GPUBufferOffsets::BodyBufferOffsets::length;
    const unsigned int m_contactBufferLength    = JORProxVelGPU::GPUBufferOffsets::ContBufferOffsets::length;
    const unsigned int m_redBufferLength        = JORProxVelGPU::GPUBufferOffsets::ReductionBufferOffsets::length;
    const unsigned int m_globalBufferLength     = JORProxVelGPU::GPUBufferOffsets::GlobalBufferOffsets::length;

    const unsigned int m_indexSetLength = JORProxVelGPU::GPUBufferOffsets::IndexBufferOffsets::length;

    const unsigned int m_implementedLengthOutput=33;

    unsigned int m_numberOfContacts;

    unsigned int m_numberOfBodies=2;
    unsigned int m_rowsInGlobalBuffer=1;
    unsigned int m_numberOfReductions=1;
    unsigned int m_contPerBody=1;

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
        return "ContactInit";
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


        std::vector<std::pair<std::string,std::string> >
    getDescriptions() {
        *m_pLog  << "Kernel used: \t\t"<< GPUVariantType::getVariantName()<<std::endl;

        std::vector<std::pair<std::string,std::string> > s;

        s.push_back( std::make_pair("Remarks" , "Initialising Kernel iterating over all the contacts" ));
        s.push_back( std::make_pair("VariantTagShort","CIN"));
        s.push_back( std::make_pair("VariantTagLong","ContactInit"));

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

#include "CudaFramework/Kernels/JORProxVel/ContactInitKernel/ContactInit.icc"


/**
* @brief GPUTestVariant which computes a contact frame for all contacts.
*/
template<typename PREC, unsigned int TVariantId>
class ContactInitGPUVariant {
public:
typedef Eigen::Matrix<PREC,Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor>  MatrixType;
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
            variantSettings.numberOfBlocks=90;
            variantSettings.numberOfThreads=64;
             break;
        }
    }

    static std::string getVariantName() {

          return "Contact Initialisation";
    }

    static std::string getVariantDescriptionShort() {
          return "Initialises the contact data, calculates the W's etc.";
    }

    static std::string getVariantDescriptionLong() {
                switch(TVariantId) {
        case 1:
             return "(Block Dim 32) [Number Of Blocks: 128]";
            break;
        case 2:
             return "[Number of Threads: 64][Number Of Blocks: 128]";
             break;
        case 3:
             return "[Number of Threads: 128][Number Of Blocks: 128]";
             break;
        case 4:
             return "[Number of Threads: 256][Number Of Blocks: 128]";
             break;
          case 5:
             return "[Number of Threads: 256][Number Of Blocks: 256]";
             break;
          case 6:
             return "[Number of Threads: 256][Number Of Blocks: 512]";
             break;
        case 7:
             return "[Number of Threads: 512][Number Of Blocks: 512]";
             break;
        default:
             return "[Number of Threads: 128][Number Of Blocks: 128]";
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
                                MatrixType & contBufferGPU,
                                MatrixUIntType & globalBufferGPU,
                                MatrixUIntType & indexSetGPU,

                                MatrixType & outputMatrixGPU) {

        size_t freeMem;
        size_t totalMem;

        CHECK_CUDA(cudaMemGetInfo (&freeMem, &totalMem));

        int bodyMatrixSize=bodyBufferGPU.cols()*bodyBufferGPU.rows();

        int globalMatrixSize=globalBufferGPU.cols()*globalBufferGPU.rows();
        int contactMatrixSize=contBufferGPU.cols()*contBufferGPU.rows();
        int indexMatrixSize=indexSetGPU.cols()*indexSetGPU.rows();
        int outputMatrixSize=outputMatrixGPU.cols()*outputMatrixGPU.rows();

        int nGPUBytes = (contactMatrixSize+outputMatrixSize+bodyMatrixSize)*(sizeof(PREC))+(sizeof(unsigned int))*(globalMatrixSize+indexMatrixSize);

        *m_pLog << "Will try to allocate ("<<nGPUBytes<<"/"<<freeMem<<") = " << (double)nGPUBytes/freeMem * 100.0 <<" % of global memory on GPU"<<std::endl;
        if(nGPUBytes > freeMem) {
            *m_pLog <<"To little memory on GPU, exit!..."<<std::endl;
            std::exit(-1);
        }

         /** allocate memory **/

        CHECK_CUDA(utilCuda::mallocMatrixDevice<true>(m_bodyDev,bodyBufferGPU));
        CHECK_CUDA(utilCuda::mallocMatrixDevice<true>(m_contactDev,contBufferGPU));
        CHECK_CUDA(utilCuda::mallocMatrixDevice<true>(m_indexSetDev,indexSetGPU));
        CHECK_CUDA(utilCuda::mallocMatrixDevice<true>(m_globalDev,globalBufferGPU));
        CHECK_CUDA(utilCuda::mallocMatrixDevice<true>(m_outDev,outputMatrixGPU));

    }
    template<typename MatrixType,typename MatrixUIntType>
    void run(unsigned int numberOfContacts,
             MatrixType & bodyBufferGPU,
             MatrixType & contBufferGPU,
             MatrixUIntType & globalBufferGPU,
             MatrixUIntType & indexSetGPU,

             MatrixType & outputMatrixGPU){


        cudaEvent_t start,stop;
               CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        *m_pLog<<"Entered GPU run"<<std::endl;

        float time[1];
        CHECK_CUDA(cudaEventRecord(start,0));

        /** copy memory **/

        CHECK_CUDA(utilCuda::copyMatrixToDevice(m_bodyDev,bodyBufferGPU));
        CHECK_CUDA(utilCuda::copyMatrixToDevice(m_contactDev,contBufferGPU));
        CHECK_CUDA(utilCuda::copyMatrixToDevice(m_indexSetDev,indexSetGPU));
        CHECK_CUDA(utilCuda::copyMatrixToDevice(m_globalDev,globalBufferGPU));

        CHECK_CUDA(cudaEventRecord(stop,0));
        CHECK_CUDA(cudaEventSynchronize(stop));
        //
        CHECK_CUDA(cudaEventElapsedTime(time,start,stop));
        m_elapsedTimeCopyToGPU=double(time[0]);


        *m_pLog<<"Copy Time To Device is: "<< m_elapsedTimeCopyToGPU <<" ms"<< std::endl;


        /// execute Kernel


        runKernel<TVariantId>(numberOfContacts);

        /// Delete Events


        CHECK_CUDA(cudaEventRecord(start,0));
        CHECK_CUDA(utilCuda::copyMatrixToHost(outputMatrixGPU,m_outDev));

        CHECK_CUDA(cudaEventRecord(stop,0));
        CHECK_CUDA(cudaEventSynchronize(stop));

        CHECK_CUDA(cudaEventElapsedTime(time,start,stop));

        *m_pLog <<"Copy Time To Host is: "<< time[0] <<" ms"<< std::endl;

        m_elapsedTimeCopyFromGPU=double(time[0]);

        m_nops=377;   /// probably wrong, difficult to count

        *m_pLog << " ---> Iteration Time GPU:"<< m_gpuIterationTime <<std::endl;
        *m_pLog << " ---> Copy time from GPU:"<< m_elapsedTimeCopyFromGPU <<std::endl;
        *m_pLog << " ---> Copy time to GPU:"<<  m_elapsedTimeCopyToGPU <<std::endl;
        *m_pLog << " ---> FLOPS GPU:"<<  double(m_nops*numberOfContacts/m_gpuIterationTime*1000) <<std::endl;

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));



    }

    template<int VariantId> inline void runKernel(unsigned int numberOfContacts) {


        float time[1];
        cudaEvent_t start,stop;

        setSettings();
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start,0));


        ContactInit::contactInitKernelWrap<true>(m_bodyDev,
        m_contactDev,
        m_indexSetDev,
        m_globalDev,
        m_outDev,
        numberOfContacts,
        variantSettings);

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
    utilCuda::CudaMatrix<unsigned int> m_indexSetDev;

private:

    std::ostream* m_pLog;

};


#endif
