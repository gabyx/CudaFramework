#ifndef ConvergenceCheck_hpp
#define ConvergenceCheck_hpp

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <random>

#include <boost/format.hpp>
#include "CudaMatrixUtilities.hpp"


#include "AssertionDebug.hpp"
#include "CudaError.hpp"
#include <Eigen/Dense>
#include "ConvergenceCheckKernelWrap.hpp"
#include "FloatingPointType.hpp"
#include "CudaMatrix.hpp"
#include "GPUBufferOffsets.hpp"
#include "VariantLaunchSettings.hpp"

//Prototyp
template<typename PREC, unsigned int TVariantId> class ConvCheckGPUVariant;

/**
* @brief Test range settings
*/
template<unsigned int _VariantId>
struct ConvCheckGPUVariantSettings {

    template <typename _PREC> // Settings from above
    struct GPUVariant {
        typedef ConvCheckGPUVariant< _PREC,_VariantId  >  GPUVariantType;
    };

};


/**
* @brief Test range settings
*/
template< unsigned int _minNContacts,
          unsigned int _maxNContacts,
          unsigned int _stepNContacts,
          unsigned int _maxNRandomRuns >
struct ConvCheckTestRangeSettings {
    static const int minNContacts = _minNContacts;
    static const int stepNContacts = _stepNContacts;
    static const int maxNContacts = _maxNContacts;
    static const int maxNRandomRuns = _maxNRandomRuns;
};
#define DEFINE_ConvCheckTestRangeSettings( _TSettings_ )  \
   static const int  minNContacts = _TSettings_::minNContacts; \
   static const int  stepNContacts = _TSettings_::stepNContacts; \
   static const int  maxNContacts = _TSettings_::maxNContacts; \
   static const int  maxNRandomRuns = _TSettings_::maxNRandomRuns;


/**
* @brief Settings for ConvCheckTestVariant
*/
template<typename _PREC, typename TTestRangeSettings, typename TGPUVariantSettings >
struct ConvCheckSettings {
    typedef _PREC PREC;
    typedef TTestRangeSettings TestRangeSettingsType;
    typedef TGPUVariantSettings GPUVariantSettingsType;

    typedef typename GPUVariantSettingsType::template GPUVariant<PREC>::GPUVariantType GPUVariantType;
};

#define DEFINE_ConvCheckSettings( __Settings__ ) \
    typedef typename __Settings__::PREC PREC; \
    typedef typename __Settings__::TestRangeSettingsType TestRangeSettingsType; \
    typedef typename __Settings__::GPUVariantSettingsType GPUVariantSettingsType; \
    typedef typename __Settings__::GPUVariantType GPUVariantType; \
    DEFINE_ConvCheckTestRangeSettings( TestRangeSettingsType )



/**
* @brief TestVariant which computes a contact frame for all contacts.
*/
template<typename TSettings >
class ConvCheckTestVariant {

public:

    typedef TSettings SettingsType;

    DEFINE_ConvCheckSettings( TSettings )

    typename SettingsType::GPUVariantType m_gpuVariant;

    /// typename PREC

    /// typename TestRangeSettingsType
        /// static const int  minNContacts
        /// static const int  stepNContacts
        /// static const int  maxNContacts
        /// static const int  maxNRandomRuns

    /// typename GPUVariantSettingsType
        /// typedef ConvCheckGPUVariant< _PREC,_VariantId  >  GPUVariantType


    ConvCheckTestVariant();

private:

    using  RandomGeneratorType = std::mt19937;  // the Mersenne Twister with a popular choice of parameters
    using  DistributionType = std::uniform_real_distribution<PREC>;
    const uint32_t m_seed =  30;

    const double Tolerance=0.00000001;

    typedef Eigen::Matrix<PREC,Eigen::Dynamic, Eigen::Dynamic,Eigen::ColMajor> MatrixType;
    typedef Eigen::Matrix<unsigned int,Eigen::Dynamic, Eigen::Dynamic,Eigen::ColMajor>  MatrixUIntType;




    MatrixType m_bodyBufferGPU;
    MatrixUIntType m_globalBufferGPU;

    MatrixType m_bodyBufferCPU;
    MatrixType m_contBufferCPU;
    MatrixUIntType m_globalBufferCPU;
    MatrixUIntType m_indexSetCPU;

    std::vector<PREC> m_redValBufferGPU;   /// buffer holding the reduced values, length = 6 * number of bodies
    std::vector<PREC> m_redValBufferCPU;

    ///================///

    MatrixType m_outputMatrixCPU;
    MatrixType m_outputMatrixGPU;

    const unsigned int m_implementedLengthInput=16;

    ///================///

    const unsigned int m_bodyBufferLength       = JORProxVelGPU::GPUBufferOffsets::BodyBufferOffsets::length;
    const unsigned int m_globalBufferLength     = JORProxVelGPU::GPUBufferOffsets::GlobalBufferOffsets::length;

    ///================///

    const unsigned int m_implementedLengthOutput=1;

    ///
    unsigned int m_numberOfBodies=1;
    unsigned int m_rowsInGlobalBuffer=1;
    ///

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
        return "ConvergenceCheck";
    }

public:

    void initialize(std::ostream * pLog, std::ostream * pData);
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

        s.push_back( std::make_pair("Remarks" , "Kernel checking the Convergence" ));
        s.push_back( std::make_pair("VariantTagShort","CCK"));
        s.push_back( std::make_pair("VariantTagLong","ConvergenceCheck"));

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

// TODO -hpp -> .icc
#include "ConvergenceCheck.icc"


/**
* @brief GPUTestVariant which computes a contact frame for all contacts.
*/
template<typename PREC, unsigned int TVariantId>
class ConvCheckGPUVariant {
public:
typedef Eigen::Matrix<PREC,Eigen::Dynamic, Eigen::Dynamic,Eigen::ColMajor>  MatrixType;
typedef Eigen::Matrix<unsigned int,Eigen::Dynamic, Eigen::Dynamic,Eigen::ColMajor>  MatrixUIntType;
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
            variantSettings.numberOfBlocks=45;
            variantSettings.numberOfThreads=256;
             break;
        }
    }

    static std::string getVariantName() {

          return "Convergence Check Kernel";
    }

    static std::string getVariantDescriptionShort() {
          return "Calculates if the energy has converged";
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

    void initializeTestProblem( MatrixType & bodyBufferGPU,
                                MatrixUIntType & globalBufferGPU,
                                MatrixType & outputMatrixGPU) {

        size_t freeMem;
        size_t totalMem;

        CHECK_CUDA(cudaMemGetInfo (&freeMem, &totalMem));
        unsigned int numberOfBodies=bodyBufferGPU.rows();

        int bodyMatrixSize=bodyBufferGPU.cols()*bodyBufferGPU.rows();

        int globalMatrixSize=globalBufferGPU.cols()*globalBufferGPU.rows();
        int outputMatrixSize=outputMatrixGPU.cols()*outputMatrixGPU.rows();

        int nGPUBytes = (outputMatrixSize+bodyMatrixSize)*(sizeof(PREC))+(sizeof(unsigned int))*(globalMatrixSize);


        *m_pLog << "Will try to allocate ("<<nGPUBytes<<"/"<<freeMem<<") = " << (double)nGPUBytes/freeMem * 100.0 <<" % of global memory on GPU"<<std::endl;
        if(nGPUBytes > freeMem) {
            *m_pLog <<"To little memory on GPU, exit!..."<<std::endl;
            std::exit(-1);
        }

        CHECK_CUDA(utilCuda::mallocMatrixDevice<true>(m_bodyDev,bodyBufferGPU));
        CHECK_CUDA(utilCuda::mallocMatrixDevice<true>(m_globalDev,globalBufferGPU));
        CHECK_CUDA(utilCuda::mallocMatrixDevice<true>(m_outDev,outputMatrixGPU));
        CHECK_CUDA(cudaMalloc((void**)& m_redBufferInDev,6*numberOfBodies*sizeof(PREC) ))   /// 6 = B::u1_l
    }

    void run(unsigned int numberOfBodies,
             MatrixType & bodyBufferGPU,
             MatrixUIntType & globalBufferGPU,
             MatrixType & outputMatrixGPU,
             PREC* redBuffer) {

        *m_pLog<<"Entered GPU run"<<std::endl;

        float time[3];

        cudaEvent_t start,stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        CHECK_CUDA(cudaEventRecord(start,0));

        CHECK_CUDA(utilCuda::copyMatrixToDevice(m_bodyDev,bodyBufferGPU));
        CHECK_CUDA(utilCuda::copyMatrixToDevice(m_globalDev,globalBufferGPU));
        cudaMemcpy(m_redBufferInDev,redBuffer,6*numberOfBodies*sizeof(PREC),cudaMemcpyHostToDevice );

        CHECK_CUDA(cudaEventRecord(stop,0));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(time,start,stop));

        *m_pLog<<"Copy Time To Device is: "<< time[0] <<" ms"<< std::endl;

        m_elapsedTimeCopyToGPU=double(time[0]);

        runKernel<TVariantId>(numberOfBodies);

        CHECK_CUDA(cudaEventRecord(start,0));

        CHECK_CUDA(utilCuda::copyMatrixToHost(outputMatrixGPU,m_outDev,time));

        CHECK_CUDA(cudaEventRecord(stop,0));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(time,start,stop));

        *m_pLog<<"Copy Time To Host is: "<< time[0] <<" ms"<< std::endl;

        m_elapsedTimeCopyFromGPU=double(time[0]);

        m_nops=36;

        *m_pLog << " ---> Iteration Time GPU:"<< m_gpuIterationTime <<std::endl;
        *m_pLog << " ---> Copy time from GPU:"<< m_elapsedTimeCopyFromGPU <<std::endl;
        *m_pLog << " ---> Copy time to GPU:"<<  m_elapsedTimeCopyToGPU <<std::endl;
        *m_pLog << " ---> FLOPS GPU:"<<  double(m_nops*numberOfBodies/m_gpuIterationTime*1000) <<std::endl;
    }

       template<int VariantId> inline void runKernel(unsigned int numberOfBodies) {


    PREC relTol=0.1;
    PREC absTol=1;



        float time[1];

        setSettings() ;

        *m_pLog<<"number of blocks"<<variantSettings.numberOfBlocks <<std::endl;
        *m_pLog<<"number of threads"<<variantSettings.numberOfThreads <<std::endl;


        PREC deltaTime=0.1;


        cudaEvent_t start,stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start,0));

        ConvCheck::convCheckKernelWrap<true,true,PREC>(  m_bodyDev,
                                                    m_globalDev,
                                                    m_outDev,
                                                    m_redBufferInDev,
                                                    numberOfBodies,
                                                    variantSettings,
                                                    relTol,
                                                    absTol);

        CHECK_CUDA(cudaEventRecord(stop,0));
        CHECK_CUDA(cudaEventSynchronize(stop));
        //
        CHECK_CUDA(cudaEventElapsedTime(time,start,stop));
        m_gpuIterationTime = double(time[0]);
       }


    void cleanUpTestProblem() {

        CHECK_CUDA(utilCuda::freeMatrixDevice(m_outDev));
        CHECK_CUDA(utilCuda::freeMatrixDevice(m_bodyDev));
        CHECK_CUDA(utilCuda::freeMatrixDevice(m_globalDev));
        CHECK_CUDA(cudaFree(m_redBufferInDev));

    }

    double m_gpuIterationTime;
    double m_elapsedTimeCopyToGPU;
    double m_elapsedTimeCopyFromGPU;
    int m_nIter;

        PREC* m_redBufferInDev;
        utilCuda::CudaMatrix<PREC> m_outDev;
        utilCuda::CudaMatrix<PREC> m_bodyDev;
        utilCuda::CudaMatrix<unsigned int> m_globalDev;

private:

    std::ostream* m_pLog;

};


#endif


