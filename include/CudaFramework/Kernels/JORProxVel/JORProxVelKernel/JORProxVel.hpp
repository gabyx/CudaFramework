#ifndef CudaFramework_Kernels_JORProxVel_JORProxVelKernel_JORProxVel_hpp
#define CudaFramework_Kernels_JORProxVel_JORProxVelKernel_JORProxVel_hpp

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <random>

#include <Eigen/Dense>

#include <tinyformat/TinyFormatInclude.hpp>
#include "CudaFramework/CudaModern/CudaMatrixUtilities.hpp"


#include "CudaFramework/General/AssertionDebug.hpp"
#include "CudaFramework/General/TypeDefs.hpp"

#include INCLUDE_MyMatrixDefs_hpp

#include "CudaFramework/CudaModern/CudaError.hpp"


#include "CudaFramework/General/FloatingPointType.hpp"
#include "CudaFramework/CudaModern/CudaMatrix.hpp"

#include "CudaFramework/Kernels/JORProxVel/GPUBufferOffsets.hpp"
#include "CudaFramework/Kernels/JORProxVel/GeneralStructs.hpp"
#include "CudaFramework/Kernels/JORProxVel/VariantLaunchSettings.hpp"
#include "CudaFramework/Kernels/JORProxVel/JORProxVelKernel/JORProxVelGPU.hpp"

#include "CudaFramework/Kernels/JORProxVel/GenRandomContactGraphClass.hpp"



//Prototyp
template<typename PREC, unsigned int TVariantId> class JORProxVelGPUVariant;

/**
* @brief Test range settings
*/
template<unsigned int _VariantId>
struct JORProxVelGPUVariantSettings {

    template <typename _PREC> // Settings from above
    struct GPUVariant {
        typedef JORProxVelGPUVariant< _PREC,_VariantId  >  GPUVariantType;
    };

};


/**
* @brief Test range settings
*/
template< unsigned int _minNContacts,
          unsigned int _maxNContacts,
          unsigned int _stepNContacts,
          unsigned int _maxNRandomRuns >
struct JORProxVelTestRangeSettings {
    static const int minNContacts = _minNContacts;
    static const int stepNContacts = _stepNContacts;
    static const int maxNContacts = _maxNContacts;
    static const int maxNRandomRuns = _maxNRandomRuns;
};
#define DEFINE_JORProxVelTestRangeSettings( _TSettings_ )  \
   static const int  minNContacts = _TSettings_::minNContacts; \
   static const int  stepNContacts = _TSettings_::stepNContacts; \
   static const int  maxNContacts = _TSettings_::maxNContacts; \
   static const int  maxNRandomRuns = _TSettings_::maxNRandomRuns;


/**
* @brief Settings for JORProxVelTestVariant
*/
template<typename _PREC, typename TTestRangeSettings, typename TGPUVariantSettings >
struct JORProxVelSettings {
    typedef _PREC PREC;
    typedef TTestRangeSettings TestRangeSettingsType;
    typedef TGPUVariantSettings GPUVariantSettingsType;

    typedef typename GPUVariantSettingsType::template GPUVariant<PREC>::GPUVariantType GPUVariantType;  // JORProxVelGPUVariant<double,1>
};

#define DEFINE_JORProxVelSettings( __Settings__ ) \
    typedef typename __Settings__::PREC PREC; \
    typedef typename __Settings__::TestRangeSettingsType TestRangeSettingsType; \
    typedef typename __Settings__::GPUVariantSettingsType GPUVariantSettingsType; \
    typedef typename __Settings__::GPUVariantType GPUVariantType; \
    DEFINE_JORProxVelTestRangeSettings( TestRangeSettingsType )



/**
* @brief TestVariant which computes a contact frame for all contacts.
*/
template<typename TSettings >
class JORProxVelTestVariant {

public:

    typedef TSettings SettingsType;

    DEFINE_JORProxVelSettings( TSettings )

    typename SettingsType::GPUVariantType m_gpuVariant;

    DEFINE_MATRIX_TYPES_OF( PREC )


    /// typename PREC

    /// typename TestRangeSettingsType
    /// static const int  minNContacts
    /// static const int  stepNContacts
    /// static const int  maxNContacts
    /// static const int  maxNRandomRuns

    /// typename GPUVariantSettingsType
    /// typedef JORProxVelGPUVariant< _PREC,_VariantId  >  GPUVariantType


    JORProxVelTestVariant();

private:

    using  RandomGeneratorType = std::mt19937;  // the Mersenne Twister with a popular choice of parameters
    using  DistributionType = std::uniform_real_distribution<PREC>;
    const uint32_t m_seed =  30;




    typedef Eigen::Matrix<PREC,Eigen::Dynamic, Eigen::Dynamic,Eigen::ColMajor> MatrixType;
    typedef Eigen::Matrix<unsigned int,Eigen::Dynamic, Eigen::Dynamic,Eigen::ColMajor>  MatrixUIntType;
    typedef std::vector<PREC> VectorType;
    typedef std::vector<int> VectorIntType;
    typedef GenRndContactGraph<VectorIntType,MatrixType,MatrixUIntType> RndGraph;


    ///================///

    MatrixType m_bodyBufferGPU;
    MatrixType m_contBufferGPU;
    MatrixUIntType m_globalBufferGPU;
    VectorType m_reductionBufferGPU;
    MatrixUIntType m_indexSetGPU;
    VectorIntType m_csrGPU;

    MatrixType m_bodyBufferCPU;
    MatrixType m_contBufferCPU;
    MatrixUIntType m_globalBufferCPU;
    VectorType m_reductionBufferCPU;
    MatrixUIntType m_indexSetCPU;
    VectorIntType m_csrCPU;

    ///================///

    const unsigned int m_implementedLengthInput=62;

    ///================///

    const unsigned int m_bodyBufferLength       = JORProxVelGPU::GPUBufferOffsets::BodyBufferOffsets::length;
    const unsigned int m_contactBufferLength    = JORProxVelGPU::GPUBufferOffsets::ContBufferOffsets::length;
    const unsigned int m_redBufferLength        = JORProxVelGPU::GPUBufferOffsets::ReductionBufferOffsets::length;
    const unsigned int m_globalBufferLength = JORProxVelGPU::GPUBufferOffsets::GlobalBufferOffsets::length;
    const unsigned int m_indexSetLength = JORProxVelGPU::GPUBufferOffsets::IndexBufferOffsets::length;

    ///================///

    const unsigned int m_implementedLengthOutput=12;

    unsigned int m_numberOfContacts;
    unsigned int m_numberOfBodies;
    unsigned int m_rowsInGlobalBuffer=1;
    unsigned int m_numberOfReductions=1;
    unsigned int m_contPerBody=2;

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
    double absTol=10E-7;
    const double Tolerance=0.0000000001;
    typename TypeWithSize<sizeof(PREC)>::UInt m_maxUlp;
    double m_avgUlp;



    static std::string getTestVariantDescription() {
        return "JORProxVel";
    }

public:

    void initialize(std::ostream * pLog, std::ostream * pData);
     /** Check settings at runtime, static settings are already checked at compile time*/
    bool checkSettings(int gpuID){WARNINGMSG(false,"checkSettings not correctly implemented!"); return true;}

    bool generateNextTestProblem();
    bool generateNextRandomRun();
    void writeHeader();
    void writeDataColumHeader();
    void runOnGPU();
    void runOnCPU();
    void checkResults();
    void cleanUpTestProblem();
    void writeData();
    void finalize() {
        m_gpuVariant.finalize();
    }
    bool compareOutput();

    bool checkReductionBuffer(VectorType,VectorType);
    bool isequal(PREC,PREC);
    void reductionCPU(std::vector<ContactData<PREC> >& ,
                      std::vector<BodyData<PREC> >& );

    void loadToRedBufferCPU(std::vector<ContactData<PREC> >&,
                        VectorType & ,
                        MatrixType & ,
                        MatrixUIntType );

    void checkResultsMatrix(MatrixType,
                            MatrixType);

    bool compareCudaMatrices(MatrixType,MatrixType);
    bool compareCudaMatrices(MatrixUIntType,MatrixUIntType);


    std::vector<std::pair<std::string,std::string> >
    getDescriptions() {

        *m_pLog  << "Kernel used: \t\t"<< GPUVariantType::getVariantName()<<std::endl;

        std::vector<std::pair<std::string,std::string> > s;

        s.push_back( std::make_pair("Remarks" , "Vel JOR Prox" ));

        s.push_back( std::make_pair("VariantTagShort","JPV"));
        s.push_back( std::make_pair("VariantTagLong","JORProxVel"));

        ///s.push_back( std::make_pair("ProxIterationType",Utilities::cutStringTillScope(typeid(TProxIterationType).name()) ));
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

#include "CudaFramework/Kernels/JORProxVel/JORProxVelKernel/JORProxVel.icc"

/**
* @brief GPUTestVariant which computes a contact frame for all contacts.
*/


#endif // TestSimple2_hpp

