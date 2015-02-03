#ifndef ReductionTestVariant_hpp
#define ReductionTestVariant_hpp

#include <cmath>
#include <cstdlib>
#include <chrono>
#include <iostream>
#include <tinyformat/TinyFormatInclude.hpp>

#include "CudaFramework/CudaModern/CudaUtilities.hpp"
#include "CudaFramework/General/AssertionDebug.hpp"
#include "CudaFramework/General/FloatingPointType.hpp"
#include "CudaFramework/CudaModern/CudaError.hpp"
#include "CudaFramework/General/Utilities.hpp"
#include "CudaFramework/CudaModern/CudaContext.hpp"
#include "CudaFramework/CudaModern/CudaPrint.hpp"
#include "CudaFramework/Kernels/JORProxVel/ReductionKernel/Reduction.hpp"
#include "CudaFramework/Kernels/JORProxVel/ReductionKernel/Enum.h"


//Prototyp
template<typename PREC, unsigned int TVariantId> class ReductionGPUVariant;

/**
* @brief Test range settings
*/
template<unsigned int _VariantId>
struct ReductionGPUVariantSettings {

    template <typename _PREC> // Settings from above
    struct GPUVariant {
        typedef ReductionGPUVariant< _PREC,_VariantId  >  GPUVariantType;
    };

};


/**
* @brief Test range settings
*/
template< unsigned int _minNContacts,
          unsigned int _maxNContacts,
          unsigned int _stepNContacts,
          unsigned int _maxNRandomRuns >
struct ReductionTestRangeSettings {
    static const int minNContacts = _minNContacts;
    static const int stepNContacts = _stepNContacts;
    static const int maxNContacts = _maxNContacts;
    static const int maxNRandomRuns = _maxNRandomRuns;
};
#define DEFINE_ReductionTestRangeSettings( _TSettings_ )  \
   static const int  minNContacts = _TSettings_::minNContacts; \
   static const int  stepNContacts = _TSettings_::stepNContacts; \
   static const int  maxNContacts = _TSettings_::maxNContacts; \
   static const int  maxNRandomRuns = _TSettings_::maxNRandomRuns;


/**
* @brief Settings for ReductionTestVariant
*/
template<typename _PREC, typename TTestRangeSettings, typename TGPUVariantSettings >
struct ReductionSettings {
    typedef _PREC PREC;
    typedef TTestRangeSettings TestRangeSettingsType;
    typedef TGPUVariantSettings GPUVariantSettingsType;

    typedef typename GPUVariantSettingsType::template GPUVariant<PREC>::GPUVariantType GPUVariantType;  // ReductionGPUVariant<double,1>
};

#define DEFINE_ReductionSettings( __Settings__ ) \
    typedef typename __Settings__::PREC PREC; \
    typedef typename __Settings__::TestRangeSettingsType TestRangeSettingsType; \
    typedef typename __Settings__::GPUVariantSettingsType GPUVariantSettingsType; \
    typedef typename __Settings__::GPUVariantType GPUVariantType; \
    DEFINE_ReductionTestRangeSettings( TestRangeSettingsType )



/**
* @brief TestVariant which computes a contact frame for all contacts.
*/
template<typename TSettings >
class ReductionTestVariant {

public:

    typedef TSettings SettingsType;

    DEFINE_ReductionSettings( TSettings )

    typename SettingsType::GPUVariantType m_gpuVariant;

    /// typename PREC

    /// typename TestRangeSettingsType
        /// static const int  minNContacts
        /// static const int  stepNContacts
        /// static const int  maxNContacts
        /// static const int  maxNRandomRuns

    /// typename GPUVariantSettingsType
        /// typedef ReductionGPUVariant< _PREC,_VariantId  >  GPUVariantType


    ReductionTestVariant();

//    ~GetBaseClass();
private:

    using  RandomGeneratorType = std::mt19937;  // the Mersenne Twister with a popular choice of parameters
    using  DistributionType = std::uniform_real_distribution<PREC>;
    const uint32_t m_seed =  30;

    const double Tolerance=0.00000001;

    typedef std::vector<PREC> VectorType;
    typedef std::vector<int> VectorIntType;

    VectorType inputVectorGPU;
    VectorIntType inputVectorGPU2;
    VectorType inputVectorCPU;
    VectorIntType inputVectorCPU2;
    VectorType outputVectorCPU;
    VectorType outputVectorGPU;

    const unsigned int m_implementedLengthInput=1;
    const unsigned int m_implementedLengthOutput=1;

    unsigned int numberOfContacts;

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
        return " Reduction ";
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
    bool compareoutput();
    bool isequal(PREC,PREC);

std::vector<std::pair<std::string,std::string> > getDescriptions() {

        *m_pLog  << "Kernel used: \t\t"<< GPUVariantType::getVariantName()<<std::endl;

        std::vector<std::pair<std::string,std::string> > s;

        s.push_back( std::make_pair("Remarks" , "Segmented Reduction" ));

        s.push_back( std::make_pair("VariantTagShort","RED"));
        s.push_back( std::make_pair("VariantTagLong","Reduction"));

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

#include "CudaFramework/Kernels/JORProxVel/ReductionKernel/ReductionTestVariant.icc"


////////////////////////////////////////////////////////////////////////////////
// demoSegReduceCsr


utilCuda::DeviceMemPtr<double> genStart(size_t count, double min, double max,utilCuda::CudaContext& context) {
	std::vector<double> data(count);
	for(size_t i = 0; i < count; ++i)

     data[i] = 1;
	 return context.malloc(data);
}

utilCuda::DeviceMemPtr<double> genStart(size_t count, std::vector<double> values,utilCuda::CudaContext& context) {
	std::vector<double> data(count);
	for(size_t i = 0; i < count; ++i)

     data[i] = values[i];
	 return context.malloc(data);
}



template <typename PREC, typename Type2>
void demoSegReduceCsr(utilCuda::CudaContext& context, std::vector<Type2>& segmentstarts1, std::vector<PREC>&  values,std::vector<PREC>*  Output,int count,float* elapsedTime){


	const int numSegments = segmentstarts1.size(); /// number of segments => 7 here

	utilCuda::DeviceMemPtr<int> csrDevice = context.malloc(&segmentstarts1[0], numSegments);
    utilCuda::DeviceMemPtr<PREC> valsDevice = genStart(count,values,context);
	utilCuda::DeviceMemPtr<PREC> resultsDevice = context.malloc<PREC>(numSegments);  /// allocate results array

    cudaEvent_t start,stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start,0));

	segReduceCsr<ReductionGPU::Operation::PLUS>(valsDevice->get(),
                                             csrDevice->get(),
                                             count,
                                             numSegments,
                                             false,
                                             resultsDevice->get(),
                                             (PREC)0,
                                             context,
                                             elapsedTime);

    CHECK_CUDA(cudaEventRecord(stop,0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    elapsedTime[0]=0;
    CHECK_CUDA(cudaEventElapsedTime(elapsedTime,start,stop));







	/***  void segReduceCsr(InputIt dataGlobal, CsrIt csrGlobal, int count,
	int numSegments, bool supportEmpty, OutputIt destGlobal, T identity, Op op,
	CudaContext& context)  ***/

	/***

	dataGlobal : intrusive pointer
	csrGlobal  :
	count       : how much input 100
	numSegments : how many segments 7
	supportEmpty: no empty segments
	destGlobal : where to put result
	identity    : 0 element   (int 0)
	op          : plus (+) on integers
	context     : cuda context (gpu address stream etc)


	***/

   ASSERTCHECK_CUDA(resultsDevice->toHost((*Output)));
}

template<typename PREC,typename PREC1>
void cudaContextWrapper( std::vector<PREC1> & inputVectorGPU2,
                   std::vector<PREC> & inputVectorGPU,
                   std::vector<PREC>* outputVectorGPU,
                   int numberOfContacts,
                   float* elapsedTime)
{

            {
                utilCuda::ContextPtrType context = utilCuda::createCudaContextOnDevice(0,false);  /// somehow needs to be false

                context->setActive();
                demoSegReduceCsr(*context,
                                 inputVectorGPU2,
                                 inputVectorGPU,
                                 outputVectorGPU,
                                 numberOfContacts,
                                 elapsedTime);
            }
           utilCuda::destroyDeviceGroup();
}

/**
* @brief GPUTestVariant which computes a contact frame for all contacts.
*/

template<typename PREC, unsigned int TVariantId>
class ReductionGPUVariant {
public:
typedef Eigen::Matrix<PREC,Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor>  VectorType;
typedef Eigen::Matrix<int,Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor>  VectorIntType;
    double m_nops;
    static const unsigned int VariantId2=1;



     static std::string getVariantName() {

          return "Reduction Kernel";
    }

    static std::string getVariantDescriptionShort() {
          return "Calculates one reduction";
    }

    static std::string getVariantDescriptionLong() {


             return "No Settings to chose";


    }
    void initialize(std::ostream* pLog) {

        m_pLog = pLog;
    }

    void finalize() {

    }


    template<typename VectorType, typename VectorIntType>
    void initializeTestProblem( unsigned int numberOfContacts,
                                VectorType & inputVectorGPU,
                                VectorIntType & inputVectorGPU2,
                                VectorType & outputVectorGPU,
                                unsigned int m_implementedLengthInput,
                                unsigned int m_implementedLengthOutput) {

        size_t freeMem;
        size_t totalMem;
        CHECK_CUDA(cudaMemGetInfo (&freeMem, &totalMem));

        int nGPUBytes = (numberOfContacts*m_implementedLengthInput+inputVectorGPU2.size()*m_implementedLengthOutput)*(sizeof(PREC));

        *m_pLog << "Will try to allocate ("<<nGPUBytes<<"/"<<freeMem<<") = " << (double)nGPUBytes/freeMem * 100.0 <<" % of global memory on GPU"<<std::endl;
        if(nGPUBytes > freeMem) {
            *m_pLog <<"To little memory on GPU, exit!..."<<std::endl;
            std::exit(-1);
        }

    }

    template<typename VectorType,typename VectorIntType>
    void run(unsigned int numberOfContacts,
             VectorType & inputVectorGPU,
             VectorIntType & inputVectorGPU2,
             VectorType & outputVectorGPU){

        float elapsedTime[1];
        elapsedTime[0]=0;
        cudaContextWrapper(inputVectorGPU2,
                           inputVectorGPU,
                           &outputVectorGPU,
                           numberOfContacts,
                           elapsedTime);

        m_elapsedTimeCopyToGPU=0;
        m_gpuIterationTime = elapsedTime[0];
        m_elapsedTimeCopyFromGPU=0;
        m_nops=0;
        *m_pLog << " ---> Iteration Time GPU:"<< m_gpuIterationTime <<std::endl;
        *m_pLog << " ---> Copy time from GPU:"<< m_elapsedTimeCopyFromGPU <<std::endl;
        *m_pLog << " ---> Copy time to GPU:"<<  m_elapsedTimeCopyToGPU <<std::endl;
        *m_pLog << " ---> FLOPS GPU  UNKNOWN HERE :" <<std::endl;
    }

    template<int VariantId> inline void runKernel() {

        if(VariantId==1) {
             ERRORMSG(" ONE RUN VARIANT 1")
        } else if(VariantId==2) {
            ERRORMSG(" ONE RUN VARIANT 2")
        }

    }


    void cleanUpTestProblem() {

    }

    double m_gpuIterationTime;
    double m_elapsedTimeCopyToGPU;
    double m_elapsedTimeCopyFromGPU;
    int m_nIter;

    utilCuda::DeviceMemPtr<PREC> resultsDevice;
    utilCuda::DeviceMemPtr<PREC> valsDevice;
    utilCuda::DeviceMemPtr<int> csrDevice;
private:

    std::ostream* m_pLog;

};


#endif // TestSimple2_hpp


/// Random syntax stuff
        //Eigen::MatrixXd A_1;
        //Eigen::MatrixXd* A_2;
        //A_2=&A_1;
        //utilCuda::CudaMatrix<PREC> A_host;
        //utilCuda::mallocMatrixHost(A_host, 3,3, true);
        //Eigen::Matrix<PREC,Eigen::Dynamic, 1> A_3(2,2);
        //A_3.setRandom();
        //A_host.Matrix=A;
        //const Eigen::MatrixBase<double> *A_3;
        //A_3=&A_1;
       // utilCuda::setMatrixRandom(A_host);
        //utilCuda::CudaMatrix<PREC> A_dev;
        //utilCuda::mallocMatrixDevice<true>(A_dev,A_3);
        //utilCuda::copyMatrixToDevice(A_dev,A_3);
        //utilCuda::copyMatrixToHost(A_3,A_dev);
        //utilitiesCuda::mallocMatrixDevice(Matrix<PREC> & A_dev, const Eigen::MatrixBase<Derived> & A);
        //CHECK_CUDA(utilCuda::mallocMatrixDevice<alignMatrix>(A_dev, A));
        //CHECK_CUDA(utilCuda::mallocMatrixDevice<false>(d_dev, d));
        //CHECK_CUDA(utilCuda::mallocMatrixDevice<false>(x_old_dev,x_old));
        //CHECK_CUDA(utilCuda::mallocMatrixDevice<false>(t_dev,t));
        //CHECK_CUDA(utilCuda::mallocMatrixDevice<false>(x_new_dev,x_old_dev.M,x_old_dev.N,false));
        //CHECK_CUDA(cudaMalloc(&pConvergedFlag_dev,sizeof(bool)));

        //ERRORMSG(" INITI HERE DEVICE ARRAYS");  /// initialised in another function ..
