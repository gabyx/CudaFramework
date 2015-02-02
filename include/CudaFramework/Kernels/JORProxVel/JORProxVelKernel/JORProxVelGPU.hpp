#ifndef JORProxVelGPU_hpp
#define JORProxVelGPU_hpp

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <random>

#include <boost/format.hpp>

#include "CudaMatrixUtilities.hpp"
#include "CudaMatrix.hpp"
#include "CudaPrint.hpp"
#include "GPUBufferOffsets.hpp"
#include "GeneralStructs.hpp"
#include "VariantLaunchSettings.hpp"



#include "JORProxVelKernelWrap.hpp"





/**
* @brief GPUTestVariant which computes a contact frame for all contacts.
*/

template<typename PREC, unsigned int TVariantId>
class JORProxVelGPUVariant {
public:

    ~JORProxVelGPUVariant() {
        m_context.reset(nullptr);
        utilCuda::destroyDeviceGroup();
    }

    typedef Eigen::Matrix<PREC,Eigen::Dynamic, Eigen::Dynamic,Eigen::ColMajor>  MatrixType;
    typedef Eigen::Matrix<unsigned int,Eigen::Dynamic, Eigen::Dynamic,Eigen::ColMajor>  MatrixUIntType;
    typedef std::vector<PREC> VectorType;
    typedef std::vector<int> VectorIntType;

    unsigned int redLength=1;
    unsigned int m_iterationCount=0;

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
            variantSettings.numberOfBlocks=128;
            variantSettings.numberOfThreads=128;
            break;
        }
    }

    static std::string getVariantName() {

        return "Vel Jor Prox Kernel";
    }

    static std::string getVariantDescriptionShort() {
        return "Calculates one iteration";
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

        /// somehow needs to be false
        m_context = utilCuda::createCudaContextOnDevice(0,false);
        *m_pLog << " Created CudaContext " << std::endl;
        m_context->setActive();
    }


    void initialize() {
        /// somehow needs to be false
        m_context = utilCuda::createCudaContextOnDevice(0,false);
        m_context->setActive();
    }

    void initializeLog(std::ostream* pLog) {

        m_pLog = pLog;
    }


    void finalize() {

    }

    void initializeTestProblem( unsigned int numberOfContacts,
                                MatrixType & bodyBufferGPU,
                                MatrixType & contBufferGPU,
                                MatrixUIntType & globalBufferGPU,
                                MatrixUIntType & indexSetGPU) {

        redLength=12*numberOfContacts;

        size_t freeMem;
        size_t totalMem;

        CHECK_CUDA(cudaMemGetInfo (&freeMem, &totalMem));

        int bodyMatrixSize=bodyBufferGPU.cols()*bodyBufferGPU.rows();

        int globalMatrixSize=globalBufferGPU.cols()*globalBufferGPU.rows();
        int contactMatrixSize=contBufferGPU.cols()*contBufferGPU.rows();



        int indexMatrixSize=indexSetGPU.cols()*indexSetGPU.rows();
        int nGPUBytes = (bodyMatrixSize+redLength+45*numberOfContacts)*(sizeof(PREC))+(sizeof(unsigned int))*(globalMatrixSize+indexMatrixSize);

        *m_pLog << "Will try to allocate ("<<nGPUBytes<<"/"<<freeMem<<") = " << (double)nGPUBytes/freeMem * 100.0 <<" % of global memory on GPU  (CSR not included, anyway very small)"<<std::endl;
        if(nGPUBytes > freeMem) {
            *m_pLog <<"To little memory on GPU, exit!..."<<std::endl;
            std::exit(-1);
        }


        m_bodyDev = m_context->mallocMatrix<PREC,true>( bodyBufferGPU );
        m_contactDev = m_context->mallocMatrix<PREC,true>( contBufferGPU);
        m_indexSetDev = m_context->mallocMatrix<unsigned int,true>( indexSetGPU );
        m_globalDev = m_context->mallocMatrix<unsigned int,false>( globalBufferGPU);

    }


    void initializeCompleteTestProblem( unsigned int elInRedBuffer,
                                        MatrixType & bodyBufferGPU,
                                        MatrixType & contBufferGPU,
                                        MatrixUIntType & globalBufferGPU,
                                        MatrixUIntType & indexSetGPU) {


        size_t freeMem;
        size_t totalMem;

        CHECK_CUDA(cudaMemGetInfo (&freeMem, &totalMem));

        CHECK_CUDA_LAST

        int bodyMatrixSize=bodyBufferGPU.cols()*bodyBufferGPU.rows();

        int globalMatrixSize=globalBufferGPU.cols()*globalBufferGPU.rows();
        int contactMatrixSize=contBufferGPU.cols()*contBufferGPU.rows();

        int indexMatrixSize=indexSetGPU.cols()*indexSetGPU.rows();


        int nGPUBytes = (bodyMatrixSize+elInRedBuffer+45*contBufferGPU.rows())*(sizeof(PREC))+(sizeof(unsigned int))*(globalMatrixSize+indexMatrixSize);

        if(nGPUBytes > freeMem) {
            *m_pLog <<"To little memory on GPU, exit!..."<<std::endl;
            std::exit(-1);
        }

        m_contactDev = m_context->mallocMatrix<PREC,false>( contBufferGPU);
        m_globalDev = m_context->mallocMatrix<unsigned int,false>( globalBufferGPU);
        m_bodyDev = m_context->mallocMatrix<PREC,false>( bodyBufferGPU );
        m_indexSetDev = m_context->mallocMatrix<unsigned int,false>( indexSetGPU );









    }
    unsigned int getLastIterationCount(void) {
        return m_iterationCount;
    }

    template<typename EigenVectorIntType>
    void runJORcomplete(
        MatrixType & bodyBufferGPU,
        MatrixType & contBufferGPU,
        MatrixUIntType & globalBufferGPU,
        MatrixUIntType & indexSetGPU,
        EigenVectorIntType & segmentStarts,
        unsigned int minIt,
        unsigned int maxIt,
        PREC deltaTime,
        PREC relTol,
        PREC absTol,
        unsigned int elInRedBuffer) {

        setSettings();
        variantSettings.var=1;

        DEFINE_JORPROXVEL_GPUBUFFER_OFFSET_NAMESPACES

        unsigned int numberOfContacts=contBufferGPU.rows();
        unsigned int numberOfBodies=bodyBufferGPU.rows();

        float time[1];
        unsigned int isConverged=1;



        CHECK_CUDA(m_indexSetDev->fromHost(indexSetGPU));
        CHECK_CUDA(m_contactDev->fromHost(contBufferGPU));
        CHECK_CUDA(m_bodyDev->fromHost(bodyBufferGPU));
        CHECK_CUDA(m_globalDev->fromHost(globalBufferGPU));





        const int numSegments = segmentStarts.size(); /// number of segments => 7 here

        utilCuda::DeviceMemPtr<int> csrDevice = m_context->malloc(&segmentStarts[0], numSegments);
        utilCuda::DeviceMemPtr<PREC> valsDevice = m_context->malloc<PREC>(elInRedBuffer*6);
        utilCuda::DeviceMemPtr<PREC> resultsDevice = m_context->malloc<PREC>(numSegments*6);  /// allocate results array


///======================================================================///

        utilCuda::DeviceMemPtr<int> limitsDevice=ReductionGPU::segReduceInnerFirst<false,int*,PREC>( csrDevice->get(),
                elInRedBuffer,
                numSegments,
                *m_context,
                time);



        ContactInit::contactInitKernelWrap<false>(   m_bodyDev->get(),
                m_contactDev->get(),
                m_indexSetDev->get(),
                m_globalDev->get(),
                m_contactDev->get(),
                numberOfContacts,
                variantSettings);



        CHECK_CUDA_LAST

        BodyInit::bodyInitKernelWrap<false>(   m_bodyDev->get(),
                                               m_globalDev->get(),
                                               m_contactDev->get(),
                                               numberOfBodies,
                                               variantSettings,
                                               deltaTime  );



        CHECK_CUDA_LAST


        unsigned int iterations=0;

        globalBufferGPU( 0, G::iter_s )=0;
        while(true) {

            globalBufferGPU( 0, G::conv_s )=0;

            m_globalDev->fromHost(globalBufferGPU);

            ContIter::contIterKernelWrap<false,true>( m_bodyDev->get(),
                                                 m_contactDev->get(),
                                                 m_globalDev->get(),
                                                 valsDevice->get(),
                                                 m_indexSetDev->get(),
                                                 m_contactDev->get(),
                                                 numberOfContacts,
                                                 variantSettings,
                                                 elInRedBuffer,
                                                 0,
                                                 0);
            CHECK_CUDA_LAST



            for(int red_i=0; red_i<6; red_i++) {

                ReductionGPU::segReduceInnerSecond<false,ReductionGPU::Operation::Type::PLUS>(  valsDevice->get()+elInRedBuffer*red_i,
                        csrDevice->get(),
                        elInRedBuffer,
                        resultsDevice->get()+numSegments*red_i,
                        PREC(0),
                        *m_context,
                        limitsDevice,
                        time);
                CHECK_CUDA_LAST

            }

            CHECK_CUDA_LAST




            ConvCheck::convCheckKernelWrap<false,true,PREC>(m_bodyDev->get(),
                    m_globalDev->get(),
                    m_bodyDev->get(),
                    resultsDevice->get(),
                    numberOfBodies,
                    variantSettings,
                    relTol,
                    absTol);
            CHECK_CUDA_LAST


            m_globalDev->toHost(globalBufferGPU);

            isConverged=globalBufferGPU(0,G::conv_s);

            if( (isConverged==0 && iterations >= minIt ) || (iterations>maxIt) ) {
                break;
            }

            iterations++; /// dont waste cuda time
        }

        m_bodyDev->toHost(bodyBufferGPU);


///=====================================================================================///

    }

    template<typename EigenVectorIntType>
    void runJORcomplete2(
        MatrixType & bodyBufferGPU,
        MatrixType & contBufferGPU,
        MatrixUIntType & globalBufferGPU,
        MatrixUIntType & indexSetGPU,
        EigenVectorIntType & segmentStarts,
        unsigned int minIt,
        unsigned int maxIt,
        PREC deltaTime,
        PREC relTol,
        PREC absTol,
        unsigned int elInRedBuffer) {

        setSettings();

        variantSettings.var=1;

        DEFINE_JORPROXVEL_GPUBUFFER_OFFSET_NAMESPACES

        unsigned int numberOfContacts=contBufferGPU.rows();
        unsigned int numberOfBodies=bodyBufferGPU.rows();

        float time[1];
        unsigned int isConverged=1;

        CHECK_CUDA(m_indexSetDev->fromHost(indexSetGPU));
        CHECK_CUDA(m_contactDev->fromHost(contBufferGPU));
        CHECK_CUDA(m_bodyDev->fromHost(bodyBufferGPU));
        CHECK_CUDA(m_globalDev->fromHost(globalBufferGPU));

        const int numSegments = segmentStarts.size(); /// number of segments

        std::vector<int> segmentStarts2;

        segmentStarts2.resize(6*numSegments);

        for(int z=0; z<6; z++) {
            for(int y=0; y<numSegments; y++) {
                segmentStarts2[y+numSegments*z] = segmentStarts[y]+z*elInRedBuffer;
            }
        }

        utilCuda::DeviceMemPtr<int> csrDevice = m_context->malloc(&segmentStarts2[0], numSegments*6);
        utilCuda::DeviceMemPtr<PREC> valsDevice = m_context->malloc<PREC>(elInRedBuffer*6);
        utilCuda::DeviceMemPtr<PREC> resultsDevice = m_context->malloc<PREC>(numSegments*6);  /// allocate results array


///======================================================================///

        utilCuda::DeviceMemPtr<int> limitsDevice=ReductionGPU::segReduceInnerFirst<false,int*,PREC>( csrDevice->get(),
                elInRedBuffer*6,
                numSegments*6,
                *m_context,
                time);


        variantSettings.numberOfThreads=64;
        variantSettings.numberOfBlocks=90;

        ContactInit::contactInitKernelWrap<false>(   m_bodyDev->get(),
                m_contactDev->get(),
                m_indexSetDev->get(),
                m_globalDev->get(),
                m_contactDev->get(),
                numberOfContacts,
                variantSettings);



        CHECK_CUDA_LAST

        variantSettings.numberOfThreads=256;
        variantSettings.numberOfBlocks=256;

        BodyInit::bodyInitKernelWrap<false>(   m_bodyDev->get(),
                                               m_globalDev->get(),
                                               m_contactDev->get(),
                                               numberOfBodies,
                                               variantSettings,
                                               deltaTime  );



        CHECK_CUDA_LAST


        unsigned int iterations=0;

        globalBufferGPU( 0, G::iter_s )=0;
        while(true) {

            globalBufferGPU( 0, G::conv_s )=0;

            m_globalDev->fromHost(globalBufferGPU);

            variantSettings.numberOfThreads=256;
            variantSettings.numberOfBlocks=30;

            ContIter::contIterKernelWrap<false,true>( m_bodyDev->get(),
                                                 m_contactDev->get(),
                                                 m_globalDev->get(),
                                                 valsDevice->get(),
                                                 m_indexSetDev->get(),
                                                 m_contactDev->get(),
                                                 numberOfContacts,
                                                 variantSettings,
                                                 elInRedBuffer,
                                                 0,
                                                 0);
            CHECK_CUDA_LAST



            ReductionGPU::segReduceInnerSecond<false,ReductionGPU::Operation::Type::PLUS>(  valsDevice->get(),
                        csrDevice->get(),
                        elInRedBuffer*6,
                        resultsDevice->get(),
                        PREC(0),
                        *m_context,
                        limitsDevice,
                        time);

            CHECK_CUDA_LAST

            variantSettings.numberOfThreads=256;
            variantSettings.numberOfBlocks=45;

            ConvCheck::convCheckKernelWrap<false,true,PREC>(m_bodyDev->get(),
                    m_globalDev->get(),
                    m_bodyDev->get(),
                    resultsDevice->get(),
                    numberOfBodies,
                    variantSettings,
                    relTol,
                    absTol);
            CHECK_CUDA_LAST


            m_globalDev->toHost(globalBufferGPU);

            isConverged=globalBufferGPU(0,G::conv_s);

            if( (isConverged==0 && iterations >= minIt ) || (iterations>maxIt) ) {
                break;
            }

            iterations++; /// dont waste cuda time
        }

        m_bodyDev->toHost(bodyBufferGPU);
        m_iterationCount=iterations;

    }



    void run(unsigned int numberOfContacts,
             MatrixType & bodyBufferGPU,
             MatrixType & contBufferGPU,
             MatrixUIntType & globalBufferGPU,
             VectorType & reductionBufferGPU,
             MatrixUIntType & indexSetGPU,
             VectorIntType & segmentStarts,
             unsigned int numberOfBodies) {

        setSettings() ;
        variantSettings.var=1;
        unsigned int elInRedBuffer=2*numberOfContacts;


        PREC deltaTime=0.1;

        cudaEvent_t start,stop;

        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        DEFINE_JORPROXVEL_GPUBUFFER_OFFSET_NAMESPACES


        *m_pLog <<"Entered GPU run"<<std::endl;

        float time[4];
        unsigned int isConverged=1;


        CHECK_CUDA(cudaEventRecord(start,0));

        CHECK_CUDA(m_indexSetDev->fromHost(indexSetGPU));
        CHECK_CUDA(m_contactDev->fromHost(contBufferGPU));
        CHECK_CUDA(m_bodyDev->fromHost(bodyBufferGPU));
        CHECK_CUDA(m_globalDev->fromHost(globalBufferGPU));

        CHECK_CUDA(cudaEventRecord(stop,0));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(time+1,start,stop));


        *m_pLog <<"Copy Time To Device is: "<< time[1] <<" ms"<< std::endl;

        const int numSegments = segmentStarts.size(); /// number of segments => 7 here

        utilCuda::DeviceMemPtr<int> csrDevice = m_context->malloc(&segmentStarts[0], numSegments);
        utilCuda::DeviceMemPtr<PREC> valsDevice = m_context->malloc<PREC>(6*elInRedBuffer);
        utilCuda::DeviceMemPtr<PREC> resultsDevice = m_context->malloc<PREC>(6*numSegments);  /// allocate results array

        m_reductionDev=valsDevice->get();

        /// allocate memory

        /// execute Kernel

///======================================================================///

        utilCuda::DeviceMemPtr<int> limitsDevice=ReductionGPU::segReduceInnerFirst<false,int*,PREC>( csrDevice->get(),
                elInRedBuffer,
                numSegments,
                *m_context,
                &time[4]);


        ContactInit::contactInitKernelWrap<false>(    m_bodyDev->get(),
                m_contactDev->get(),
                m_indexSetDev->get(),
                m_globalDev->get(),
                m_contactDev->get(),
                numberOfContacts,
                variantSettings);




        BodyInit::bodyInitKernelWrap<false,PREC>(    m_bodyDev->get(),
                m_globalDev->get(),
                m_contactDev->get(),
                numberOfBodies,
                variantSettings ,
                deltaTime);






        unsigned int max_iter=1;
        unsigned int iterations=0;

        while(isConverged==1&&iterations<max_iter) {
            CHECK_CUDA(cudaEventRecord(start,0));
            globalBufferGPU( 0, G::conv_s )=0;
            m_globalDev->fromHost(globalBufferGPU);

            ContIter::contIterKernelWrap<false,true>( m_bodyDev->get(),
                                                 m_contactDev->get(),
                                                 m_globalDev->get(),
                                                 m_reductionDev,
                                                 m_indexSetDev->get(),
                                                 m_contactDev->get(),
                                                 numberOfContacts,
                                                 variantSettings,
                                                 elInRedBuffer,
                                                 0,
                                                 0);

            iterations ++;






            for(int red_i=0; red_i<6; red_i++) {

                ReductionGPU::segReduceInnerSecond<false,ReductionGPU::Operation::Type::PLUS>( valsDevice->get()+elInRedBuffer*red_i,
                        csrDevice->get(),
                        elInRedBuffer,
                        resultsDevice->get()+numSegments*red_i,
                        PREC(0),
                        *m_context,
                        limitsDevice,
                        &time[3]);

            }

            PREC relTol=0.1;
            PREC absTol=1;
            ConvCheck::convCheckKernelWrap<false,true,PREC>(m_bodyDev->get(),
                    m_globalDev->get(),
                    m_bodyDev->get(),
                    resultsDevice->get(),
                    numberOfBodies,
                    variantSettings,
                    relTol,
                    absTol);

            CHECK_CUDA(cudaEventRecord(stop,0));
            CHECK_CUDA(cudaEventSynchronize(stop));
        }

        CHECK_CUDA(m_globalDev->toHost(globalBufferGPU));

        CHECK_CUDA(cudaEventElapsedTime(time,start,stop));

        CHECK_CUDA(cudaEventRecord(start,0));


        CHECK_CUDA(m_globalDev->toHost(globalBufferGPU));
        CHECK_CUDA(valsDevice->toHost(reductionBufferGPU));

        CHECK_CUDA( m_contactDev->toHost(contBufferGPU));

        CHECK_CUDA( m_bodyDev->toHost(bodyBufferGPU));

        CHECK_CUDA(cudaEventRecord(stop,0));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(time+2,start,stop));

        /// Delete Events
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));

        *m_pLog <<"Copy Time To Host is: "<< time[2] <<" ms"<< std::endl;






///=====================================================================================///

        m_elapsedTimeCopyToGPU=double(time[1]);
        m_gpuIterationTime = double(time[0]);
        m_elapsedTimeCopyFromGPU=double(time[2]);

        m_nops=99999;  ///< unknown
        *m_pLog << " ---> Iteration Time GPU:"<< m_gpuIterationTime <<std::endl;
        *m_pLog << " ---> Copy time from GPU:"<< m_elapsedTimeCopyFromGPU <<std::endl;
        *m_pLog << " ---> Copy time to GPU:"<<  m_elapsedTimeCopyToGPU <<std::endl;
        *m_pLog << " ---> FLOPS GPU:"<<  double(m_nops*numberOfContacts/m_gpuIterationTime*1000) <<std::endl;
    }

    template<int VariantId> inline void runKernel() {

        if(VariantId==1) {
            ERRORMSG(" ONE RUN VARIANT 1")
        } else if(VariantId==2) {
            ERRORMSG(" ONE RUN VARIANT 2")
        }

    }

    void cleanUpTestProblem() {

        CHECK_CUDA(cudaFree(m_reductionDev));

    }

    double m_gpuIterationTime;
    double m_elapsedTimeCopyToGPU;
    double m_elapsedTimeCopyFromGPU;
    int m_nIter;

    utilCuda::DeviceMatrixPtr<PREC> m_bodyDev;
    utilCuda::DeviceMatrixPtr<PREC> m_contactDev;
    utilCuda::DeviceMatrixPtr<PREC> m_outputMatrixDev;
    utilCuda::DeviceMatrixPtr<unsigned int> m_globalDev;
    utilCuda::DeviceMatrixPtr<unsigned int> m_indexSetDev;

    PREC* m_reductionDev=nullptr;


private:

    utilCuda::ContextPtrType m_context;

    std::ostream* m_pLog;

};


#endif // TestSimple2_hpp

