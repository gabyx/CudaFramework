#ifndef JORProxVelocityGPUModule_hpp
#define JORProxVelocityGPUModule_hpp

#include <vector>
#include <unordered_map>

#include "TypeDefs.hpp"
#include "LogDefines.hpp"

#include "CollisionData.hpp"
#include "GPUBufferLoadStore.hpp"

#include "ContactGraphNodeData.hpp"

#include "JORProxVelGPU.hpp"


class JorProxVelocityGPUModule{
private:

    DEFINE_DYNAMICSSYTEM_CONFIG_TYPES

    JORProxVelGPUVariant<PREC,5> m_gpuVariant;

    unsigned int m_nContactsInGraph = 0;
    unsigned int m_nSimBodiesInGraph = 0;

    /**
    * \defgroup GPUBuffers Temporary CPU Buffers to upload to the GPU
    * @{
    */
    using MatrixUIntType = typename MyMatrix<unsigned int>::MatrixDynDyn;

    MatrixDynDyn  m_bodyBuffer;       ///< the body buffer
    MatrixDynDyn  m_contBuffer;       ///< the contact buffer
    MatrixUIntType m_globalBuffer;    ///< the global buffer
    MatrixUIntType m_indexBuffer;     ///< the index buffer
    MatrixUIntType m_reductionBuffer; ///< the reduction buffer (only for debug)

    using VectorUIntType = typename MyMatrix<int>::VectorDyn;
    VectorUIntType m_csrReductionBuffer;
    unsigned int m_redBufferLength;

    const unsigned int bodyBufferLength     = JORProxVelGPU::GPUBufferOffsets::BodyBufferOffsets::length;
    const unsigned int contactBufferLength  = JORProxVelGPU::GPUBufferOffsets::ContBufferOffsets::length;
    const unsigned int redBufferLength      = JORProxVelGPU::GPUBufferOffsets::ReductionBufferOffsets::length;
    const unsigned int globalBufferLength   = JORProxVelGPU::GPUBufferOffsets::GlobalBufferOffsets::length;
    const unsigned int indexBufferLength    = JORProxVelGPU::GPUBufferOffsets::IndexBufferOffsets::length;
    /** @} */



    Logging::Log * m_pSolverLog = nullptr; ///< Solver Log

public:

    JorProxVelocityGPUModule(){
        m_gpuVariant.initialize();
    }
    ~JorProxVelocityGPUModule(){}

    void initializeLog( Logging::Log * pSolverLog){
        m_pSolverLog = pSolverLog;


        m_gpuVariant.initializeLog(&std::cout);
    };

    /** General reset function which cleans GPUVariant stuff and other data structures in this class*/
    void reset(){
    }

    /** Get the tradeoff barrier, when the GPU implementation is prevalent in comparision to the CPU
        @param nSimBodies is the number of simulated bodies in the contact graph
        @param nContacts is the number of contacts in the contact graph
        @return true if GPU should be used, false if CPU should be used
    */
    bool computeOnGPU(unsigned int nContactsInGraph, unsigned int nSimBodiesInGraph){
        //TODO
        // decide with these two values if GPU should be used!
        /** GPU IS BETTER FOR ALMOST THE COMPLETE CONTACT RANGE \ cite{Thierry2014}  ;) **/
        return true;
    }

    /**
    * Initialize GPU variant and data structures to be able to run the gpu iteration
    */
    template<typename TContactNodeList, typename TBodyToContactNodeMap>
    bool runOnGPU(const TContactNodeList & contactDataList,
                    const TBodyToContactNodeMap & bodyToNodes,
                    unsigned int minIterations,
                    unsigned int maxIterations,
                    PREC absTol,
                    PREC relTol,
                    PREC alphaProx,
                    PREC deltaT,
                    unsigned int &iterationNumber){



        LOGSLLEVEL1(m_pSolverLog,"---> JORProxVelGPUModule: GPU run started ... "<< std::endl; )
        m_nContactsInGraph = contactDataList.size();
        m_nSimBodiesInGraph = bodyToNodes.size();

        ASSERTMSG(m_nContactsInGraph && m_nSimBodiesInGraph, "No contacts: "<< m_nContactsInGraph  << " or bodies: " << m_nSimBodiesInGraph << " added in m_bodyToContacts" )


        // resize matrices
        m_bodyBuffer.resize(m_nSimBodiesInGraph,bodyBufferLength);
        m_contBuffer.resize(m_nContactsInGraph,contactBufferLength);
        m_globalBuffer.resize(1,globalBufferLength);
        m_csrReductionBuffer.resize(m_nSimBodiesInGraph);
        m_indexBuffer.resize(m_nContactsInGraph,indexBufferLength);

        LOGSLLEVEL1(m_pSolverLog,"---> Init Buffers"<< std::endl; )

        JORProxVelGPU::initializeBuffers(m_globalBuffer,
                                         m_indexBuffer,
                                         m_contBuffer,
                                         m_bodyBuffer,
                                         m_csrReductionBuffer,
                                         m_redBufferLength,
                                         contactDataList,
                                         bodyToNodes,
                                         alphaProx);


        // Print reduction buffer
        LOGSLLEVEL2(m_pSolverLog,"---> Reduction Buffer Size [in length of u]: " << m_redBufferLength << std::endl; )
        LOGSLLEVEL2(m_pSolverLog,"---> Reduction Buffer CSR: " << m_csrReductionBuffer.transpose() << std::endl; )


            // Run the GPU Code (copy, launch kernels, load back)
            // initialize -> gpu memory
            m_gpuVariant.initializeCompleteTestProblem(m_redBufferLength,m_bodyBuffer,m_contBuffer,m_globalBuffer,m_indexBuffer);
            m_gpuVariant.runJORcomplete2(m_bodyBuffer,
                                        m_contBuffer,
                                        m_globalBuffer,
                                        m_indexBuffer,
                                        m_csrReductionBuffer,
                                        minIterations,
                                        maxIterations,
                                        deltaT,
                                        relTol,
                                        absTol,
                                        m_redBufferLength
                                        );

        m_gpuVariant.cleanUpTestProblem();

        iterationNumber=m_gpuVariant.getLastIterationCount();

        LOGSLLEVEL2(m_pSolverLog,"---> Apply body buffer ... "<< std::endl; )
        JORProxVelGPU::applyBodyBuffer(m_bodyBuffer,bodyToNodes,0);
        LOGSLLEVEL1(m_pSolverLog,"---> JORProxVelGPUModule: GPU run finished!"<< std::endl; )
    }



};



#endif // JORProxVelocityGPUModule_hpp
