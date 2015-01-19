// ========================================================================================
//  CudaFramework
//  Copyright (C) 2014 by Gabriel Nützi <nuetzig (at) imes (d0t) mavt (d0t) ethz (d0t) ch>
//
//  This Source Code Form is subject to the terms of the GNU GPL 3.0 licence. 
//  If a copy of the GNU GPL 3.0 was not distributed with this
//  file, you can obtain one at http://opensource.org/licenses/GPL-3.0.
// ========================================================================================
#ifndef CudaFramework_Kernels_JORProxVel_JORProxVelocityGPUModule_hpp
#define CudaFramework_Kernels_JORProxVel_JORProxVelocityGPUModule_hpp

#include <vector>
#include <unordered_map>

#include "TypeDefs.hpp"
#include "LogDefines.hpp"

#include "CollisionData.hpp"
#include "CudaFramework/Kernels/JORProxVel/GPUBufferOffsets.hpp"
#include "CudaFramework/Kernels/JORProxVel/GPUBufferLoadStore.hpp"

#include "ContactGraphNodeData.hpp"


class JorProxVelocityGPUModule{
private:

    DEFINE_DYNAMICSSYTEM_CONFIG_TYPES


    unsigned int m_nContactsInGraph = 0;
    unsigned int m_nSimBodiesInGraph = 0;

    /**
    * \defgroup GPUBuffers Temporary CPU Buffers to upload to the GPU
    * @{
    */
    using MatrixUIntType = typename MyMatrix<unsigned int>::MatrixDynDyn;
    //TODO change vector sized data to MyMatrix<unsigned int>::::VectorDyn

    MatrixDynDyn  m_bodyBuffer;       ///< the body buffer
    MatrixDynDyn  m_contBuffer;       ///< the contact buffer
    MatrixUIntType m_globalBuffer;    ///< the global buffer
    MatrixUIntType m_indexBuffer;     ///< the index buffer
    MatrixUIntType m_reductionBuffer; ///< the reduction buffer (only for debug)

    using VectorUIntType = typename MyMatrix<unsigned int>::VectorDyn;
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

    JorProxVelocityGPUModule(){}
    ~JorProxVelocityGPUModule(){}

    void initializeLog( Logging::Log * pSolverLog){
        m_pSolverLog = pSolverLog;
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
        return true;
    }

    /**
    * Initialize GPU variant and data structures to be able to run the gpu iteration
    */
    template<typename TContactNodeList, typename TBodyToContactNodeMap>
    bool runOnGPU(const TContactNodeList & contactDataList,
                    const TBodyToContactNodeMap & bodyToNodes,
                    PREC alphaProx ){

        LOG(m_pSolverLog,"---> JORProxVelGPUModule: GPU run started ... "<< std::endl; )
        m_nContactsInGraph = contactDataList.size();
        m_nSimBodiesInGraph = bodyToNodes.size();

        ASSERTMSG(m_nContactsInGraph && m_nSimBodiesInGraph, "No contacts: "<< m_nContactsInGraph  << " or bodies: " << m_nSimBodiesInGraph << " added in m_bodyToContacts" )


        // resize matrices
        m_bodyBuffer.resize(m_nSimBodiesInGraph,bodyBufferLength);
        m_contBuffer.resize(m_nContactsInGraph,contactBufferLength);
        m_globalBuffer.resize(1,globalBufferLength);
        m_csrReductionBuffer.resize(m_nSimBodiesInGraph);
        m_indexBuffer.resize(m_nContactsInGraph,indexBufferLength);

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
        //TODO (MAKE LOGLEVEL1 macro)
        LOG(m_pSolverLog,"---> Reduction Buffer Size [in length of u]: " << m_redBufferLength << std::endl; )
        LOG(m_pSolverLog,"---> Reduction Buffer CSR: " << m_csrReductionBuffer.transpose() << std::endl; )

        //Resize Reduction buffer (only for load back, debug)
        m_reductionBuffer.resize(1,redBufferLength);


        // Run the GPU Code (copy, launch kernels, load back)

        // TODO (use the instantiated (not here) GPUVariant class to launch the kernel wrapper with the CudaContext)

        // Apply all velocities to the bodies
        //TODO The last number determines which body velocity buffer should be applied
        LOG(m_pSolverLog,"---> Apply body buffer ... "<< std::endl; )
        JORProxVelGPU::applyBodyBuffer(m_bodyBuffer,bodyToNodes,0);
        LOG(m_pSolverLog,"---> JORProxVelGPUModule: GPU run finished!"<< std::endl; )
    }



};



#endif // JORProxVelocityGPUModule_hpp
