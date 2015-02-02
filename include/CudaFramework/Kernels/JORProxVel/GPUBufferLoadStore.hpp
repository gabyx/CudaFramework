
#ifndef GPUBufferLoadStore_hpp
#define GPUBufferLoadStore_hpp

#include <type_traits>
#include "TypeDefs.hpp"
#include "StaticAssert.hpp"
#include "AssertionDebug.hpp"

#include "GPUBufferOffsets.hpp"

#include "EnumClassHelper.hpp"
#include "ContactModels.hpp"


/** Layout for the JORProxVelGPU Buffers
*  Abbreviations:  *_s  : offset start,
*                  *_l  :  offset length
*  Offsets always the size of the underlying type of the corresponding buffer matrices
*/
namespace JORProxVelGPU{


    /**
    * Initialize the JORProxVelGPU Buffers
    * @param[in] bodyToNodesMap are only simulated bodies who are in contact
    */
    template<typename PREC,
             typename MatrixType,
             typename MatrixUIntType,
             typename VectorUIntType,
             typename ContactDataListType,
             typename BodyToContactMapType>
    void initializeBuffers(MatrixUIntType & globalBuffer,
                           MatrixUIntType & indexSet,
                           MatrixType & contBuffer,
                           MatrixType & bodyBuffer,
                           VectorUIntType & csrReductionBuffer,
                           unsigned int & reductionBufferLength,
                           const ContactDataListType  & contactNodeList,
                           const BodyToContactMapType & bodyToNodesMap,
                           PREC alphaProx)
    {
        DEFINE_JORPROXVEL_GPUBUFFER_OFFSET_NAMESPACES
        DEFINE_MATRIX_TYPES_OF(PREC)

        // Safety check
        ASSERTMSG( globalBuffer.rows() == 1  &&  globalBuffer.cols() == G::length, "Wrong Size");
        ASSERTMSG( indexSet.rows()  == contactNodeList.size()   &&  indexSet.cols()     == I::length, "Wrong Size");
        ASSERTMSG( contBuffer.rows() == contactNodeList.size()   &&  contBuffer.cols()   == C::length, "Wrong Size");
        ASSERTMSG( bodyBuffer.rows() == bodyToNodesMap.size()    &&  bodyBuffer.cols()   == B::length, "Wrong Size");
        ASSERTMSG( csrReductionBuffer.rows() == bodyToNodesMap.size()    &&  csrReductionBuffer.cols()   == 1, "Wrong Size");



        // fill body buffer
        std::unordered_map<const void * , unsigned int> bodyToIdx; // idx in bodyBuffer
        unsigned int i=0;
        for(auto & d : bodyToNodesMap) {
             auto bodyPtr = d.first;
             bodyBuffer.template block<1,B::u1_l>(i,B::u1_s)       = bodyPtr->m_pSolverData->m_uBuffer.m_back;  // velocity
             bodyBuffer.template block<1,B::q_l>(i,B::q_s)         = bodyPtr->m_q_KI;                           // quaternion
             bodyBuffer.template block<1,B::f_l>(i,B::f_s)         = bodyPtr->m_h_term.template head<3>();      // external force
             bodyBuffer.template block<1,B::tq_l>(i,B::tq_s)       = bodyPtr->m_h_term.template tail<3>();      // external torque
             bodyBuffer.template block<1,B::thetaInv_l>(i,B::thetaInv_s) = bodyPtr->m_K_Theta_S.array().inverse();    // theta inverse
             bodyBuffer(i,B::mInv_s)                                  = 1.0/bodyPtr->m_mass;                       // mass inverse
             bodyToIdx.emplace(bodyPtr,i);
             i++;
        }

        // fill contact buffer
        i=0;
        std::unordered_map<const void * , unsigned int> nodeToIdx; // idx in contBuffer
        for(auto & c : contactNodeList) {

            auto & nodeData = c->m_nodeData;
            auto & pCollData = nodeData.m_pCollData;


            if(nodeData.m_contactParameter.m_contactModel == ContactModels::Enum::UCF){

            }else{
                ERRORMSG("This GPU Buffer initialization function works only for UCF contact models!")
            }

            using CMT = typename CONTACTMODELTYPE(ContactModels::Enum::UCF);


            contBuffer.template block<1,C::lambda_l>(i,C::lambda_s)    = Vector3(0,0,0); ///< lambda old is zero at start !
            contBuffer.template block<1,C::n_l>(i,C::n_s)              = pCollData->m_cFrame.m_e_z;
            contBuffer.template block<1,C::rSC1_l>(i,C::rSC1_s)        = pCollData->m_r_S1C1;
            contBuffer.template block<1,C::rSC2_l>(i,C::rSC2_s)        = pCollData->m_r_S2C2;
            contBuffer.template block<1,C::chi_l>(i,C::chi_s)          = nodeData.m_chi;
            contBuffer.template block<1,C::eps_l>(i,C::eps_s)          = nodeData.m_eps;       ///< only the diag of eps

            STATIC_ASSERT(C::alpha_l == 1 && C::mu_l == 1)
            contBuffer(i,C::alpha_s)                                   = alphaProx;
            contBuffer(i,C::mu_s)                                      = nodeData.m_contactParameter.m_params[CMT::muIdx];

            // insert bodyIdx into index set

            auto mode1 = pCollData->m_pBody1->m_eMode; auto mode2 = pCollData->m_pBody2->m_eMode;

            STATIC_ASSERT(I::b1Idx_l == 1 && I::b2Idx_l == 1)
             using RigidBodyType = typename std::remove_pointer<decltype(pCollData->m_pBody1)>::type;
            if( mode1 == RigidBodyType::BodyMode::SIMULATED && mode2 == RigidBodyType::BodyMode::SIMULATED){
                indexSet(i,I::b1Idx_s) = bodyToIdx[pCollData->m_pBody1];
                indexSet(i,I::b2Idx_s) = bodyToIdx[pCollData->m_pBody2];
            }else if (mode1 == RigidBodyType::BodyMode::SIMULATED &&
                     (mode2 == RigidBodyType::BodyMode::ANIMATED || mode2 == RigidBodyType::BodyMode::STATIC)){
                indexSet(i,I::b1Idx_s) = bodyToIdx[pCollData->m_pBody1];
                indexSet(i,I::b2Idx_s) = indexSet(i,I::b1Idx_s);
            }else if (mode2 == RigidBodyType::BodyMode::SIMULATED &&
                     (mode1 == RigidBodyType::BodyMode::ANIMATED || mode1 == RigidBodyType::BodyMode::STATIC)){
                indexSet(i,I::b1Idx_s) = bodyToIdx[pCollData->m_pBody2];
                indexSet(i,I::b2Idx_s) = indexSet(i,I::b1Idx_s);
            }else{
                ERRORMSG("body1 state: " << EnumConversion::toIntegral(mode1) << " body2 state: " << EnumConversion::toIntegral(mode2) <<"!");
            }

            nodeToIdx.emplace(c,i);

            //
            ++i;
        }


        // Generate csr for the velocity reduction buffer
        csrReductionBuffer[0] = 0;
        // take care: the iteration order of bodyToNodesMap defines the body index order!!!
        auto p = bodyToNodesMap.begin(); auto itEnd = bodyToNodesMap.end();
        i = 0;
        for( ;std::next(p) != itEnd; ++p){ // p is std::pair<bodyPtr, contactNodeList>
            ++i;
            csrReductionBuffer[i] = csrReductionBuffer[i-1] + p->second.size(); // get each bodies nodeList size
        }
        reductionBufferLength = csrReductionBuffer[i] + p->second.size();

        // Iterate over all contact nodes and set the reduction buffer idx in the corresponding contact node
        unsigned int redIdx = 0;
        p = bodyToNodesMap.begin(); itEnd = bodyToNodesMap.end();
        for( ;p != itEnd; ++p){
         //Init reduction idx in indexBuffer
            for( auto & nodePtr : p->second){
                auto it = nodeToIdx.find(nodePtr);
                // it->second  is the index in the contactBuffer for this contact
                ASSERTMSG(it != nodeToIdx.end(),"NodePtr not found!");
                ASSERTMSG(it->second >= 0 && it->second < contactNodeList.size(), "wrong idx");
                // Check if this body is body 1 or body 2
                if(p->first == nodePtr->m_nodeData.m_pCollData->m_pBody1){
                    indexSet(it->second ,I::redIdx_s + 0) = redIdx;
                }else{
                    ASSERTMSG(p->first == nodePtr->m_nodeData.m_pCollData->m_pBody2 , "Body pointer : " << p->first << "not found in contact: " << nodePtr);
                    indexSet(it->second ,I::redIdx_s + 1) = redIdx;
                }
                redIdx++;
            }
        }

        //Initialize global buffer
        globalBuffer.setZero();

    };


    /**
    * Apply the body buffer to the bodies
    * Apply all converged velocities in  bodyBuffer to the velocity of the bodies.
    * @param[in] velocityBufferIdx is 0 to apply the first body buffer and 1 to apply the second!
    */
    template<typename MatrixType,
             typename BodyToContactMapType>
    void applyBodyBuffer(  MatrixType & bodyBuffer,
                           BodyToContactMapType & bodyToNodesMap,
                           unsigned int velocityBufferIdx)
    {

        DEFINE_JORPROXVEL_GPUBUFFER_OFFSET_NAMESPACES
        ASSERTMSG(velocityBufferIdx == 0 || velocityBufferIdx == 1, "Idx needs to be 0 or 1")

        using RigidBodyPtrType = typename std::remove_const<decltype(bodyToNodesMap.begin()->first)>::type; // remove const from key in map (which is const)
        RigidBodyPtrType bodyPtr;

        unsigned int i=0;
        if(velocityBufferIdx==0) // if switch outside to be faster inside!
            for(auto & d : bodyToNodesMap) {
                 bodyPtr = d.first;
                 bodyPtr->m_pSolverData->m_uBuffer.m_front = bodyBuffer.template block<1,B::u1_l>(i,B::u1_s);  // velocity
                 i++;
            }
        else{
            for(auto & d : bodyToNodesMap) {
                 auto bodyPtr = d.first;
                 bodyPtr->m_pSolverData->m_uBuffer.m_front = bodyBuffer.template block<1,B::u1_l>(i,B::u1_s);
                 i++;
            }
        }
    }

};


#endif // GPUBufferLoadStore
