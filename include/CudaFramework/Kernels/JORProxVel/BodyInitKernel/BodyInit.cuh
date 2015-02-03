// ========================================================================================
//  CudaFramework
//  Copyright (C) 2014 by Gabriel Nützi <nuetzig (at) imes (d0t) mavt (d0t) ethz (d0t) ch>
//
//  This Source Code Form is subject to the terms of the GNU GPL 3.0 licence. 
//  If a copy of the GNU GPL 3.0 was not distributed with this
//  file, you can obtain one at http://opensource.org/licenses/GPL-3.0.
// ========================================================================================

#ifndef CudaFramework_Kernels_JORProxVel_BodyInitKernel_BodyInit_cuh
#define CudaFramework_Kernels_JORProxVel_BodyInitKernel_BodyInit_cuh



#include <cuda_runtime.h>
#include "CudaFramework/Kernels/JORProxVel/VariantLaunchSettings.hpp"
#include "CudaFramework/General/StaticAssert.hpp"
#include "CudaFramework/CudaModern/CudaError.hpp"
#include "CudaFramework/CudaModern/CudaMatrix.hpp"

#include "CudaFramework/Kernels/JORProxVel/UtilitiesMatrixVector.cuh"
#include "CudaFramework/Kernels/JORProxVel/GPUBufferOffsets.hpp"




#define Elem_RowMuInt(_A,_row,_col)                             (     *( (unsigned int*)((char*)(_A.m_pDevice) + (_row) * (_A.m_outerStrideBytes)) + (_col) )     )
#define Elem_ColMuInt(_A,_row,_col)                                Elem_RowMuInt(_A,_col,_row)



namespace BodyInit{

template <typename MatrixType>
inline bool isSecondBodySimulated(MatrixType& indexSetBuffer,unsigned int thid){

DEFINE_JORPROXVEL_GPUBUFFER_OFFSET_NAMESPACES
if(Elem_ColMuInt(indexSetBuffer,thid,I::b1Idx_s)==Elem_ColMuInt(indexSetBuffer,thid,I::b2Idx_s)){
    return true;
}else{
return false;}
}



template<bool TOutputBool,typename PREC,typename TCudaMatrix,typename TCudaIntMatrix>
__global__ void bodyInitKernel( TCudaMatrix outputBuffer,
                                TCudaMatrix bodyBuffer,
                                TCudaIntMatrix globalBuffer,
                                unsigned int totalBodyNumber,
                                PREC deltaTime) {



    /**
    Kernel Calculates:
    u_0=u_s+ Minv*h*deltaT   ( where h are the external forces )
    **/

   /**

   DESCRIPTION: This version is the simplest, not optimizations regarding the number of registers have been included.
   Very straight forward approach -> load all, calculate all, load all back. Is Used for referencing.

   */

    DEFINE_JORPROXVEL_GPUBUFFER_OFFSET_NAMESPACES

    STATIC_ASSERT( utilCuda::CudaMatrixFlags::template isRowMajor<TCudaMatrix::Flags>::value == false )
    STATIC_ASSERT( utilCuda::CudaMatrixFlags::template isRowMajor<TCudaIntMatrix::Flags>::value == false )


    unsigned long int thid=threadIdx.x+blockDim.x*blockIdx.x;


    PREC v[3]; ///< translatorical velocity
    PREC omega[3];  ///< rotational velocity

    PREC mInv; ///< inverse of the mass

    PREC thetaInv[3]; ///< inverse of the thetas (element wise)


    PREC force[3]; ///< external force
    PREC moment[3]; ///< external torque

    unsigned int uIdx1;  ///< index indicating the active velocity buffer

    uIdx1=B::u1_s;

    while (thid<totalBodyNumber) {

        ///Initialise

        for (int i=0; i<B::f_l; i++) {

                /**
                Load the _all_ data into the Buffers
                **/
            v[i]=Elem_ColM(bodyBuffer,thid,uIdx1+i);
            omega[i]=Elem_ColM(bodyBuffer,thid,uIdx1+B::omegaOff+i);
            thetaInv[i]=Elem_ColM(bodyBuffer,thid,B::thetaInv_s+i);
            force[i]=Elem_ColM(bodyBuffer,thid,B::f_s+i);
            moment[i]=Elem_ColM(bodyBuffer,thid,B::tq_s+i);
        }



        mInv=Elem_ColM(bodyBuffer,thid,B::mInv_s);  ///< get m inverse (Minv)


        MatVecUtil::vecMultScalar<PREC,B::f_l>(force,(deltaTime*mInv));///<  fexternal*deltaT*Minv
        MatVecUtil::vecAdd<PREC,B::f_l>(v,force,v); ///< v0=vs+f*Minv*deltaT
        for (int z=0; z<B::thetaInv_l; z++) {
           omega[z]=deltaTime*(thetaInv[z])*moment[z]+omega[z];  ///< omega0=omegas+torque*Thetainv*deltaT
        }

        if(TOutputBool) {
            /// write into outputBuffer
            for (int j=0; j<3; j++) {
                Elem_ColM(outputBuffer,thid,j)=v[j];
                Elem_ColM(outputBuffer,thid,B::omegaOff+j)=omega[j];
            }
        } else{
            for (int j=0; j<3; j++) {
                Elem_ColM(bodyBuffer,thid,uIdx1+j)=v[j];
                Elem_ColM(bodyBuffer,thid,uIdx1+B::omegaOff+j)=omega[j];
            }
        }


        thid+=gridDim.x*blockDim.x;
        /// ~24 flop
    }

}


template<bool TOutputBool,typename PREC,typename TCudaMatrix,typename TCudaIntMatrix>
__global__ void bodyInitKernel2( TCudaMatrix outputBuffer,
                                TCudaMatrix bodyBuffer,
                                TCudaIntMatrix globalBuffer,
                                unsigned int totalBodyNumber,
                                PREC deltaTime) {
    /**

    Slightly optimized kernel
    First the translatorical velocities are updated then the rotational (saves some registers)


    **/

    DEFINE_JORPROXVEL_GPUBUFFER_OFFSET_NAMESPACES

    STATIC_ASSERT( utilCuda::CudaMatrixFlags::template isRowMajor<TCudaMatrix::Flags>::value == false )
    STATIC_ASSERT( utilCuda::CudaMatrixFlags::template isRowMajor<TCudaIntMatrix::Flags>::value == false )

    unsigned long int thid=threadIdx.x+blockDim.x*blockIdx.x;



    PREC u[3];


    PREC mInv;

    PREC thetaInv[3];


    PREC force[3];


    unsigned int uIdx1;
    uIdx1=B::u1_s;

    while (thid<totalBodyNumber) {

        ///get normal vector from data buffer
        ///Initialise
        ///delta_t=Elem_ColM(globalBuffer,thid,G::deltaT_s);

        /// Test zeile


        for (int i=0; i<B::omegaOff; i++) {
            u[i]=Elem_ColM(bodyBuffer,thid,uIdx1+i);
            thetaInv[i]=Elem_ColM(bodyBuffer,thid,B::thetaInv_s+i);
            force[i]=Elem_ColM(bodyBuffer,thid,B::f_s+i);
        }


        /// Calculate
        mInv=Elem_ColM(bodyBuffer,thid,B::mInv_s);

        MatVecUtil::vecMultScalar<PREC,B::f_l>(force,(deltaTime*mInv));
        MatVecUtil::vecAdd<PREC,B::f_l>(u,force,u);

                if(TOutputBool) {
            /// write into outputBuffer
            for (int j=0; j<B::omegaOff; j++) {
                Elem_ColM(outputBuffer,thid,j)=u[j];
            }
        } else{
            for (int j=0; j<B::omegaOff; j++) {
                Elem_ColM(bodyBuffer,thid,uIdx1+j)=u[j];
            }
        }


        for (int i=0; i<(B::u1_l-B::omegaOff); i++) {
            u[i]=Elem_ColM(bodyBuffer,thid,uIdx1+B::omegaOff+i);
            force[i]=Elem_ColM(bodyBuffer,thid,B::tq_s+i);

        }


        for (int z=0; z<(B::u1_l-B::omegaOff); z++) {
           u[z]=deltaTime*(thetaInv[z])*force[z]+u[z];

        }

        if(TOutputBool) {
            /// write into outputBuffer
            for (int j=0; j<(B::u1_l-B::omegaOff); j++) {
                Elem_ColM(outputBuffer,thid,B::omegaOff+j)=u[j];
            }
        } else{
            for (int j=0; j<(B::u1_l-B::omegaOff); j++) {
                Elem_ColM(bodyBuffer,thid,uIdx1+B::omegaOff+j)=u[j];
            }
        }


        thid+=gridDim.x*blockDim.x;
        /// ~24 flop
    }

}


    template<bool TOutputBool,typename PREC,typename TCudaMatrix,typename TCudaIntMatrix>
    __global__ void bodyInitKernel3( TCudaMatrix outputBuffer,
    TCudaMatrix bodyBuffer,
    TCudaIntMatrix globalBuffer,
    unsigned int totalBodyNumber,
    PREC deltaTime) {


    /**

    Fully register reduced kernel
    all updates are followed by a loading and storing step that way the number of variables can be minimized,
    however alot of load and store options are interrupted by calculations


    **/

        DEFINE_JORPROXVEL_GPUBUFFER_OFFSET_NAMESPACES

        STATIC_ASSERT( utilCuda::CudaMatrixFlags::template isRowMajor<TCudaMatrix::Flags>::value == false )
        STATIC_ASSERT( utilCuda::CudaMatrixFlags::template isRowMajor<TCudaIntMatrix::Flags>::value == false )

        unsigned long int thid=threadIdx.x+blockDim.x*blockIdx.x;


        PREC u;

        PREC mInv;

        PREC force;


        unsigned int uIdx1;

        uIdx1=B::u1_s;

        while (thid<totalBodyNumber) {

            /// Test zeile

               mInv=Elem_ColM(bodyBuffer,thid,B::mInv_s);
            #pragma unroll
            for (int i=0; i<B::omegaOff; i++) {
            ///<calculate the updates for the translatorical velocities
                u=Elem_ColM(bodyBuffer,thid,uIdx1+i);
                force=Elem_ColM(bodyBuffer,thid,B::f_s+i);
                u=u+force*deltaTime*mInv;

                if(TOutputBool) {
                    Elem_ColM(outputBuffer,thid,i)=u;
                } else {
                    Elem_ColM(bodyBuffer,thid,uIdx1+i)=u;
                }
            }

             #pragma unroll
            for (int i=0; i<(B::u1_l-B::omegaOff); i++){
             ///<calculate the updates for the rotational velocities

                mInv=Elem_ColM(bodyBuffer,thid,B::thetaInv_s+i);

                u=Elem_ColM(bodyBuffer,thid,uIdx1+B::omegaOff+i);
                force=Elem_ColM(bodyBuffer,thid,B::tq_s+i);
                u=deltaTime*(mInv)*force+u;


                if(TOutputBool) {
                    Elem_ColM(outputBuffer,thid,B::omegaOff+i)=u;
                } else {
                    Elem_ColM(bodyBuffer,thid,uIdx1+i+B::omegaOff)=u;
                }
            }




            thid+=gridDim.x*blockDim.x;
            /// ~24 flop
        }

    }

template<bool generateOutput,typename PREC,typename TSettings ,typename TCudaMatrix,typename TCudaIntMatrix>
void bodyInitKernelWrap(TCudaMatrix bodyBuffer,
                        TCudaIntMatrix globalBuffer,
                        TCudaMatrix outputBuffer,
                        unsigned int numberOfBodies,
                        TSettings variantSettings,
                        PREC deltaTime) {

        switch(variantSettings.var) {
        case 1:
            bodyInitKernel<generateOutput,PREC> <<<variantSettings.numberOfBlocks,variantSettings.numberOfThreads>>>(   outputBuffer,
            bodyBuffer,
            globalBuffer,
            numberOfBodies,
            deltaTime);
            break;
        case 2:

            bodyInitKernel2<generateOutput,PREC> <<<variantSettings.numberOfBlocks,variantSettings.numberOfThreads>>>(   outputBuffer,
            bodyBuffer,
            globalBuffer,
            numberOfBodies,
            deltaTime);
            break;

        case 3:
            bodyInitKernel3<generateOutput,PREC> <<<variantSettings.numberOfBlocks,variantSettings.numberOfThreads>>>(   outputBuffer,
            bodyBuffer,
            globalBuffer,
            numberOfBodies,
            deltaTime);
            break;

        default:

            bodyInitKernel<generateOutput,PREC> <<<variantSettings.numberOfBlocks,variantSettings.numberOfThreads>>>(   outputBuffer,
            bodyBuffer,
            globalBuffer,
            numberOfBodies,
            deltaTime);
            break;
        }
      CHECK_CUDA_LAST


    return;
}

}

#endif



