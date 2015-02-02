#ifndef ConvergenceCheck_cuh
#define ConvergenceCheck_cuh



#include <cuda_runtime.h>
#include "CudaError.hpp"
#include "CudaMatrix.hpp"
#include "StaticAssert.hpp"

#include "UtilitiesMatrixVector.cuh"
#include "GPUBufferOffsets.hpp"
#include "DeviceIntrinsics.cuh"
#include "VariantLaunchSettings.hpp"


#define Elem_RowMuInt(_A,_row,_col)                             (     *( (unsigned int*)((char*)(_A.m_pDevice) + (_row) * (_A.m_outerStrideBytes)) + (_col) )     )
#define Elem_ColMuInt(_A,_row,_col)                                Elem_RowMuInt(_A,_col,_row)


namespace ConvCheck {
/**

Convergence check version 1.
Partly optimised for registers

**/



template<bool genOutput,bool convInVel,typename PREC,typename TCudaMatrix,typename TCudaIntMatrix>
__global__ void convCheckKernel(TCudaMatrix outputBuffer,
                                TCudaMatrix bodyBuffer,
                                PREC* redBufferIn,
                                TCudaIntMatrix globalBuffer,
                                unsigned int totalBodyNumber,
                                PREC relTol,
                                PREC absTol) {


    STATIC_ASSERT( utilCuda::CudaMatrixFlags::template isRowMajor<TCudaMatrix::Flags>::value == false )
    STATIC_ASSERT( utilCuda::CudaMatrixFlags::template isRowMajor<TCudaIntMatrix::Flags>::value == false )

    DEFINE_JORPROXVEL_GPUBUFFER_OFFSET_NAMESPACES

    unsigned long int thid=threadIdx.x+blockDim.x*blockIdx.x;


    PREC v;

    PREC m;

    PREC value1;
    PREC value2;

    PREC vNew[6];

    bool test;

    unsigned int u_s1;

    u_s1=B::u1_s;

    while (thid<totalBodyNumber) {


        for(int z=0; z<6; z++) {

            vNew[z]= redBufferIn[thid+(z*totalBodyNumber)] + Elem_ColM(bodyBuffer,thid,u_s1+z); /// new = delta + old

        }


        if(convInVel==true) {
            value1=0;
            value2=0;

            ///get normal vector from data buffer

            m=divDev<PREC>(1,Elem_ColM(bodyBuffer,thid,B::mInv_s));

            // #pragma unroll
            for(int i=0; i<3; i++) {
                v=Elem_ColM(bodyBuffer,thid,u_s1+i);
                value1 += v*m*v;
            }
            // #pragma unroll
            for(int i=0; i<3; i++) {

                m=divDev<PREC>(1,Elem_ColM(bodyBuffer,thid,B::thetaInv_s+i));
                v=Elem_ColM(bodyBuffer,thid,u_s1+i+3);
                value1 += v*m*v;
            }

            m=divDev<PREC>(1,Elem_ColM(bodyBuffer,thid,B::mInv_s));
            // #pragma unroll
            for(int i=0; i<3; i++) {

                v=vNew[i];
                value2 += v*m*v;
            }
            // #pragma unroll
            for(int i=0; i<3; i++) {

                m=divDev<PREC>(1,Elem_ColM(bodyBuffer,thid,B::thetaInv_s+i));
                v=vNew[i+3];
                value2 += v*m*v;
            }

            for(int z=0; z<6; z++) {

                Elem_ColM(bodyBuffer,thid,u_s1+z) =vNew[z]; /// save new value

            }

            test = (std::abs(value1-value2) < (std::abs(relTol*value1)+ 2*absTol));
            if (test) {

                if(genOutput) {
                    Elem_ColM(outputBuffer,thid,0)=0;
                } else {
                    //  do nothing
                }

            } else {

                if(genOutput) {
                    Elem_ColM(outputBuffer,thid,0)=1;
                } else {
                    Elem_ColMuInt(globalBuffer,0,G::conv_s)=1;
                }
            }

            if(thid == 0) {
                if( !genOutput ) {
                    Elem_ColMuInt(globalBuffer,0,G::iter_s)=Elem_ColMuInt(globalBuffer,0,G::iter_s)+1;
                }

            }
        } else {
            for(int z=0; z<6; z++) {

                Elem_ColM(bodyBuffer,thid,u_s1+z) =vNew[z]; /// save new value, even if the convergence check is in lambda

            }
        }

        thid+=gridDim.x*blockDim.x;
        /// ~36 flop
    }

}

template<bool genOutput,bool convInVel,typename PREC,typename TCudaMatrix,typename TCudaIntMatrix>
void convCheckKernelWrap(TCudaMatrix bodyBuffer,
                         TCudaIntMatrix globalBuffer,
                         TCudaMatrix outputBuffer,
                         PREC* redBufferIn,
                         unsigned int numberOfBodies,
                         VariantLaunchSettings variantSettings,
                         PREC relTol,
                         PREC absTol) {




    convCheckKernel<genOutput,convInVel,PREC> <<<variantSettings.numberOfBlocks,variantSettings.numberOfThreads>>>(outputBuffer,
            bodyBuffer,
            redBufferIn,
            globalBuffer,
            numberOfBodies,
            relTol,
            absTol);



    CHECK_CUDA_LAST
}

}





#endif



