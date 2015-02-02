#ifndef ContactIteration_cuh
#define ContactIteration_cuh



#include <cuda_runtime.h>
#include "CudaFramework/CudaModern/CudaError.hpp"
#include "CudaFramework/CudaModern/CudaMatrix.hpp"
#include "CudaFramework/General/StaticAssert.hpp"
#include "CudaFramework/Kernels/JORProxVel/UtilitiesMatrixVector.cuh"
#include "CudaFramework/Kernels/JORProxVel/GPUBufferOffsets.hpp"
#include "CudaFramework/Kernels/JORProxVel/DeviceIntrinsics.cuh"
#include "CudaFramework/Kernels/JORProxVel/VariantLaunchSettings.hpp"

#define Elem_RowMuInt(_A,_row,_col)                             (     *( (unsigned int*)((char*)(_A.m_pDevice) + (_row) * (_A.m_outerStrideBytes)) + (_col) )     )
#define Elem_ColMuInt(_A,_row,_col)                                Elem_RowMuInt(_A,_col,_row)



/*** ALL MATRICES ARE SAVED IN COLUMN MAJOR FORM !   ***/
namespace ContIter{


template<bool genOutput,bool convInVel,typename TCudaMatrix,typename TCudaIntMatrix,typename TOutBuffer>
__global__ void contIterKernel(     TOutBuffer outputBuffer,
                                    TCudaMatrix bodyBuffer,
                                    TCudaMatrix contBuffer,
                                    TCudaIntMatrix globalBuffer,
                                    typename TCudaMatrix::PREC* redBuffer,
                                    TCudaIntMatrix indexBuffer,
                                    unsigned int totalContactNumber,
                                    unsigned int totalRedNumber,
                                    typename TCudaMatrix::PREC relTol,
                                    typename TCudaMatrix::PREC absTol) {




    typedef typename TCudaMatrix::PREC PREC;


    STATIC_ASSERT( utilCuda::CudaMatrixFlags::template isRowMajor<TCudaMatrix::Flags>::value == false )
    STATIC_ASSERT( utilCuda::CudaMatrixFlags::template isRowMajor<TCudaIntMatrix::Flags>::value == false )

    DEFINE_JORPROXVEL_GPUBUFFER_OFFSET_NAMESPACES

    unsigned long int thid=threadIdx.x+blockDim.x*blockIdx.x;


    PREC wBase[9];
    PREC wR1[9];
    PREC wR2[9];

    PREC invR[3];

    PREC mu[1];

    PREC generalPurpose[9];    ///

    unsigned int bodyIdx1;
    unsigned int bodyIdx2;

    unsigned int uIdx;

    unsigned int redIndex1;
    unsigned int redIndex2;

    bool isBody2Simulated;

    uIdx=B::u1_s;

    while (thid<totalContactNumber) {




         /** load the reduction indexes for body 1 and body 2 **/

        redIndex1=Elem_ColMuInt(indexBuffer,thid,I::redIdx_s);
        redIndex2=Elem_ColMuInt(indexBuffer,thid,I::redIdx_s+1);

        /** load the body indexes for body 1 and body 2 **/

        bodyIdx1 = Elem_ColMuInt(indexBuffer,thid,I::b1Idx_s);
        bodyIdx2 = Elem_ColMuInt(indexBuffer,thid,I::b2Idx_s);

        if(bodyIdx1==bodyIdx2) {
                isBody2Simulated = false;
            } else{
                isBody2Simulated = true;
            }


        /** load the friction coefficient **/
        mu[0]=Elem_ColM(contBuffer,thid,C::mu_s);


        for(int i=0; i<9; i++){

             /** load the data needed to constrtuct mat_W1 and mat_W2 **/
             wBase[i]=Elem_ColM(contBuffer,thid,C::w1_s+i);
             wR1[i]=Elem_ColM(contBuffer,thid,C::w1r_s+i);
             wR2[i]=Elem_ColM(contBuffer,thid,C::w2r_s+i);
        }



            for(int i=0; i<6; i++) {
                /** generalPurpose buffer  (GP)  [3-8] is now u body 1 **/
                generalPurpose[i+3]=Elem_ColM(bodyBuffer,bodyIdx1,(uIdx+i));///< u_0 expected not u_start
            }


         /// uk_body1 => generalPurpose   [3:8]
        /// Wbody1 * uk_body1 => generalPurpose   [0:2]

        MatVecUtil::calcWtransU<PREC,true>(wBase, wR1,generalPurpose+3,generalPurpose);
        ///< nflop = 36

        for(int i=0; i<3; i++) {
             /** generalPurpose buffer  (GP)  [3-5] is now b **/
            generalPurpose[i+3]=Elem_ColM(contBuffer,thid,C::b_s+i);
        }


          /** generalPurpose buffer  (GP)  [0-2] is now b+Wtranspose1*u1 **/
        MatVecUtil::vecAdd<PREC,3>(generalPurpose+3,generalPurpose,generalPurpose);
            ///< nflop = 39

        if(isBody2Simulated){
                 ///< check if the second body is simulated if not we know that minv2 is 0 and u2 is 0 => Wb2 and Wr2 and not needed anymore and there will be no deltau2 !

              for(int i=0; i<6; i++) {
             /** generalPurpose buffer  (GP)  [3-8] is now ubody 2 **/
            generalPurpose[i+3]=Elem_ColM(bodyBuffer,bodyIdx2,(uIdx+i));///< u_0 expected not u_start

        }

         /** invR buffer  [0-2] is now wtranspose2 * ubody 2 **/
        MatVecUtil::calcWtransU<PREC,false>(wBase, wR2,generalPurpose+3,invR);
        ///< nflop =75

         }else{
             for (int i=0;i<3;i++){
                invR[i]=0;
             }
         }

         /** generalPurpose buffer  (GP)  [0-2] is now     = Wbody1 * uk_body1 + Wbody2 * uk_body2 + b  **/
        /** generalPurpose buffer  (GP)  [3-8] is now     free  **/
        MatVecUtil::vecAdd<PREC,3>(invR,generalPurpose,generalPurpose);
         ///< nflop =78

        for(int i=0; i<3; i++) {
            /** load R inverse it is needed for the prox **/
            invR[i]=Elem_ColM(contBuffer,thid,C::r_s+i);

        }

        MatVecUtil::vecMultElWise<PREC,3>(generalPurpose,invR,generalPurpose);
        MatVecUtil::vecMultScalar<PREC,3>(generalPurpose,(-1));
          ///< nflop =84


        /** generalPurpose buffer  (GP)  [0-2] is now     = -R⁻¹*(Wbody1 * uk_body1 + Wbody2 * uk_body2 + b ) **/


        for(int i=0; i<3; i++) {

             /** generalPurpose buffer  (GP)  [3-5] is now = lambda old **/
            generalPurpose[i+3]=Elem_ColM(contBuffer,thid,C::lambda_s+i);
        }
            /** generalPurpose buffer  (GP)  [0-2] is now proxable **/
            MatVecUtil::vecAdd<PREC,3>(generalPurpose,generalPurpose+3,generalPurpose);
               ///< nflop =87




            MatVecUtil::proxTriplet1(generalPurpose,generalPurpose,mu);

            if(genOutput) {
                for(int i=0; i<3; i++) {
                    Elem_ColM(outputBuffer,thid,12+i)=generalPurpose[i];
                }
            } else{
                for(int i=0; i<3; i++) {
                    Elem_ColM(contBuffer,thid,C::lambda_s+i)=generalPurpose[i];
                }
            }
               ///< nflop =98

           /// MatVecUtil::proxTriplet1(generalPurpose,generalPurpose,mu);  /// DEBUG

            /** generalPurpose buffer  (GP)  [0-2] is now lambda new **/

            /// calc Lambdanew-Lambdaold save in generalPurpose buffer  (GP)  [0-2]




            ///
            ///
            ///
            if(convInVel==false) {

                /// convergence check in lmabda
                /// generalpurpose [0-2] lambda new
                /// generalpurpose [3-5] lambda old

                bool test = true;
                for(int i=0; i<3; i++) {
                    if(abs(generalPurpose[i]-generalPurpose[i+3]) < relTol * (abs(generalPurpose[i])) + absTol == false) {
                        test=false;
                    }
                }
                if (test) {
                } else {
                    Elem_ColMuInt(globalBuffer,0,G::conv_s)=1;
                }
            }
            ///
            ///
            ///




            MatVecUtil::vecMultScalar<PREC,3>(generalPurpose+3,(-1));
            MatVecUtil::vecAdd<PREC,3>(generalPurpose,generalPurpose+3,generalPurpose);
            ///< nflop =104

            ///  (matWbody1 * deltaLambda)
             /** generalPurpose buffer  (GP)  [3-8] is now  * (matWbody1 * deltaLambda) **/

            MatVecUtil::calcWTimesVec<PREC,true>(wBase,wR1,generalPurpose,generalPurpose+3);
            ///< nflop =134


                  for(int i=0; i<3; i++) {
            ///< W2 is used as a buffer for M_inv now
            wR1[i]=Elem_ColM(bodyBuffer,bodyIdx1,B::mInv_s);
            wR1[i+3]=Elem_ColM(bodyBuffer,bodyIdx1,B::thetaInv_s+i);

            }

        /** SUMMARY

        generalPurpose buffer  (GP)  [0-2] is delta lambda
        generalPurpose buffer  (GP)  [3-8] is   (matWbody1 * deltaLambda)
        wR1[0-5]                       is   M1⁻¹

        **/


        /**

        Calculates  M1⁻¹ * (matWbody1 * deltaLambda)
        saves in generalPurpose buffer  (GP)  [3-8]

        **/


        MatVecUtil::vecMultElWise<PREC,6>(generalPurpose+3,wR1,generalPurpose+3);
        ///< nflop =140


        /**  Save result for deltau1 new in output **/

        for(int z=0; z<6; z++) {
            if(genOutput) {
                Elem_ColM(outputBuffer,thid,z)=(generalPurpose+3)[z];
            } else {
                redBuffer[redIndex1+z*totalRedNumber]=(generalPurpose+3)[z];
            }
        }


        /** Do the same calculationbs for body2 to get the delta body 2

         generalPurpose buffer  (GP)  [0-2] is still delta lambda

          **/
            if(isBody2Simulated) {
                ///< check if the second body is simulated if not we know rthat minv2 is 0 and u2 is 0 => Wb2 and Wr2 and not needed anymore and there will be no deltau2 !

                MatVecUtil::calcWTimesVec<PREC,false>(wBase,wR2,generalPurpose,generalPurpose+3);
                ///< nflop =170


                for(int i=0; i<3; i++) {
                    ///< W2 is used as a buffer for M_inv now
                    wR2[i]=Elem_ColM(bodyBuffer,bodyIdx2,B::mInv_s);
                    wR2[i+3]=Elem_ColM(bodyBuffer,bodyIdx2,B::thetaInv_s+i);

                }

                MatVecUtil::vecMultElWise<PREC,6>(generalPurpose+3,wR2,generalPurpose+3);   ///< Minv2 * W2 * delta Lambda
                ///< nflop =176



                for(int z=0; z<6; z++) {
                    if(genOutput) {
                        Elem_ColM(outputBuffer,thid,z+6)=(generalPurpose+3)[z];
                    } else {
                        redBuffer[redIndex2+z*totalRedNumber]=(generalPurpose+3)[z];
                    } /// < has to be this way so it is easier to split everything up for the reduction kernel
                    ///< leaves [v1b1, v1b2,v1b3 .... v2b1 v2b2 v2b3 v2b4 .... , ... .. , w3b5,w3b6 ... ]
                }

            }


        thid+=gridDim.x*blockDim.x;
        /// ~=176 flop

    }
}


template<bool genOutput,bool convInVel,typename TCudaMatrix,typename TCudaIntMatrix>
void contIterKernelWrap(TCudaMatrix bodyBuffer,
                        TCudaMatrix contactBuffer,
                        TCudaIntMatrix globalBuffer,
                        typename TCudaMatrix::PREC * reductionBuffer,
                        TCudaIntMatrix indexSetBuffer,
                        TCudaMatrix outputBuffer,
                        unsigned int numberOfContacts,
                        VariantLaunchSettings variantSettings,
                        unsigned int totalRedNumber,
                        typename TCudaMatrix::PREC relTol,
                        typename TCudaMatrix::PREC absTol) {


    typedef typename TCudaMatrix::PREC PREC;


    contIterKernel<genOutput,convInVel> <<<variantSettings.numberOfBlocks,variantSettings.numberOfThreads>>>( outputBuffer,
                                                                                                    bodyBuffer,
                                                                                                    contactBuffer,
                                                                                                    globalBuffer,
                                                                                                    reductionBuffer,
                                                                                                    indexSetBuffer,
                                                                                                    numberOfContacts,
                                                                                                    totalRedNumber,
                                                                                                    relTol,
                                                                                                    absTol);

    CHECK_CUDA_LAST
}









}


#endif




