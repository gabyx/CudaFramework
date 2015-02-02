#ifndef ContactInit_cuh
#define ContactInit_cuh



#include <cuda_runtime.h>

#include "StaticAssert.hpp"

#include "CudaError.hpp"
#include "CudaMatrix.hpp"

#include "UtilitiesMatrixVector.cuh"
#include "GPUBufferOffsets.hpp"
#include "DeviceIntrinsics.cuh"
#include "VariantLaunchSettings.hpp"


#define Elem_RowMuInt(_A,_row,_col)                             (     *( (unsigned int*)((char*)(_A.m_pDevice) + (_row) * (_A.m_outerStrideBytes)) + (_col) )     )
#define Elem_ColMuInt(_A,_row,_col)                                Elem_RowMuInt(_A,_col,_row)

namespace ContactInit{
/*** ALL MATRICES ARE SAVED IN COLUMN MAJOR FORM !   ***/


template<typename PREC>
__device__ void getTildeMat(PREC* vecIn,PREC * matOut){

    /// calculates the tilde matrix from a vector
    /// matrix in column major
        PREC buffer3[3];  ///< no alias


        buffer3[0]=vecIn[0];
        buffer3[1]=vecIn[1];
        buffer3[2]=vecIn[2];

        matOut[0] = 0.0;
        matOut[1] = buffer3[2];
        matOut[2] = -buffer3[1];
        matOut[3] = -buffer3[2];
        matOut[4] = 0.0;
        matOut[5] = buffer3[0];
        matOut[6] = buffer3[1];
        matOut[7] = -buffer3[0];
        matOut[8] = 0.0;

}




template<typename PREC>
__device__ void getMatfromQuat(PREC quat[4], PREC matOut[9])
{

        /// calculates a transformation matrix from a quaternion rotation
        /// matrix in column major


        matOut[0]=(1-2*(quat[3]*quat[3]+quat[2]*quat[2]));
        matOut[3]=2*(quat[0]*quat[3]+quat[2]*quat[1]);
        matOut[6]=2*(-quat[0]*quat[2]+quat[1]*quat[3]);
        matOut[1]=2*(-quat[0]*quat[3]+quat[2]*quat[1]);
        matOut[4]=(1-2*(quat[3]*quat[3]+quat[1]*quat[1]));
        matOut[7]=2*(quat[0]*quat[1]+quat[2]*quat[3]);
        matOut[2]=2*(quat[0]*quat[2]+quat[1]*quat[3]);

        matOut[5]=2*(-quat[0]*quat[1]+quat[2]*quat[3]);
        matOut[8]=(1-2*(quat[1]*quat[1]+quat[2]*quat[2]));

        //No check if quaternion is unit...(performance)

        // My fast cpp version
        /// alternative algorithms
//       PREC fTx  = 2.0*quat(1);
//       PREC fTy  = 2.0*quat(2);
//       PREC fTz  = 2.0*quat(3);
//       PREC fTwx = fTx*quat(0);
//       PREC fTwy = fTy*quat(0);
//       PREC fTwz = fTz*quat(0);
//       PREC fTxx = fTx*quat(1);
//       PREC fTxy = fTy*quat(1);
//       PREC fTxz = fTz*quat(1);
//       PREC fTyy = fTy*quat(2);
//       PREC fTyz = fTz*quat(2);
//       PREC fTzz = fTz*quat(3);
//
//	A(0,0) = 1.0f-(fTyy+fTzz);
//	A(0,1) = fTxy-fTwz;
//	A(0,2) = fTxz+fTwy;
//	A(1,0) = fTxy+fTwz;
//	A(1,1) = 1.0f-(fTxx+fTzz);
//	A(1,2) = fTyz-fTwx;
//	A(2,0) = fTxz-fTwy;
//	A(2,1) = fTyz+fTwx;
//	A(2,2) = 1.0f-(fTxx+fTyy);

}

template<typename PREC>
__device__ void getBasefromNormalVec(PREC* n,PREC* basis)
{

        /// calculates one possible base from a normal vector

        basis[0]=-n[0];
        basis[1]=-n[1];
        basis[2]=-n[2];

        ///< nflop = 3

        bool decide= abs(basis[0]) > abs(basis[2]);

        ///maybe use an inline function

        /// squared values of normal vector

        PREC sq_1=(basis[0]*basis[0]);
        PREC sq_2=(basis[1]*basis[1]);
        PREC sq_3=(basis[2]*basis[2]);

        ///< nflop = 6

        /// buffer is now length of normal vector

        PREC buffer = sqrtDev(sq_1+sq_2+sq_3);

        ///< nflop = 9


        /// normalise normal vector

        basis[0]=divDev(basis[0],buffer);
        basis[1]=divDev(basis[1],buffer);
        basis[2]=divDev(basis[2],buffer);

        ///< nflop = 12

        // /// squared values of normal vector

        sq_1=(basis[0]*basis[0]);
        sq_2=(basis[1]*basis[1]);
        sq_3=(basis[2]*basis[2]);


        ///< nflop = 15

        ///  get t1

        if(decide) {
            /// buffer is now length of t1

            buffer= rsqrtDev(sq_1+sq_3);

            basis[3]=-basis[2]*buffer;
            basis[4]=0;
            basis[5]=basis[0]*buffer;

        } else {

            /// buffer is now length of t1

            buffer=rsqrtDev(sq_2 + sq_3);


            basis[3]= 0;
            basis[4]= basis[2]*buffer;
            basis[5]=-basis[1]*buffer;
        }
        ///< nflop = 19

        ///do cross product n x t1:

        basis[6]=(basis[1]*basis[5]) - (basis[2]*basis[4]);
        basis[7]=(basis[2]*basis[3]) - (basis[0]*basis[5]);
        basis[8]=(basis[0]*basis[4]) - (basis[1]*basis[3]);

        ///< nflop = 28

}



template<bool genoutput, typename TCudaMatrix,typename TCudaIntMatrix>
__global__ void contactInitKernel(
        TCudaMatrix outputBuffer,
        TCudaMatrix bodyBuffer,
        TCudaMatrix contactBuffer,
        TCudaIntMatrix indexBuffer,
        TCudaIntMatrix globalBuffer,
        unsigned int totalContactNumber) {

        STATIC_ASSERT( utilCuda::CudaMatrixFlags::template isRowMajor<TCudaMatrix::Flags>::value == false )
        STATIC_ASSERT( utilCuda::CudaMatrixFlags::template isRowMajor<TCudaIntMatrix::Flags>::value == false )
        typedef typename TCudaMatrix::PREC PREC;

        DEFINE_JORPROXVEL_GPUBUFFER_OFFSET_NAMESPACES

        unsigned long int thid=threadIdx.x+blockDim.x*blockIdx.x;

        PREC contactFrame[9];

        PREC wR[9];

        PREC chiBuffer[3];

        PREC generalPurpose[13];

        int bodyIdx;  /// < body number 1 (Index for body 1)

        int u_sIdx;

        bool isBody2Simulated;

        u_sIdx=B::u1_s;

        while (thid<totalContactNumber) {

            /** get body index for body 1 **/


            /**
            !!!!!!!!!!! BODY 1 !!!!!!!!!!!!!
            **/
            bodyIdx = Elem_ColMuInt(indexBuffer,thid,I::b1Idx_s);


            if(bodyIdx==Elem_ColMuInt(indexBuffer,thid,I::b2Idx_s)) {
                isBody2Simulated = false;
            } else{
                isBody2Simulated = true;
            }


            for (int i=0; i<B::q_l; i++) {

                /** get quaternion 1 **/
                /** generalPurpose [9-12] = quat**/
                generalPurpose[i+9]=Elem_ColM(bodyBuffer,bodyIdx,B::q_s+i);
            }


            for(int i=0; i<3; i++) {

                /** load n vector into contact frame  **/
                contactFrame[i]=Elem_ColM(contactBuffer,thid,C::n_s+i);

                /** generalPurpose [0-2] = vec_iRsc**/
                generalPurpose[i]=Elem_ColM( contactBuffer,thid, C::rSC1_s+i);;

            }


            getBasefromNormalVec<PREC>(contactFrame,contactFrame);  ///< calculates Wbase from the normal vector and stores it in  contactFrame [0-8]
            ///< nflop = 28

            /** generalPurpose [0-2] = vec_iRsc**/
            getTildeMat<PREC>(generalPurpose,generalPurpose);  ///< calculates mat_iRscTilde
            /** generalPurpose [0-8] = mat_iRscTilde**/



            getMatfromQuat<PREC>(generalPurpose+9,wR);   ///< calculates a transformation matrix from a quaternion rotation

            MatVecUtil::matProd3x3<PREC>(wR,generalPurpose); ///< A*mat_iRscTilde = wR[0-9]

            ///< nflop = 28+36 = 64

            /** generalPurpose [0-12] = free **/

            MatVecUtil::matProd3x3<PREC>(wR,contactFrame); ///< A*mat_iRscTilde*wBase = wR[0-9]

            ///< nflop = 64+36 = 100


            for (int i=0; i<3; i++) {
                /** generalPurpose [3-8] = Minv**/

                generalPurpose[i+3]=Elem_ColM(bodyBuffer,bodyIdx,B::mInv_s);
                generalPurpose[i+6]=Elem_ColM(bodyBuffer,bodyIdx,B::thetaInv_s+i);

                /** generalPurpose [0-2] = Rinv**/
                generalPurpose[i]=0;

            }
            MatVecUtil::calcRinvPartly<PREC>(generalPurpose,contactFrame,wR,generalPurpose+3) ;   ///< Wt1*Minv1*W1 = generalPurpose[0-2] /// aber nur die diag werte werden gerechnet ..
             ///< nflop =100+55 = 155
            /// /** generalPurpose [3-5] = free at the moment and used as buffer of length 3**/



            for(int i=0; i<6; i++) {

                /** generalPurpose [6-11] = u body 1**/
                generalPurpose[i+6]=Elem_ColM(bodyBuffer,bodyIdx,u_sIdx+i);
            }

            MatVecUtil::calcWtransU<PREC,true>(contactFrame,wR,generalPurpose+6,generalPurpose+3);  ///< Wt1*u1 = generalPurpose[3-5]
             ///< nflop = 155+36 = 191


            /** generalPurpose [6-8] = eps**/
            /** generalPurpose [9-11] = second buffer length 3**/

            for(int i =0; i<3; i++) {
                chiBuffer[i]=Elem_ColM(contactBuffer,thid,C::chi_s+i);
                generalPurpose[i+6]=Elem_ColM(contactBuffer,thid,C::eps_s+i);
            }

            MatVecUtil::matProdDiag<PREC,3>(generalPurpose+6,generalPurpose+3,generalPurpose+9); ///<eps*wt1*u1

            ///< nflop =  191+1=192

           /// MatVecUtil::matProdDiag<PREC,3>(generalPurpose+6,chiBuffer,generalPurpose+3); ///<chi*eps

            for(int i=0; i<3; i++) {
                (generalPurpose+3)[i]=generalPurpose[i+6]*chiBuffer[i]; ///<chi*eps
            }
            /// matProdColMaj<PREC,3,3,1>(eps,chi,buffer3);
            MatVecUtil::vecAdd<PREC,3>(generalPurpose+3,chiBuffer,chiBuffer); ///< chi+chi*eps
            MatVecUtil::vecAdd<PREC,3>(chiBuffer,generalPurpose+9,chiBuffer); ///< chi+chi*eps+eps*Wt1*u1
            ///< nflop =  192+6=198




            /// < now for body 2

            if(isBody2Simulated==true) {

                bodyIdx = Elem_ColMuInt(indexBuffer,thid,I::b2Idx_s);

                /** generalPurpose [3-11] = free **/


                /** generalPurpose [3-6] = quatbdy2 **/
                for (int i=0; i<4; i++) {

                    generalPurpose[i+3]=Elem_ColM(bodyBuffer,bodyIdx,B::q_s+i);
                }

                /** generalPurpose [6-8] = veciRsc2 **/
                for(int i=0; i<3; i++) {

                    generalPurpose[i+7]=Elem_ColM( contactBuffer,thid, C::rSC2_s+i );
                }

                /// fill the output buffer
            }
            if(genoutput) {
                for(int i=0; i<9; i++) {
                    /// matrices in column major
                    Elem_ColM(outputBuffer,thid,i)   = contactFrame[i];
                    Elem_ColM(outputBuffer,thid,9+i) = wR[i];
                }

            } else {
                for(int i=0; i<9; i++) {

                    /// matrices in column major
                    Elem_ColM(contactBuffer,thid,C::w1_s+i)  = contactFrame[i]; /// DEBUG
                    Elem_ColM(contactBuffer,thid,C::w1r_s+i) = wR[i];

                }
            }
            if(isBody2Simulated==true){
            getMatfromQuat<PREC>(generalPurpose+3,wR);

            /** generalPurpose [3-11] = mat_iRscTilde **/
            getTildeMat<PREC>(generalPurpose+7,generalPurpose+3);
            MatVecUtil::matProd3x3<PREC>(wR,generalPurpose+3);

            /** generalPurpose [3-11] = free **/
            MatVecUtil::matProd3x3<PREC>(wR,contactFrame);
            ///< nflop =  198+2*36=270

            for(int i=0; i<9 ; i++) {
                (wR)[i]=(-1)*(wR)[i];
            }
            ///< nflop = 270+9=279



            for (int i=0; i<3; i++) {
                /** generalPurpose [3-8] = Minv2 **/
                generalPurpose[i+3]=Elem_ColM(bodyBuffer,bodyIdx,B::mInv_s);
                generalPurpose[i+6]=Elem_ColM(bodyBuffer,bodyIdx,B::thetaInv_s+i);


            }


            MatVecUtil::calcRinvPartly<PREC>(generalPurpose,contactFrame,wR,generalPurpose+3) ; ///< Rinv = Wt1*Minv1*W1+Wt2*Minv2*W2  /// aber nur die diag werte werden gerechnet ..
            ///< nflop = 279+55= 334


            for(int i=0; i<6; i++) {

                /** generalPurpose [3-8] = uBody2 **/
                generalPurpose[i+3]=Elem_ColM(bodyBuffer,bodyIdx,u_sIdx+i);
            }

            MatVecUtil::calcWtransU<PREC,false>(contactFrame,wR,generalPurpose+3,generalPurpose+9);

             ///< nflop = 334 +36=370


            /** generalPurpose [9-11] = buffer3 **/
            /** generalPurpose [9-11] = Wt2*u2 **/

            for(int i=0; i<3; i++) {
                /** generalPurpose [3-5] = eps **/
                generalPurpose[i+3]=Elem_ColM(contactBuffer,thid,C::eps_s+i);
            }


            MatVecUtil::matProdDiag<PREC,3>(generalPurpose+3,generalPurpose+9,generalPurpose+3);
               ///< nflop = 370+3=373

            /** generalPurpose [3-5] = buffer3b **/
            /** generalPurpose [3-5] = eps*wt2*u2 **/

            MatVecUtil::vecAdd<PREC,3>(chiBuffer,generalPurpose+3,chiBuffer);
             ///< nflop = 376

            /** contactframe 0 is now alpha **/
            }else{
                for (int i=0; i<9; i++) {
                    wR[i]=0;
                }
            }

            /**   Load everything back **/

            MatVecUtil::calcRinvFinish<PREC>(generalPurpose);
               ///< nflop = 377
            contactFrame[0]=Elem_ColM(contactBuffer,thid,C::alpha_s);
            for(int i=0;i<3;i++){
                        generalPurpose[i] =  contactFrame[0]*generalPurpose[i];  /// DEBUG

            }


            if(genoutput) {
                for(int i=0; i<9; i++) {
                    Elem_ColM(outputBuffer,thid,18+i)=wR[i];
                }

                for(int i=0; i<3; i++) {
                    Elem_ColM(outputBuffer,thid,27+i) =chiBuffer[i];
                    Elem_ColM(outputBuffer,thid,30+i) =generalPurpose[i];
                }

            } else {
                for(int i=0; i<9; i++) {

                    Elem_ColM(contactBuffer,thid,C::w2r_s+i) = wR[i];

                }

                for(int i=0; i<3; i++) {
                    Elem_ColM(contactBuffer,thid,C::b_s+i) = chiBuffer[i];
                    Elem_ColM(contactBuffer,thid,C::r_s+i) = generalPurpose[i];
                }
            }

            thid+=gridDim.x*blockDim.x;
            /// ~377 flop
        }
}


template<bool genoutput,typename TCudaMatrix,typename TCudaIntMatrix>
void contactInitKernelWrap(TCudaMatrix bodyBuffer,
                            TCudaMatrix contactBuffer,
                            TCudaIntMatrix indexBuffer,
                            TCudaIntMatrix globalBuffer,
                            TCudaMatrix outputBuffer,
                            unsigned int numberOfContacts,
                            VariantLaunchSettings variantSettings
                            ) {

    contactInitKernel<genoutput> <<<variantSettings.numberOfBlocks,variantSettings.numberOfThreads>>>(outputBuffer,
                                                    bodyBuffer,
                                                    contactBuffer,
                                                    indexBuffer,
                                                    globalBuffer,
                                                    numberOfContacts);

    CHECK_CUDA_LAST;

    return;

}




}


#endif

