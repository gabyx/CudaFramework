#ifndef CudaFramework_Kernels_JORProxVel_UtilitiesMatrixVector_cuh
#define CudaFramework_Kernels_JORProxVel_UtilitiesMatrixVector_cuh


#include <cuda_runtime.h>
#include "CudaFramework/CudaModern/CudaError.hpp"
#include "CudaFramework/CudaModern/CudaMatrix.hpp"
#include "CudaFramework/Kernels/JORProxVel/DeviceIntrinsics.cuh"


namespace MatVecUtil {


template<typename PREC,unsigned int nRows>
__device__ void vecAdd(PREC* v1,PREC* v2,PREC* output) {

///< nflop = nRows

    for(unsigned int i=0; i< nRows; i++) {
        output[i]=v1[i]+v2[i];
    }
}

template<typename PREC,bool isW1>
__device__ void calcWtransU(PREC* w, PREC * wR, PREC* u,PREC* output) {


///< nflop = 3*3*4 = 36
PREC buffer[3];

    for (int i=0; i<3; i++) {
        buffer[i]=0;


    }
    if(isW1) {
        for (int j=0; j<3; j++) {
            for (int i=0; i<3; i++) {
                buffer[j]+=wR[i+j*3]*u[i+3]+w[i+j*3]*u[i];
            }
        }
    } else {

        for (int j=0; j<3; j++) {
            for (int i=0; i<3; i++) {
                buffer[j]+=wR[i+j*3]*u[i+3]-w[i+j*3]*u[i];
            }
        }
    }
       for (int i=0; i<3; i++) {
        output[i]=buffer[i];
    }
}



template<typename PREC,bool isW1>
__device__ void calcWTimesVec(PREC* w, PREC * wR, PREC* u,PREC* output) {

///< nflop = 3*10 = 30
  PREC buffer[6];


    if(isW1) {
        for (int j=0; j<3; j++) {
            buffer[j]=w[j]*u[0]+w[j+3]*u[1]+w[j+6]*u[2];
            buffer[j+3]=wR[j]*u[0]+wR[j+3]*u[1]+wR[j+6]*u[2];
        }
    } else {

        for (int j=0; j<3; j++) {
            buffer[j]=-w[j]*u[0]-w[j+3]*u[1]-w[j+6]*u[2];
            buffer[j+3]=wR[j]*u[0]+wR[j+3]*u[1]+wR[j+6]*u[2];
        }

    }
    for (int j=0; j<6; j++) {
output[j]=buffer[j];
    }


}



template<typename PREC, unsigned int nRows>
__device__ void vecMultElWise(PREC* v1, PREC* v2, PREC* vOut) {

    ///< nflop = nRows
    for(int i=0; i<nRows; i++) {
        vOut[i]=v1[i]*v2[i];
    }
}

template<typename PREC,unsigned int nRows>
__device__ void vecMultScalar(PREC* v1,PREC scalar) {

///< nflop = nRows

    for(unsigned int i=0; i< nRows; i++) {
        v1[i] *= scalar;
    }
}

template<typename PREC>
__device__ void matProd3x3(PREC* a,PREC* b) {


    ///< nflop = 3*3*4=36

    ///  matrix product for 3x3
    ///  everything in column major

    PREC buffer[3];
#pragma unroll
    for(int i=0; i<3; i++) {
#pragma unroll
        for (int j=0; j<3; j++) {

            buffer[j]= (a[i]*b[3*j]+ a[3+i]*b[1+3*j]+ a[6+i]*b[2+3*j]);

        }
#pragma unroll
        for (int j=0; j<3; j++) {

            a[3*j+i]=buffer[j];

        }

    }
}

template <typename PREC,unsigned int dim>
inline __device__ void matProdDiag(PREC* matDiag, PREC* matNonDiag,PREC* matOutput) {

///< nflop = dim

#pragma unroll
    for (int j=0; j<dim; j++ ) {
        matOutput[j]=matDiag[j]*matNonDiag[j];

    }

}


template <typename PREC>
__device__ void proxTripletV2(PREC* outBuffer, PREC* value,PREC* mu) {

    PREC buffer3[3];

///< nflop = 11

    buffer3[0]=-value[0];
    buffer3[1]=value[1];
    buffer3[2]=value[2];


    if(buffer3[0]<0) {
        outBuffer[0]=-buffer3[0];
    } else {
        outBuffer[0]= 0;
    }
    PREC radius= mu[0]*outBuffer[0];

    PREC absValue = buffer3[2]*buffer3[2]+buffer3[1]*buffer3[1];

    if(absValue>(radius*radius)) {
        if(IsSame<PREC,double>::result) {
            absValue=radius*rsqrt(absValue);
        } else {
            absValue=radius*rsqrtf(absValue);
        }
        outBuffer[1]=absValue*buffer3[1];
        outBuffer[2]=absValue*buffer3[2];
    } else {
        outBuffer[1]=buffer3[1];
        outBuffer[2]=buffer3[2];
    }

}

template <typename PREC>
__device__ void proxDEBUG(PREC* outBuffer, PREC* value, PREC* mu) {

///< nflop = 8
PREC buffer3[1];


    if(value[0]<0) {
     buffer3[0]=0;
    }else{
     buffer3[0]=value[0];
    }

outBuffer[0]=buffer3[0];

        outBuffer[1]=0;
        outBuffer[2]=0;
}

template <typename PREC>
__device__ void proxTriplet1(PREC* outBuffer, PREC* value, PREC* mu) {

///< nflop = 8

    outBuffer[0]=value[0];
    if(value[0]<0) {
        outBuffer[0]=0;
    }


    PREC radius;
    PREC absValue;

    radius=mu[0]*outBuffer[0];
    absValue=value[1]*value[1]+value[2]*value[2];

    outBuffer[1]=value[1];
    outBuffer[2]=value[2];


    if(absValue>(radius*radius) ) {
        absValue=radius*rsqrtDev(absValue);
        outBuffer[1]=value[1]*absValue;
        outBuffer[2]=value[2]*absValue;
    }


}


template <typename PREC>
__device__ void calcRinvPartly(PREC* outBuffer, PREC* wBase,PREC* wr,PREC* mInv1) {

///< nflop = 54

    for(int i=0; i<3; i++) {
            outBuffer[0]+=wBase[i]*mInv1[i]*wBase[i]+wr[i]*mInv1[i+3]*wr[i];
            outBuffer[1]+=wBase[i+3]*mInv1[i]*wBase[i+3]+wr[i+3]*mInv1[i+3]*wr[i+3];
            outBuffer[2]+=wBase[i+6]*mInv1[i]*wBase[i+6]+wr[i+6]*mInv1[i+3]*wr[i+6];
    }
}

template <typename PREC>
__device__ void calcRinvFinish(PREC* outBuffer) {

///< nflop = 1

    for(int z=0; z<3; z++) {
        outBuffer[z]=divDev<PREC>(1,outBuffer[z]);
    }
    if(abs(outBuffer[1])>abs(outBuffer[2])) {
        outBuffer[2]=outBuffer[1];
    } else {
        outBuffer[1]=outBuffer[2];
    }

}


}

/***   Behalte die Funktionen, habe sie getestet und sie funktionieren ! Wurde jedoch im code nicht gebraucht. Evt. später nützlich (Thierry Oct 2014)  **/
/* Hier sind funktionen die im moment ncihtmehr gebraucht werden, vll spaeter mal wieder nuetzlich


template<typename PREC,unsigned int nCols1, unsigned int nRows1, unsigned int nCols2>
__device__ void matProdColMaj(PREC* m1,PREC* m2,PREC* outputBuffer) {

    PREC* buffer=(PREC*)malloc(nCols1*sizeof(PREC));

    for(int m=0; m<nRows1; m++) {
        for (int j=0; j<nCols1 ; j++) {
            buffer[j]=m1[m+j*nRows1];

        }

        for (int n=0; n<nCols2; n++)   {
             output[m+n*nRows1]=0;
            for (int k=0; k<nCols1; k++) {

                output[m+n*nRows1]+=buffer[k]* m2[k+n*nCols1];
            }
        }
    }
    free(buffer);
}

template<typename PREC,unsigned int nRows>
__device__ void vecAddElWise(PREC* v1,PREC v2,PREC* output) {

   for(unsigned int i=0; i< nRows; i++)
   {
       output[i]=v1[i]+v2;
   }
}


template<typename PREC,unsigned int nCols1, unsigned int nRows1, unsigned int nCols2>
__device__ void colMajTransProdColMaj(PREC* m1,PREC* m2,PREC* output) {


    PREC* buffer2=(PREC*)malloc((nCols2*nRows1)*sizeof(PREC));

    for(int m=0; m<nRows1; m++) {

        for (int n=0; n<nCols2; n++)   {

        buffer2[m+n*nRows1]=0;

            for (int k=0; k<nCols1; k++) {
                 buffer2[m+n*nRows1]+=m1[m*nCols1+k]* m2[k+n*nCols1];
            }
        }
    }

        for(int m=0; m<nRows1; m++) {

        for (int n=0; n<nCols2; n++)   {
                 output[m+n*nRows1]=buffer2[m+n*nRows1];
        }
    }
    free(buffer2);
}


template<typename PREC>
__device__ void calcWtransU(PREC* w, PREC * wR, PREC* u,PREC* output){


///< nflop = 3*3*4 = 36

for (int i=0; i<3; i++){
    output[i]=0;


}
        for (int j=0; j<3; j++) {
            for (int i=0; i<3; i++) {
                output[j]+=wR[i+j*3]*u[i+3]+w[i+j*3]*u[i];
            }
        }

}


template <typename PREC>
__device__ void proxTripletV2(PREC* outBuffer, PREC* value) {
    if(value[0]<0) {
        //value[0]=0;
        outBuffer[0]=0;
    }
}


template <typename PREC>
__device__ void prox_DEBUG(PREC* outBuffer, PREC* value, PREC* mu) {


   outBuffer[0]=value[0];
   outBuffer[1]=value[1];
   outBuffer[2]=value[2];


}



template <typename PREC>
__device__ void calc_Wt_times_uk(PREC * W, PREC* uk,PREC* outBuffer) {

    for(int i=0; i<3; i++) {
        outBuffer[i]=W[0+6*i]*uk[0]+
                     W[1+6*i]*uk[1]+
                     W[2+6*i]*uk[2]+
                     W[3+6*i]*uk[3]+
                     W[4+6*i]*uk[4]+
                     W[5+6*i]*uk[5];
    }
}

template <typename PREC>
__device__ void calcRinv_1(PREC* outBuffer, PREC* wBase,PREC* w1r,PREC* w2r,PREC* mInv1,PREC* mInv2) {



    PREC w[18];


        for(int i=0; i<3; i++) {


            w[i+3]=w1r[i];   ///< matrices are all column major
            w[i+9]=w1r[i+3];
            w[i+15]=w1r[i+6];

            w[i]=wBase[i];
            w[i+6]=wBase[i+3];
            w[i+12]=wBase[i+6];
        }
    PREC buffer1[18];

    PREC buffer3[9];

    for(int j=0; j<3 ; j++) {
        for(int z=0; z<6 ; z++) {
            buffer1[z+6*j]=w[z+6*j]*mInv1[z];
        }
    }

    colMajTransProdColMaj<PREC,6,3,3>(buffer1, w,buffer3) ;

        for(int i=0; i<3; i++) {
            w[i+3]=w2r[i];
            w[i+9]=w2r[i+3];
            w[i+15]=w2r[i+6];

            w[i]=-wBase[i];
            w[i+6]=-wBase[i+3];
            w[i+12]=-wBase[i+6];
        }

    for(int j=0; j<3 ; j++) {
        for(int z=0; z<6 ; z++) {
            buffer1[z+6*j]=w[z+6*j]*mInv2[z];
        }
    }


    colMajTransProdColMaj<PREC,6,3,3>(buffer1,w ,buffer1) ;

    vecAdd<PREC,9>(buffer1,buffer3,buffer1);

    for(int z=0; z<3; z++) {
       outBuffer[z]= 1/buffer1[4*z];
    }

    if(outBuffer[1]>outBuffer[2]) {
        outBuffer[2]=outBuffer[1];
    } else {
        outBuffer[1]=outBuffer[2];
    }

}



template <typename PREC>
__device__ void calcRinv_1_secondtry(PREC* outBuffer, PREC* wBase,PREC* w1r,PREC* w2r,PREC* mInv1,PREC* mInv2) {


      PREC invR[3];



 for(int i=0; i<3; i++){
    for(int j=0; j<3; j++){
         invR[i]+=wBase[j+3*i]*mInv2[j]*wBase[j+3*i]+w2r[j+3*i]*mInv2[j+3]*w2r[j+3*i];
    }
 }

    for(int z=0; z<3; z++) {
       outBuffer[z]= 1/invR[z];
    }
    if(outBuffer[1]>outBuffer[2]) {
        outBuffer[2]=outBuffer[1];
    } else {
        outBuffer[1]=outBuffer[2];
    }

}

*/

#endif



