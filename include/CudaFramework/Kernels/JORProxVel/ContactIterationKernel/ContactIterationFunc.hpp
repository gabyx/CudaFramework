// ========================================================================================
//  CudaFramework
//  Copyright (C) 2014 by Gabriel Nützi <nuetzig (at) imes (d0t) mavt (d0t) ethz (d0t) ch>
//
//  This Source Code Form is subject to the terms of the GNU GPL 3.0 licence. 
//  If a copy of the GNU GPL 3.0 was not distributed with this
//  file, you can obtain one at http://opensource.org/licenses/GPL-3.0.
// ========================================================================================

#ifndef CudaFramework_Kernels_JORProxVel_ContactIterationKernel_ContactIterationFunc_hpp
#define CudaFramework_Kernels_JORProxVel_ContactIterationKernel_ContactIterationFunc_hpp

#include "CudaFramework/Kernels/JORProxVel/GeneralStructs.hpp"

namespace ContIter{

template<typename Vector3,typename PREC>
void proxtriplet(Vector3& input,PREC mu,PREC* output) {
    output[0]=input(0);

    if(input(0)<0) {
        output[0]=0;
    }


    PREC radius;
    PREC absValue;

    radius=mu* output[0];
    absValue=input(1)*input(1)+input(2)*input(2);

    output[1] =input(1);
    output[2] =input(2);

    if(absValue>(radius*radius) ) {
        absValue=radius/sqrt(absValue);
        output[1]=input(1)*absValue;
        output[2] =input(2)*absValue;

    }

}

template<typename Vector3,typename PREC>
void proxDEBUG(Vector3& input,PREC mu,PREC* output) {

    output[0]=input(0);
    output[1]=input(1);
    output[2]=input(2);
}

template<typename PREC ,typename ContactDataListType,typename BodyDataType>
void contactiterationCPU(ContactDataListType& contactDataList,BodyDataType& bodyDataList)
{

typedef Eigen::Matrix<PREC,3,3> Matrix3x3;
typedef Eigen::Matrix<PREC,3,1> Vector3;
PREC buffer3b[3];

Vector3 buffer3;
Vector3 lambdaNew;
Vector3 output;
Matrix3x3 buffer3x3;

     for(auto & d : contactDataList) {


        buffer3=(d.lambdaOld-d.invR*(d.matWbody1.transpose()*bodyDataList[d.bodyIdx1].u+d.matWbody2.transpose()*bodyDataList[d.bodyIdx2].u+d.b));

        proxtriplet(buffer3,d.mu,buffer3b);

        output=buffer3;

        lambdaNew(0)=buffer3b[0];
        lambdaNew(1)=buffer3b[1];
        lambdaNew(2)=buffer3b[2];

        buffer3=(lambdaNew-d.lambdaOld);
        d.delta_uBody1=bodyDataList[d.bodyIdx1].mInv*d.matWbody1*buffer3;
        d.delta_uBody2=bodyDataList[d.bodyIdx2].mInv*d.matWbody2*buffer3;
        d.lambdaOld=lambdaNew;


    }
}
}



#endif // TestSimple2_hpp

