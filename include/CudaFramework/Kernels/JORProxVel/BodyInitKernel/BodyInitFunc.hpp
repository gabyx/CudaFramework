#ifndef BodyInitFunc_hpp
#define BodyInitFunc_hpp

#include "GeneralStructs.hpp"
#include "AssertionDebug.hpp"

namespace BodyInit{

template<typename bodyBufferListT,typename EigenMatrix,typename EigenIntMatrix>
void loadBodyInitCPU(bodyBufferListT &bodyDataList,
                     EigenMatrix& bodyBufferCPU,
                     EigenIntMatrix& globalBufferCPU
                     ){
    DEFINE_JORPROXVEL_GPUBUFFER_OFFSET_NAMESPACES


    unsigned int uIdx;

     uIdx=B::u1_s;


  unsigned int i = 0;
    for(auto & d : bodyDataList) {
        d.u      = bodyBufferCPU.template block<1,6>(i,uIdx);
        d.h_f    = bodyBufferCPU.template block<1,3>(i,B::f_s);
        d.h_m    = bodyBufferCPU.template block<1,3>(i,B::tq_s);
        d.deltaT = 0.1;
        d.mInv.Zero();
        for(int j=0;j<3;j++){

        d.mInv(j,j)  = bodyBufferCPU(i,B::mInv_s);
        d.mInv(j+3,j+3)  = bodyBufferCPU(i,B::thetaInv_s+j);
        }
        i++;
    }
}


template<typename bodyBufferListT>
void calcBodyInitCPU(bodyBufferListT& bodyDataList) {

    for(auto & d : bodyDataList) {
        d.u.template block<3,1>(0,0) +=  (d.deltaT*d.mInv.template block<3,3>(0,0))*d.h_f;
        d.u.template block<3,1>(3,0) +=  d.deltaT * d.mInv.template block<3,3>(3,3) * d.h_m;
    }
}



}

#endif




