// ========================================================================================
//  CudaFramework
//  Copyright (C) 2014 by Gabriel Nützi <nuetzig (at) imes (d0t) mavt (d0t) ethz (d0t) ch>
//
//  This Source Code Form is subject to the terms of the GNU GPL 3.0 licence. 
//  If a copy of the GNU GPL 3.0 was not distributed with this
//  file, you can obtain one at http://opensource.org/licenses/GPL-3.0.
// ========================================================================================

#ifndef CudaFramework_Kernels_JORProxVel_ConvergenceCheckKernel_ConvergenceCheckFunc_hpp
#define CudaFramework_Kernels_JORProxVel_ConvergenceCheckKernel_ConvergenceCheckFunc_hpp
namespace ConvCheck{


template<typename PREC,typename BodyListType>
void calcConvCheck(BodyListType& bodyDataList){
    PREC value1;
    PREC value2;

    PREC relTol=0.1;
    PREC absTol=1;

    for(auto & d : bodyDataList) {

        value1 =  0.5*d.u.transpose()*d.regM*d.u;

        value2 =  0.5*d.u_2.transpose()*d.regM*d.u_2;

        d.test= (std::abs(value1-value2)<(relTol*value1+absTol));

    }
}
}

#endif



