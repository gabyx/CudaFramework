// ========================================================================================
//  CudaFramework
//  Copyright (C) 2014 by Gabriel Nützi <nuetzig (at) imes (d0t) mavt (d0t) ethz (d0t) ch>
//
//  This Source Code Form is subject to the terms of the GNU GPL 3.0 licence.
//  If a copy of the GNU GPL 3.0 was not distributed with this
//  file, you can obtain one at http://opensource.org/licenses/GPL-3.0.
// ========================================================================================

#ifndef CudaFramework_General_FlopsCounting_hpp
#define CudaFramework_General_FlopsCounting_hpp

#define FLOPS_WITH_ASSIGNMENTS() (0)
#define FLOPS_DIVISION() (4)
#define FLOPS_SQRT() (1)
#define FLOPS_COMAPARE() (0)


// FLOPS Calculations for the specific kernels and functions

template<typename T>
double proxContactOrdered_RPlusAndDisk_1threads_kernel_FLOPS( T _nC) {
 return (double)_nC * (double)(/*Normal*/ 1 + /*Tangetial*/ 1 + 3 + 1 + 1*FLOPS_SQRT() + 2 + FLOPS_WITH_ASSIGNMENTS() * 7 );
}

template<typename T>
double evaluateProxTermJOR_FLOPS( T _MG, T _NG){
 double MG= _MG;
 double NG= _NG;
 return ( (2*MG *NG - MG) + MG   + (double)FLOPS_WITH_ASSIGNMENTS() * (MG + MG) )  ;
}


template<typename T>
double evaluateProxTermSOR_FLOPS(T _MG, T _NG){
 double MG= _MG;
 double NG= _NG;
 return ( (2*MG *NG - MG) + MG   + (double)FLOPS_WITH_ASSIGNMENTS() * (MG + MG) )  ;
}

#endif
