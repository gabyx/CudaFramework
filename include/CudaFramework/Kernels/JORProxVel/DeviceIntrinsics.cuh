// ========================================================================================
//  CudaFramework
//  Copyright (C) 2014 by Gabriel Nützi <nuetzig (at) imes (d0t) mavt (d0t) ethz (d0t) ch>
//
//  This Source Code Form is subject to the terms of the GNU GPL 3.0 licence. 
//  If a copy of the GNU GPL 3.0 was not distributed with this
//  file, you can obtain one at http://opensource.org/licenses/GPL-3.0.
// ========================================================================================

#ifndef CudaFramework_Kernels_JORProxVel_DeviceIntrinsics_cuh
#define CudaFramework_Kernels_JORProxVel_DeviceIntrinsics_cuh

#include "CudaFramework/Kernels/JORProxVel/UtilitiesMatrixVector.cuh"

struct Default{};
    template<bool OnlyIfVariantIsSame, typename _M, typename _D> struct ManualOrDefault;
    template<typename _M,typename _D> struct ManualOrDefault<true, _M,_D>{ typedef  _M TValue; }; // Take Manual Value
    template<typename _M,typename _D> struct ManualOrDefault<false, _M,_D>{ typedef  _D TValue; }; // Take Default Value
    template<typename _D> struct ManualOrDefault<true,  Default ,_D>{  typedef  _D TValue; };
    template<typename _D> struct ManualOrDefault<false, Default ,_D>{  typedef  _D TValue; };
    template<int N,int M> struct isEqual{ static const bool result=false;};
    template<int M>struct isEqual<M,M>{   static const bool result=true;};


    template <typename T1, typename T2>
    struct IsSame
    {
      static const bool result = false;
    };


    template <typename T>
    struct IsSame<T,T>
    {
      static const bool result = true;
};




template <typename PREC>
PREC __device__ divDev(PREC x,PREC y) {


    if(IsSame<PREC,double>::result) {
        return __ddiv_rn (x,y);

    } else {
        return __fdiv_rn (x,y);
    }


}


template <typename PREC>
PREC __device__ rsqrtDev(PREC x) {

    if(IsSame<PREC,double>::result) {
        return rsqrt (x);

    } else {
        return rsqrtf (x);
    }


}


template <typename PREC>
PREC __device__ sqrtDev(PREC x) {

    if(IsSame<PREC,double>::result) {
        return sqrt(x);

    } else {
        return sqrtf(x);
    }


}
#endif



