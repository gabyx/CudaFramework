// ========================================================================================
//  CudaFramework
//  Copyright (C) 2014 by Gabriel Nützi <nuetzig (at) imes (d0t) mavt (d0t) ethz (d0t) ch>
//
//  This Source Code Form is subject to the terms of the GNU GPL 3.0 licence.
//  If a copy of the GNU GPL 3.0 was not distributed with this
//  file, you can obtain one at http://opensource.org/licenses/GPL-3.0.
// ========================================================================================

#ifndef CudaFramework_General_TypeTraitsHelper_hpp
#define CudaFramework_General_TypeTraitsHelper_hpp

/**
* @brief Simply dummy class to check if we use the internal standart kernel settings for the variant
* @{
*/
namespace TypeTraitsHelper{

    struct Default{};
    template<bool OnlyIfVariantIsSame, typename _M, typename _D> struct ManualOrDefault;
    template<typename _M,typename _D> struct ManualOrDefault<true, _M,_D>{ typedef  _M TValue; }; // Take Manual Value
    template<typename _M,typename _D> struct ManualOrDefault<false, _M,_D>{ typedef  _D TValue; }; // Take Default Value
    template<typename _D> struct ManualOrDefault<true,  Default ,_D>{  typedef  _D TValue; };
    template<typename _D> struct ManualOrDefault<false, Default ,_D>{  typedef  _D TValue; };
    template<int N,int M> struct IsEqual{ static const bool result=false;};
    template<int M>struct IsEqual<M,M>{   static const bool result=true;};



    //struct FalseType { static const bool  value = false ; };
    //struct TrueType {  static const bool  value = true ; };


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


};
/** @} */

#endif
