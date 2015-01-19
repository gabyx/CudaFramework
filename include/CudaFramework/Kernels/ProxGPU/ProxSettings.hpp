// ========================================================================================
//  CudaFramework
//  Copyright (C) 2014 by Gabriel Nützi <nuetzig (at) imes (d0t) mavt (d0t) ethz (d0t) ch>
//
//  This Source Code Form is subject to the terms of the GNU GPL 3.0 licence.
//  If a copy of the GNU GPL 3.0 was not distributed with this
//  file, you can obtain one at http://opensource.org/licenses/GPL-3.0.
// ========================================================================================

#ifndef CudaFramework_Kernels_ProxGPU_ProxSettings_hpp
#define CudaFramework_Kernels_ProxGPU_ProxSettings_hpp

#include "CudaFramework/General/StaticAssert.hpp"
#include "CudaFramework/General/TypeTraitsHelper.hpp"

/**
* ProxIterationType can be either SOR or JOR, or lets see if we can implement a MIX
*/
struct ProxIterationType{
   struct SOR{};
   struct JOR{};
};

template<typename _PREC, int _VariantId, typename _ConvexSet, int _alignMatrix, int _nMaxIterations, bool _bAbortIfConverged, int _nCheckConvergedFlag, bool _bMatchCPUToGPU, typename _KernelSettings>
struct SorProxGPUVariantSettingsWrapper{
   typedef  _PREC PREC;
   typedef  _ConvexSet  TConvexSet;
   typedef  _KernelSettings  TKernelSettings;
   static const int VariantId = _VariantId;
   STATIC_ASSERTM(VariantId >0, VARIANT_ID_NEEDS_TO_BE_BIGGER_THAN_ZERO)
   static const int alignMatrix = _alignMatrix;
   static const int nMaxIterations = _nMaxIterations;
   static const bool bAbortIfConverged = _bAbortIfConverged;
   static const int nCheckConvergedFlag = _nCheckConvergedFlag;
   static const bool bMatchCPUToGPU = _bMatchCPUToGPU;
};

#define DEFINE_SorProxGPUVariant_SettingsWrapper(_TSettings_) \
   typedef typename _TSettings_::PREC PREC; \
   typedef typename _TSettings_::TConvexSet  TConvexSet; \
   typedef typename _TSettings_::TKernelSettings TKernelSettings; \
   static const int VariantId = _TSettings_::VariantId; \
   static const int alignMatrix = _TSettings_::alignMatrix; \
   static const int nMaxIterations = _TSettings_::nMaxIterations; \
   static const bool bAbortIfConverged = _TSettings_::bAbortIfConverged; \
   static const int nCheckConvergedFlag = _TSettings_::nCheckConvergedFlag; \
   static const bool bMatchCPUToGPU = _TSettings_::bMatchCPUToGPU; \


template<typename TSorProxGPUVariantSettingsWrapper, typename TConvexSet> class SorProxGPUVariant; // Prototype

template<int _VariantId, typename _ConvexSet, bool _alignMatrix, typename _KernelSettings = typename TypeTraitsHelper::Default> // Settings from below
struct SorProxGPUVariantSettings{
   typedef _ConvexSet TConvexSet;
   //STATIC_ASSERT( (TypeTraitsHelper::IsSame<TConvexSet,ConvexSets::RPlusAndDisk>::result))
   typedef char static_assert_failed[ ((TypeTraitsHelper::IsSame<_ConvexSet,ConvexSets::RPlusAndDisk>::result)) ? 1 : -1 ];
   typedef typename ProxIterationType::SOR TProxIterationType;

   template <typename _PREC, int _nMaxIterations, bool _bAbortIfConverged, int _nCheckConvergedFlag, bool _bMatchCPUToGPU> // Settings from above
   struct GPUVariant{
      // TODO Specialize this GPUVariant on all ConvexSets, the class is gonna be so much different, that its worth to specialize the whole class new!
      typedef SorProxGPUVariant< SorProxGPUVariantSettingsWrapper<_PREC, _VariantId, TConvexSet,_alignMatrix,_nMaxIterations,_bAbortIfConverged, _nCheckConvergedFlag,_bMatchCPUToGPU,_KernelSettings>, TConvexSet >  TGPUVariant;
   };

};

template<typename _PREC, int _VariantId, typename _ConvexSet, int _alignMatrix, int _nMaxIterations, bool _bAbortIfConverged, int _nCheckConvergedFlag, bool _bMatchCPUToGPU, typename _KernelSettings>
struct JorProxGPUVariantSettingsWrapper{
   typedef  _PREC PREC;
   typedef  _ConvexSet  TConvexSet;
   typedef  _KernelSettings  TKernelSettings;
   static const int VariantId = _VariantId;
   static const int alignMatrix = _alignMatrix;
   static const int nMaxIterations = _nMaxIterations;
   static const bool bAbortIfConverged = _bAbortIfConverged;
   static const int nCheckConvergedFlag = _nCheckConvergedFlag;
   static const bool bMatchCPUToGPU = _bMatchCPUToGPU;
};

#define DEFINE_JorProxGPUVariant_SettingsWrapper(_TSettings_) \
   typedef typename _TSettings_::PREC PREC; \
   typedef typename _TSettings_::TConvexSet  TConvexSet; \
   typedef typename _TSettings_::TKernelSettings TKernelSettings; \
   static const int VariantId = _TSettings_::VariantId; \
   static const int alignMatrix = _TSettings_::alignMatrix; \
   static const int nMaxIterations = _TSettings_::nMaxIterations; \
   static const bool bAbortIfConverged = _TSettings_::bAbortIfConverged; \
   static const int nCheckConvergedFlag = _TSettings_::nCheckConvergedFlag; \
   static const bool bMatchCPUToGPU = _TSettings_::bMatchCPUToGPU;

template<typename TJorProxGPUVariantSettingsWrapper, typename TConvexSet> class JorProxGPUVariant; // Prototype

template<int _VariantId, typename _ConvexSet, bool _alignMatrix, typename _KernelSettings = TypeTraitsHelper::Default> // Settings from below
struct JorProxGPUVariantSettings{
   typedef _ConvexSet TConvexSet;
   STATIC_ASSERTM( (TypeTraitsHelper::IsSame<TConvexSet,ConvexSets::RPlusAndDisk>::result) , ONLY_RPLUS_AND_DISK_IS_IMPLEMENTED_SO_FAR)
   typedef typename ProxIterationType::JOR TProxIterationType;

   template <typename _PREC, int _nMaxIterations, bool _bAbortIfConverged, int _nCheckConvergedFlag, bool _bMatchCPUToGPU > // Settings from above
   struct GPUVariant{
      // TODO Specialize this GPUVariant on all ConvexSets, the class is gonna be so much different, that its worth to specialize the whole class new!
      typedef JorProxGPUVariant< JorProxGPUVariantSettingsWrapper<_PREC,_VariantId, TConvexSet,_alignMatrix,_nMaxIterations,_bAbortIfConverged, _nCheckConvergedFlag, _bMatchCPUToGPU,_KernelSettings>, TConvexSet>  TGPUVariant;
   };

};


#endif
