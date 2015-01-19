// ========================================================================================
//  CudaFramework
//  Copyright (C) 2014 by Gabriel Nützi <nuetzig (at) imes (d0t) mavt (d0t) ethz (d0t) ch>
//
//  This Source Code Form is subject to the terms of the GNU GPL 3.0 licence. 
//  If a copy of the GNU GPL 3.0 was not distributed with this
//  file, you can obtain one at http://opensource.org/licenses/GPL-3.0.
// ========================================================================================
#ifndef CudaFramework_Kernels_ProxGPU_RelaxedSorProxKernelSettings_hpp
#define CudaFramework_Kernels_ProxGPU_RelaxedSorProxKernelSettings_hpp


#include "CudaFramework/General/StaticAssert.hpp"
#include "TypenameComparision.hpp"
#include "CudaFramework/General/GPUDefines.hpp"

#include "ConvexSets.hpp"




/**
* These are the settings for the relaxed dependency SOR algorithm
* Good number for MaxProxPackagesKernelA are: 
*/
template<int _MaxProxPackagesKernelA, typename _ConvexSet>
struct RelaxedSorProxKernelSettings{
   //Kernel A, related parameters!
   static const int ProxPackages = _MaxProxPackagesKernelA;         // How many ProxPackages we have in the Kernel Block A
   typedef typename _ConvexSet TConvexSet;
   static const int ProxPackageSize = _ConvexSet::Dimension;                   // So many threads are used out of BLOCK_DIM to prox all 42 packages, 1 thread = 1 package!
   
   STATIC_ASSERT( (IsSame<TConvexSet,ConvexSets::RPlusAndDisk>::Result::value));    // Only this has been implemented so far
   
   static const int BlockDimKernelA = ProxPackages * ProxPackageSize;      // How big the entire Block matrix is
    STATIC_ASSERT(ProxPackages == 3 || ProxPackages == 256);
   STATIC_ASSERT((BlockDimKernelA % ProxPackageSize) == 0); // It does not work if we take this in??? CUDA Compiler BUG?

   static const int ThreadsPerBlockKernelA = 128;
   STATIC_ASSERT( ThreadsPerBlockKernelA % GPU_WarpSize == 0);      // ThreadsPerBlock needs to be a multiple of the warp size!


   //Kernel B, related parameters!
   static const int BlockDimKernelB = 128;
   static const int ThreadsPerBlockKernelB = 128;
   static const int XElementsPerThreadKernelB = 3 ;
   static const int UnrollBlockDotProductKernelB = 6;
   

   
};

typedef RelaxedSorProxKernelSettings<3,ConvexSets::RPlusAndDisk> RelaxedSorProxSettings1RPlusAndDisk;
typedef RelaxedSorProxKernelSettings<256,ConvexSets::RPlusAndDisk> RelaxedSorProxSettings2RPlusAndDisk;



#define RELAXED_SOR_PROXKERNEL_SETTINGS_STR( _ProxSettings_ ) \
"ThdsPerBl A: " << _ProxSettings_::ThreadsPerBlockKernelA << SEP_STR << \
"BlDim A: " << _ProxSettings_::BlockDimKernelA << SEP_STR << \
"ProxPkgSize A: " << _ProxSettings_::ProxPackageSize << SEP_STR << \
"ProxPkgs A: " << _ProxSettings_::ProxPackages << SEP_STR << \
"ThdsPerBl B: " << _ProxSettings_::ThreadsPerBlockKernelB << SEP_STR << \
"BlDim B: " << _ProxSettings_::BlockDimKernelB << SEP_STR << \
"XElemsPerThd B: " << _ProxSettings_::XElementsPerThreadKernelB << SEP_STR << \
"UnrollBlDotProd B: " << _ProxSettings_::UnrollBlockDotProductKernelB << SEP_STR


#define RELAXED_SOR_PROXKERNEL_A_SETTINGS_STR( _ProxSettings_ ) \
"ThdsPerBl A: " << _ProxSettings_::ThreadsPerBlockKernelA << SEP_STR << \
"BlDim A: " << _ProxSettings_::BlockDimKernelA << SEP_STR << \
"ProxPkgSize A: " << _ProxSettings_::ProxPackageSize << SEP_STR << \
"ProxPkgs A: " << _ProxSettings_::ProxPackages << SEP_STR 


#define RELAXED_SOR_PROXKERNEL_B_SETTINGS_STR( _ProxSettings_ ) \
"ThdsPerBl B: " << _ProxSettings_::ThreadsPerBlockKernelB << SEP_STR << \
"BlDim B: " << _ProxSettings_::BlockDimKernelB << SEP_STR << \
"XElemsPerThd B: " << _ProxSettings_::XElementsPerThreadKernelB << SEP_STR << \
"UnrollBlDotProd B: " << _ProxSettings_::UnrollBlockDotProductKernelB << SEP_STR

#endif