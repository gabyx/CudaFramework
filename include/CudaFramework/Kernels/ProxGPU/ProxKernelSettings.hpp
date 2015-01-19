// ========================================================================================
//  CudaFramework
//  Copyright (C) 2014 by Gabriel Nützi <nuetzig (at) imes (d0t) mavt (d0t) ethz (d0t) ch>
//
//  This Source Code Form is subject to the terms of the GNU GPL 3.0 licence.
//  If a copy of the GNU GPL 3.0 was not distributed with this
//  file, you can obtain one at http://opensource.org/licenses/GPL-3.0.
// ========================================================================================

#ifndef CudaFramework_Kernels_ProxGPU_ProxKernelSettings_hpp
#define CudaFramework_Kernels_ProxGPU_ProxKernelSettings_hpp


#include "CudaFramework/General/StaticAssert.hpp"
#include "CudaFramework/General/GPUDefines.hpp"
#include "CudaFramework/General/TypeTraitsHelper.hpp"
#include "ConvexSets.hpp"


/**
* These are the defines for the following kernel: jorProxContactOrdered_RPlusAndDisk_1threads_kernel( utilCuda::CudaMatrix<PREC> mu_dev,utilCuda::CudaMatrix<PREC> y_dev, int incr_y,PREC alpha, utilCuda::CudaMatrix<PREC> A_dev,utilCuda::CudaMatrix<PREC> x_dev,int incr_x,PREC beta, utilCuda::CudaMatrix<PREC> b_dev,int incr_b)
*/


template<int _ThreadsPerBlock,int _BlockDim, int _XElementsPerThread, typename _ConvexSet ,int _UnrollBlockDotProduct>
struct JorProxKernelSettings{
   static const int ThreadsPerBlock = _ThreadsPerBlock;                       // How many threads we have assigned to each block
   static const int BlockDim = _BlockDim;                                     // How many threads calculate 1 result in y_dev, only BLOCK_DIM results are computed. THREADS_PER_BLOCK - BLOCK_DIM threads are idle!. Can also be less than THREADS_PER_BLOCK
   STATIC_ASSERT(BlockDim <= ThreadsPerBlock)                                // Because this is less then 128, it gives very bad performance


   static const int XElementsPerThread = _XElementsPerThread;                 // How many elements are processed by one thread!, Currently supported, 1,2,4!
   STATIC_ASSERT(XElementsPerThread  == 1 || XElementsPerThread  == 2 || XElementsPerThread  == 4)

   typedef _ConvexSet TConvexSet;
   static const int ProxPackageSize = TConvexSet::Dimension;                   // So many threads are used out of BLOCK_DIM to prox all 42 packages, 1 thread = 1 package!

   STATIC_ASSERT( (TypeTraitsHelper::IsSame<TConvexSet,ConvexSets::RPlusAndDisk>::result) )    // Only this has been implemented so far
   STATIC_ASSERT(BlockDim % ProxPackageSize == 0)
   static const int ProxPackages = BlockDim / ProxPackageSize;


   static const int UnrollBlockDotProduct = _UnrollBlockDotProduct;
};

typedef JorProxKernelSettings<128,126,4,ConvexSets::RPlusAndDisk,6> JorProxSettings3RPlusAndDisk;
typedef JorProxKernelSettings<192,192,4,ConvexSets::RPlusAndDisk,6> JorProxSettings4RPlusAndDisk;



#define STR(tok) #tok
#define XSTR(tok) STR(tok)
#define SEP_STR ", "
#define JORPROXKERNEL_SETTINGS_STR( _ProxSettings_ ) \
"["<<"ThreadsPerBlock: " << _ProxSettings_::ThreadsPerBlock << SEP_STR << \
"BlockDim: " << _ProxSettings_::BlockDim << SEP_STR << \
"XElementsPerThread: " << _ProxSettings_::XElementsPerThread << SEP_STR << \
"ProxPackageSize: " << _ProxSettings_::ProxPackageSize << SEP_STR << \
"ProxPackages: " << _ProxSettings_::ProxPackages << SEP_STR << \
"UnrollBlockDotProduct: " << _ProxSettings_::UnrollBlockDotProduct <<"]"








/**
* These are the settings for the full dependency SOR algorithm
* Good number for MaxProxPackagesKernelA are: 32    64    96   128   160   192   224   256   288
* So far: 32 or (48) gives the best results
*/
template<int _MaxProxPackagesKernelA, typename _ConvexSet>
class SorProxKernelSettings{
public:
   //Kernel A, related parameters!
   static const int ProxPackages = _MaxProxPackagesKernelA;         // How many ProxPackages we have in the Kernel Block A
   typedef _ConvexSet TConvexSet;
   static const int ProxPackageSize = TConvexSet::Dimension;                   // So many threads are used out of BLOCK_DIM to prox all 42 packages, 1 thread = 1 package!

   STATIC_ASSERT( (TypeTraitsHelper::IsSame<TConvexSet,ConvexSets::RPlusAndDisk>::result) )    // Only this has been implemented so far

   static const int BlockDimKernelA = ProxPackages * ProxPackageSize;      // How big the entire Block matrix is

   STATIC_ASSERT(BlockDimKernelA % ProxPackageSize == 0)

   static const int ThreadsPerBlockKernelA = BlockDimKernelA;
   STATIC_ASSERT( ThreadsPerBlockKernelA % GPU_WarpSize == 0 )      // ThreadsPerBlock needs to be a multiple of the warp size!


   //Kernel B, related parameters!
   static const int BlockDimKernelB = 128;
   static const int ThreadsPerBlockKernelB = 128;
   static const int XElementsPerThreadKernelB = ((BlockDimKernelA + (ThreadsPerBlockKernelB-1)) / (ThreadsPerBlockKernelB) );
   static const int UnrollBlockDotProductKernelB = 6;

};

typedef SorProxKernelSettings<32,ConvexSets::RPlusAndDisk> SorProxSettings1RPlusAndDisk;
typedef SorProxKernelSettings<64,ConvexSets::RPlusAndDisk> SorProxSettings2RPlusAndDisk;


#define SOR_PROXKERNEL_SETTINGS_STR( _ProxSettings_ ) \
"["<<"ThdsPerBl A: " << _ProxSettings_::ThreadsPerBlockKernelA << SEP_STR << \
"BlDim A: " << _ProxSettings_::BlockDimKernelA << SEP_STR << \
"ProxPkgSize A: " << _ProxSettings_::ProxPackageSize << SEP_STR << \
"ProxPkgs A: " << _ProxSettings_::ProxPackages << "]" << \
"["<<"ThdsPerBl B: " << _ProxSettings_::ThreadsPerBlockKernelB << SEP_STR << \
"BlDim B: " << _ProxSettings_::BlockDimKernelB << SEP_STR << \
"XElemsPerThd B: " << _ProxSettings_::XElementsPerThreadKernelB << SEP_STR << \
"UnrollBlDotProd B: " << _ProxSettings_::UnrollBlockDotProductKernelB << "]"


#define SOR_PROXKERNEL_A_SETTINGS_STR( _ProxSettings_ ) \
"["<<"ThdsPerBl A: " << _ProxSettings_::ThreadsPerBlockKernelA << SEP_STR << \
"BlDim A: " << _ProxSettings_::BlockDimKernelA << SEP_STR << \
"ProxPkgSize A: " << _ProxSettings_::ProxPackageSize << SEP_STR << \
"ProxPkgs A: " << _ProxSettings_::ProxPackages << "]"


#define SOR_PROXKERNEL_B_SETTINGS_STR( _ProxSettings_ ) \
"["<<"ThdsPerBl B: " << _ProxSettings_::ThreadsPerBlockKernelB << SEP_STR << \
"BlDim B: " << _ProxSettings_::BlockDimKernelB << SEP_STR << \
"XElemsPerThd B: " << _ProxSettings_::XElementsPerThreadKernelB << SEP_STR << \
"UnrollBlDotProd B: " << _ProxSettings_::UnrollBlockDotProductKernelB << "]"





/**
* These are the settings for the relaxed dependency SOR algorithm
* Good number for MaxProxPackagesKernelA are:
*/
template<int _MaxProxPackagesKernelA, typename _ConvexSet>
class RelaxedSorProxKernelSettings{
public:
   //Kernel A, related parameters!
   static const int ProxPackages = _MaxProxPackagesKernelA;         // How many ProxPackages we have in the Kernel Block A
   typedef _ConvexSet TConvexSet;
   static const int ProxPackageSize = TConvexSet::Dimension;                   // So many threads are used out of BLOCK_DIM to prox all 42 packages, 1 thread = 1 package!

   STATIC_ASSERT( (TypeTraitsHelper::IsSame<TConvexSet,ConvexSets::RPlusAndDisk>::result) )    // Only this has been implemented so far

   static const int BlockDimKernelA = ProxPackages * ProxPackageSize;      // How big the entire Block matrix is

   STATIC_ASSERT( (BlockDimKernelA % (ProxPackageSize*1) == 0) )   // It does not work if we take out "*1" ?? CUDA Compiler BUG?

   static const int ThreadsPerBlockKernelA = 128;
   STATIC_ASSERT( ThreadsPerBlockKernelA % GPU_WarpSize == 0)      // ThreadsPerBlock needs to be a multiple of the warp size!


   ////Kernel B, related parameters!
   static const int BlockDimKernelB = 128;
   static const int ThreadsPerBlockKernelB = 128;
   static const int XElementsPerThreadKernelB = 4 ;
   static const int UnrollBlockDotProductKernelB = 6;

};

typedef RelaxedSorProxKernelSettings<8,ConvexSets::RPlusAndDisk> RelaxedSorProxSettings1RPlusAndDisk;


#define RELAXED_SOR_PROXKERNEL_SETTINGS_STR( _ProxSettings_ ) \
"["<<"ThdsPerBl A: " << _ProxSettings_::ThreadsPerBlockKernelA << SEP_STR << \
"BlDim A: " << _ProxSettings_::BlockDimKernelA << SEP_STR << \
"ProxPkgSize A: " << _ProxSettings_::ProxPackageSize << SEP_STR << \
"ProxPkgs A: " << _ProxSettings_::ProxPackages << "]" << \
"["<<"ThdsPerBl B: " << _ProxSettings_::ThreadsPerBlockKernelB << SEP_STR << \
"BlDim B: " << _ProxSettings_::BlockDimKernelB << SEP_STR << \
"XElemsPerThd B: " << _ProxSettings_::XElementsPerThreadKernelB << SEP_STR << \
"UnrollBlDotProd B: " << _ProxSettings_::UnrollBlockDotProductKernelB << "]"


#define RELAXED_SOR_PROXKERNEL_A_SETTINGS_STR( _ProxSettings_ ) \
"["<<"ThdsPerBl A: " << _ProxSettings_::ThreadsPerBlockKernelA << SEP_STR << \
"BlDim A: " << _ProxSettings_::BlockDimKernelA << SEP_STR << \
"ProxPkgSize A: " << _ProxSettings_::ProxPackageSize << SEP_STR << \
"ProxPkgs A: " << _ProxSettings_::ProxPackages << "]"


#define RELAXED_SOR_PROXKERNEL_B_SETTINGS_STR( _ProxSettings_ ) \
"["<<"ThdsPerBl B: " << _ProxSettings_::ThreadsPerBlockKernelB << SEP_STR << \
"BlDim B: " << _ProxSettings_::BlockDimKernelB << SEP_STR << \
"XElemsPerThd B: " << _ProxSettings_::XElementsPerThreadKernelB << SEP_STR << \
"UnrollBlDotProd B: " << _ProxSettings_::UnrollBlockDotProductKernelB << "]"




#endif
