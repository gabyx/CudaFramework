// ========================================================================================
//  CudaFramework
//  Copyright (C) 2014 by Gabriel Nützi <nuetzig (at) imes (d0t) mavt (d0t) ethz (d0t) ch>
//
//  This Source Code Form is subject to the terms of the GNU GPL 3.0 licence. 
//  If a copy of the GNU GPL 3.0 was not distributed with this
//  file, you can obtain one at http://opensource.org/licenses/GPL-3.0.
// ========================================================================================


#ifndef CudaFramework_Kernels_JORProxVel_BodyInitKernel_BodyInitKernelWrap_hpp
#define CudaFramework_Kernels_JORProxVel_BodyInitKernel_BodyInitKernelWrap_hpp


/// Declaration for GCC
namespace  BodyInit{
template<bool generateOutput,typename PREC,typename VariantLaunchSettings,typename TCudaMatrix,typename TCudaIntMatrix>
void  bodyInitKernelWrap(TCudaMatrix bodyBuffer,
                          TCudaIntMatrix globalBuffer,
                          TCudaMatrix outputBuffer,
                          unsigned int numberOfBodies,
                          VariantLaunchSettings variantSettings,
                          PREC deltaTime
                          );


}

#endif
