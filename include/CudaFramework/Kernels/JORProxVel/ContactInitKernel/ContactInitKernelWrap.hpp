// ========================================================================================
//  CudaFramework
//  Copyright (C) 2014 by Gabriel Nützi <nuetzig (at) imes (d0t) mavt (d0t) ethz (d0t) ch>
//
//  This Source Code Form is subject to the terms of the GNU GPL 3.0 licence. 
//  If a copy of the GNU GPL 3.0 was not distributed with this
//  file, you can obtain one at http://opensource.org/licenses/GPL-3.0.
// ========================================================================================


#ifndef CudaFramework_Kernels_JORProxVel_ContactInitKernel_ContactInitKernelWrap_hpp
#define CudaFramework_Kernels_JORProxVel_ContactInitKernel_ContactInitKernelWrap_hpp

#include <cmath>
#include <iostream>
#include "CudaFramework/CudaModern/CudaMatrix.hpp"
#include "CudaFramework/Kernels/JORProxVel/VariantLaunchSettings.hpp"

namespace ContactInit{
template<bool genOutput,typename TCudaMatrix,typename TCudaIntMatrix>
void contactInitKernelWrap(        TCudaMatrix bodyBuffer,
                                   TCudaMatrix contactBuffer,
                                   TCudaIntMatrix indexBuffer,
                                   TCudaIntMatrix globalBuffer,
                                   TCudaMatrix outputBuffer,
                                   unsigned int numberOfContacts,
                                   VariantLaunchSettings variantSettings);


}
#endif
