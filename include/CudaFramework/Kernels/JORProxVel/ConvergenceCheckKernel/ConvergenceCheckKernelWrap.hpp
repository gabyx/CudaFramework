// ========================================================================================
//  CudaFramework
//  Copyright (C) 2014 by Gabriel Nützi <nuetzig (at) imes (d0t) mavt (d0t) ethz (d0t) ch>
//
//  This Source Code Form is subject to the terms of the GNU GPL 3.0 licence. 
//  If a copy of the GNU GPL 3.0 was not distributed with this
//  file, you can obtain one at http://opensource.org/licenses/GPL-3.0.
// ========================================================================================


#ifndef CudaFramework_Kernels_JORProxVel_ConvergenceCheckKernel_ConvergenceCheckKernelWrap_hpp
#define CudaFramework_Kernels_JORProxVel_ConvergenceCheckKernel_ConvergenceCheckKernelWrap_hpp

#include <cmath>
#include <iostream>
#include "CudaFramework/CudaModern/CudaMatrix.hpp"
#include "CudaFramework/Kernels/JORProxVel/VariantLaunchSettings.hpp"

namespace ConvCheck{
template<bool genOutput,bool convInVel,typename PREC,typename TCudaMatrix,typename TCudaIntMatrix>
void convCheckKernelWrap(TCudaMatrix bodyBuffer,
                         TCudaIntMatrix globalBuffer,
                         TCudaMatrix outputBuffer,
                         PREC* redBufferIn,
                         unsigned int numberOfBodies,
                         VariantLaunchSettings variantSettings,
                         PREC relTol,
                         PREC absTol) ;


}

#endif
