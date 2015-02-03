// ========================================================================================
//  CudaFramework
//  Copyright (C) 2014 by Gabriel Nützi <nuetzig (at) imes (d0t) mavt (d0t) ethz (d0t) ch>
//
//  This Source Code Form is subject to the terms of the GNU GPL 3.0 licence. 
//  If a copy of the GNU GPL 3.0 was not distributed with this
//  file, you can obtain one at http://opensource.org/licenses/GPL-3.0.
// ========================================================================================


#ifndef CudaFramework_Kernels_JORProxVel_ContactIterationKernel_ContactIterationKernelWrap_hpp
#define CudaFramework_Kernels_JORProxVel_ContactIterationKernel_ContactIterationKernelWrap_hpp

#include <cmath>
#include <iostream>
#include "CudaFramework/CudaModern/CudaMatrix.hpp"
#include "CudaFramework/Kernels/JORProxVel/VariantLaunchSettings.hpp"

namespace ContIter{
template<bool genOutput,bool convInVel,typename TCudaMatrix,typename TCudaIntMatrix>
void contIterKernelWrap(    TCudaMatrix bodyDev,
                            TCudaMatrix contactDev,
                            TCudaIntMatrix globalDev,
                            typename TCudaMatrix::PREC * reductionDev,
                            TCudaIntMatrix indexSetDev,
                            TCudaMatrix outputBuffer,
                            unsigned int numberOfContacts,
                            VariantLaunchSettings variantSettings,
                            unsigned int totalRedNumber,
                            typename TCudaMatrix::PREC relTol,
                            typename TCudaMatrix::PREC absTol);


}



#endif
