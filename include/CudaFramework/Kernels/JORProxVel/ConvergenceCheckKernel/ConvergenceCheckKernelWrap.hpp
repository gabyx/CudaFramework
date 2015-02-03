
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
