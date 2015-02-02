
#ifndef ConvergenceCheckKernelWrap_hpp
#define ConvergenceCheckKernelWrap_hpp

#include <cmath>
#include <iostream>
#include "CudaMatrix.hpp"
#include "VariantLaunchSettings.hpp"

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
