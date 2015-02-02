
#ifndef ContactKernelFrameKernel_hpp
#define ContactKernelFrameKernel_hpp

#include <cmath>
#include <iostream>
#include "CudaMatrix.hpp"
#include "VariantLaunchSettings.hpp"

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
