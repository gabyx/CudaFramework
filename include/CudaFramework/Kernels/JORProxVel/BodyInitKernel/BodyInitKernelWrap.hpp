
#ifndef BodyInitKernelWrap_hpp
#define BodyInitKernelWrap_hpp


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
