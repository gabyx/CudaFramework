#include <cuda_runtime.h>

#include <BodyInit.cuh>
#include "CudaMatrix.hpp"
#include "VariantLaunchSettings.hpp"



#define PREC double
template __host__ void   BodyInit::bodyInitKernelWrap<false,PREC>(    utilCuda::CudaMatrix<PREC> bodyBuffer,
                                                           utilCuda::CudaMatrix<unsigned int> globalBuffer,
                                                           utilCuda::CudaMatrix<PREC> outputBuffer,
                                                           unsigned int numberOfContacts,
                                                           VariantLaunchSettings variantSettings,
                                                           PREC deltaTime);
#undef PREC

#define PREC float
template __host__ void   BodyInit::bodyInitKernelWrap<false,PREC>(    utilCuda::CudaMatrix<PREC> bodyBuffer,
                                                           utilCuda::CudaMatrix<unsigned int> globalBuffer,
                                                           utilCuda::CudaMatrix<PREC> outputBuffer,
                                                           unsigned int numberOfContacts,
                                                           VariantLaunchSettings variantSettings,
                                                           PREC deltaTime);
#undef PREC
// Explicit code generation for the CUDA compile side (for all template parameter)
#define PREC double
template __host__ void   BodyInit::bodyInitKernelWrap<true,PREC>(  utilCuda::CudaMatrix<PREC> bodyBuffer,
                                                   utilCuda::CudaMatrix<unsigned int> globalBuffer,
                                                   utilCuda::CudaMatrix<PREC> outputBuffer,
                                                   unsigned int numberOfBodies,
                                                   VariantLaunchSettings variantSettings,
                                                   PREC deltaTime);
#undef PREC

#define PREC float
template __host__ void   BodyInit::bodyInitKernelWrap<true,PREC>(  utilCuda::CudaMatrix<PREC> bodyBuffer,
                                                   utilCuda::CudaMatrix<unsigned int> globalBuffer,
                                                   utilCuda::CudaMatrix<PREC> outputBuffer,
                                                   unsigned int numberOfBodies,
                                                   VariantLaunchSettings variantSettings,
                                                   PREC deltaTime);
#undef PREC

