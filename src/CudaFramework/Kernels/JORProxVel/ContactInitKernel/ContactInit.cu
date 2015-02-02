#include <cuda_runtime.h>

#include "CudaFramework/Kernels/JORProxVel/ContactInitKernel/ContactInit.cuh"

#include "CudaFramework/CudaModern/CudaMatrix.hpp"
#include "CudaFramework/Kernels/JORProxVel/VariantLaunchSettings.hpp"

#define PREC double
template __host__ void  ContactInit::contactInitKernelWrap<false>(utilCuda::CudaMatrix<PREC> bodyBuffer,
                                                     utilCuda::CudaMatrix<PREC> contactBuffer,
                                                     utilCuda::CudaMatrix<unsigned int> indexBuffer,
                                                     utilCuda::CudaMatrix<unsigned int> globalBuffer,
                                                     utilCuda::CudaMatrix<PREC> outputBuffer,
                                                     unsigned int numberOfContacts,
                                                     VariantLaunchSettings variantSettings);
#undef PREC

#define PREC float
template __host__ void  ContactInit::contactInitKernelWrap<false>(utilCuda::CudaMatrix<PREC> bodyBuffer,
                                                     utilCuda::CudaMatrix<PREC> contactBuffer,
                                                     utilCuda::CudaMatrix<unsigned int> indexBuffer,
                                                     utilCuda::CudaMatrix<unsigned int> globalBuffer,
                                                     utilCuda::CudaMatrix<PREC> outputBuffer,
                                                     unsigned int numberOfContacts,
                                                     VariantLaunchSettings variantSettings);
#undef PREC
// Explicit code generation for the CUDA compile side (for all template parameter)
#define PREC double
template __host__ void  ContactInit::contactInitKernelWrap<true>(utilCuda::CudaMatrix<PREC> bodyBuffer,
                                                     utilCuda::CudaMatrix<PREC> contactBuffer,
                                                     utilCuda::CudaMatrix<unsigned int> indexBuffer,
                                                     utilCuda::CudaMatrix<unsigned int> globalBuffer,
                                                     utilCuda::CudaMatrix<PREC> outputBuffer,
                                                     unsigned int numberOfContacts,
                                                     VariantLaunchSettings variantSettings);
#undef PREC

#define PREC float
template __host__ void  ContactInit::contactInitKernelWrap<true>(utilCuda::CudaMatrix<PREC> bodyBuffer,
                                                     utilCuda::CudaMatrix<PREC> contactBuffer,
                                                     utilCuda::CudaMatrix<unsigned int> indexBuffer,
                                                     utilCuda::CudaMatrix<unsigned int> globalBuffer,
                                                     utilCuda::CudaMatrix<PREC> outputBuffer,
                                                     unsigned int numberOfContacts,
                                                     VariantLaunchSettings variantSettings);
#undef PREC


