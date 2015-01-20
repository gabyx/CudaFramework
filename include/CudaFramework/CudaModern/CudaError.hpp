/******************************************************************************
* Copyright (c) 2013, NVIDIA CORPORATION. All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
* * Redistributions of source code must retain the above copyright
* notice, this list of conditions and the following disclaimer.
* * Redistributions in binary form must reproduce the above copyright
* notice, this list of conditions and the following disclaimer in the
* documentation and/or other materials provided with the distribution.
* * Neither the name of the NVIDIA CORPORATION nor the
* names of its contributors may be used to endorse or promote products
* derived from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
* ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
*
*  Source code modified and extended from moderngpu.com
******************************************************************************/

#ifndef CudaFramework_CudaModern_CudaError_hpp
#define CudaFramework_CudaModern_CudaError_hpp

#include <typeinfo>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "CudaFramework/General/Exception.hpp"
#include "CudaFramework/CudaModern/CudaException.hpp"


#define HANDLE_CUDA_ERRORS 1

// ERROR CHECKS CUDA ===============================================================
#if HANDLE_CUDA_ERRORS == 1


	#define CHECK_CUDA( __term__ ) \
        { \
            cudaError_t __err__ = __term__ ; \
            if ( __err__ != cudaSuccess) { \
                THROW_CUDA("CUDA ERROR: " <<cudaGetErrorString( __err__ )  , __err__ ); \
            } \
        }

    #define CHECK_CUDA_M( __term__ , __message__ ) \
        { \
            cudaError_t __err__ = __term__ ; \
            if (__err__ != cudaSuccess) { \
                THROW_CUDA( __message__ << " , CUDA ERROR: " <<cudaGetErrorString( __err__ )  , __err__ ); \
            } \
        }

    #define CHECK_CUDA_LAST  \
        { \
            cudaError_t __err__ = cudaGetLastError() ; \
            if (__err__ != cudaSuccess) { \
                THROW_CUDA("CUDA ERROR: " <<cudaGetErrorString( __err__ ) ,  __err__ ); \
            } \
        }

// CUBLAS_STATUS_SUCCESS           0x00000000 = 0
// CUBLAS_STATUS_NOT_INITIALIZED   0x00000001 = 1
// CUBLAS_STATUS_ALLOC_FAILED      0x00000003 = 3
// CUBLAS_STATUS_INVALID_VALUE     0x00000007 = 6
// CUBLAS_STATUS_ARCH_MISMATCH     0x00000008 = 8
// CUBLAS_STATUS_MAPPING_ERROR     0x0000000B = 11
// CUBLAS_STATUS_EXECUTION_FAILED  0x0000000D = 13
// CUBLAS_STATUS_INTERNAL_ERROR    0x0000000E = 14

	#define CHECK_CUBLAS( __term__ ) \
        { \
            cublasStatus_t __err__ = __term__ ; \
            if (__err__ != CUBLAS_STATUS_SUCCESS) { \
                THROW_CUBLAS( "CUBLAS ERROR: " << __err__   , __err__ ); \
            } \
        }


#else
	#define CHECK_CUDA( __term__ ) __term__ ;
	#define CHECK_CUDA_M( __term__ , __message__)  __term__ ;
    #define CHECK_CUDA_LAST
	#define CHECK_CUBLAS( __term__ ) __term__ ;
#endif


// ASSERTS ==================================================

#ifndef NDEBUG
    //DEBUG
    #define ASSERTCHECK_CUDA( __term__ ) \
        { \
            cudaError_t __err__ = __term__ ; \
            if (__err__ != cudaSuccess) { \
                THROW_CUDA("CUDA ERROR: " <<cudaGetErrorString(__err__ )  , __err__ ); \
            } \
        }

    #define ASSERTCHECK_CUDA_LAST  \
        { \
            cudaError_t __err__ = cudaGetLastError() ; \
            if (__err__ != cudaSuccess) { \
                THROW_CUDA("CUDA ERROR: " <<cudaGetErrorString( __err__ ) ,  __err__ ); \
            } \
        }

#else

    #define ASSERTCHECK_CUDA( __term__ ) __term__
    #define ASSERTCHECK_CUDA_LAST

#endif


// ERRORS ===========================================================================00
#define ERRORMSG_CUDA( __message__ ) \
    THROWEXCEPTION ( __message__ )


// DESTRUCTOR MESSAGE ================================================================
#ifndef NDEBUG
    #define CUDA_DESTRUCTOR_MESSAGE( __thisptr__) \
        std::cout << "Destruction of: "<< typeid( __thisptr__ ).name() << " @ " << __thisptr__ << std::endl;
#else
    #define CUDA_DESTRUCTOR_MESSAGE( __thisptr__)
#endif

#endif
