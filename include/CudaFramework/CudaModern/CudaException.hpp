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

#ifndef CudaFramework_CudaModern_CudaException_hpp
#define CudaFramework_CudaModern_CudaException_hpp

#include <iostream>
#include <stdexcept>
#include <exception>
#include <string>
#include <sstream>

#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace utilCuda{

    class CudaException : public std::runtime_error {
    public:
        cudaError_t m_error;

        CudaException() throw() : std::runtime_error("")  { }
        CudaException(cudaError_t e) throw() :  std::runtime_error("") , m_error(e) { }
        CudaException(const std::stringstream & ss, cudaError_t e): std::runtime_error(ss.str()), m_error(e) {};
        CudaException(const CudaException& e) throw() : std::runtime_error(*this),  m_error(e.m_error) { }

        virtual const char* what() const throw() {
            return std::runtime_error::what();
        }
    };

    class CublasException : public std::runtime_error {
    public:
        cublasStatus_t m_error;

        CublasException() throw() : std::runtime_error("")  { }
        CublasException(cublasStatus_t e) throw() :  std::runtime_error("") , m_error(e) { }
        CublasException(const std::stringstream & ss, cublasStatus_t e): std::runtime_error(ss.str()), m_error(e) {};
        CublasException(const CublasException& e) throw() : std::runtime_error(*this),  m_error(e.m_error) { }

        virtual const char* what() const throw() {
            return std::runtime_error::what();
        }
    };

};

#define THROW_CUDA( message , err ) \
    { \
        std::stringstream s; \
        s << message << std::endl << " @ " << __FILE__ << " (" << __LINE__ << ")" << std::endl; \
        throw utilCuda::CudaException(s,err); \
    }

#define THROW_CUBLAS( message , err ) \
    { \
        std::stringstream s; \
        s << message << std::endl << " @ " << __FILE__ << " (" << __LINE__ << ")" << std::endl; \
        throw utilCuda::CublasException(s,err); \
    }

#endif
