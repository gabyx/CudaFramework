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

#ifndef CudaFramework_CudaModern_CudaDeviceMemory_hpp
#define CudaFramework_CudaModern_CudaDeviceMemory_hpp

#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "CudaFramework/CudaModern/CudaTypeDefs.hpp"
#include "CudaFramework/CudaModern/CudaRefcounting.hpp"
#include "CudaFramework/CudaModern/CudaAlloc.hpp"


namespace utilCuda{

    class CudaMemSupport;
    class CudaMatrixSupport;

    template<typename T>
    class CudaDeviceMem : public ReferenceCounting<void,CudaDeviceMem<T> > {
        friend class CudaMemSupport;
        friend class CudaMatrixSupport;
    public:
        ~CudaDeviceMem();

        const T* get() const { return m_pDevice; }
        T* get() { return m_pDevice; }

        operator const T*() const { return get(); }
        operator T*() { return get(); }

        // Size is in units of T, not bytes.
        size_t size() const { return m_size; }

        // Copy from this to the argument array.
        cudaError_t toDevice(T* data, size_t count) const;
        cudaError_t toDevice(size_t srcOffest, size_t bytes, void* data) const;
        cudaError_t toHost(T* data, size_t count) const;
        cudaError_t toHost(std::vector<T>& data) const;
        cudaError_t toHost(std::vector<T>& data, size_t count) const;
        cudaError_t toHost(size_t srcOffset, size_t bytes, void* data) const;

        // Copy from the argument array to this.
        cudaError_t fromDevice(const T* data, size_t count);
        cudaError_t fromDevice(size_t dstOffset, size_t bytes, const void* data);
        cudaError_t fromHost(const std::vector<T>& data);
        cudaError_t fromHost(const std::vector<T>& data, size_t count);
        cudaError_t fromHost(const T* data, size_t count);
        cudaError_t fromHost(size_t destOffset, size_t bytes, const void* data);

    protected:
        friend class CudaContext;
        CudaDeviceMem(CudaAlloc* alloc, size_t size) : m_pDevice(nullptr), m_size(size), m_alloc(alloc) { }

        AllocPtrType m_alloc;
        T* m_pDevice;
        size_t m_size;
    };

};


#define MGPU_MEM(type) utilCuda::IntrusivePtr< utilCuda::CudaDeviceMem< type > >


////////////////////////////////////////////////////////////////////////////////
// CudaDeviceMem method implementations
namespace utilCuda{

    template<typename T>
    cudaError_t CudaDeviceMem<T>::toDevice(T* data, size_t count) const {
        return toDevice(0, sizeof(T) * count, data);
    }
    template<typename T>
    cudaError_t CudaDeviceMem<T>::toDevice(size_t srcOffset, size_t bytes,
        void* data) const {
        return cudaMemcpy(data, (char*)m_pDevice + srcOffset, bytes, cudaMemcpyDeviceToDevice);
    }

    template<typename T>
    cudaError_t CudaDeviceMem<T>::toHost(T* data, size_t count) const {
        return toHost(0, sizeof(T) * count, data);
    }
    template<typename T>
    cudaError_t CudaDeviceMem<T>::toHost(std::vector<T>& data, size_t count) const {
        data.resize(count);
        cudaError_t error = cudaSuccess;
        if(m_size) error = toHost(&data[0], count);
        return error;
    }
    template<typename T>
    cudaError_t CudaDeviceMem<T>::toHost(std::vector<T>& data) const {
        return toHost(data, m_size);
    }
    template<typename T>
    cudaError_t CudaDeviceMem<T>::toHost(size_t srcOffset, size_t bytes,
        void* data) const {
        return cudaMemcpy(data, (char*)m_pDevice + srcOffset, bytes, cudaMemcpyDeviceToHost);;
    }

    template<typename T>
    cudaError_t CudaDeviceMem<T>::fromDevice(const T* data, size_t count) {
        return fromDevice(0, sizeof(T) * count, data);
    }
    template<typename T>
    cudaError_t CudaDeviceMem<T>::fromDevice(size_t dstOffset, size_t bytes,
        const void* data) {
        if(dstOffset + bytes > sizeof(T) * m_size)
            return cudaErrorInvalidValue;
        cudaMemcpy(m_pDevice + dstOffset, data, bytes, cudaMemcpyDeviceToDevice);
        return cudaSuccess;
    }
    template<typename T>
    cudaError_t CudaDeviceMem<T>::fromHost(const std::vector<T>& data,
        size_t count) {
        cudaError_t error = cudaSuccess;
        if(data.size()) error = fromHost(&data[0], count);
        return error;
    }
    template<typename T>
    cudaError_t CudaDeviceMem<T>::fromHost(const std::vector<T>& data) {
        return fromHost(data, data.size());
    }
    template<typename T>
    cudaError_t CudaDeviceMem<T>::fromHost(const T* data, size_t count) {
        return fromHost(0, sizeof(T) * count, data);
    }
    template<typename T>
    cudaError_t CudaDeviceMem<T>::fromHost(size_t dstOffset, size_t bytes,
        const void* data) {
        if(dstOffset + bytes > sizeof(T) * m_size)
            return cudaErrorInvalidValue;
        cudaMemcpy(m_pDevice + dstOffset, data, bytes, cudaMemcpyHostToDevice);
        return cudaSuccess;
    }
    template<typename T>
    CudaDeviceMem<T>::~CudaDeviceMem() {
        //CUDA_DESTRUCTOR_MESSAGE(this);
        m_alloc->free(m_pDevice);
    }
};

#endif

