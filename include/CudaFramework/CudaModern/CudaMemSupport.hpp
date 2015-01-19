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

#ifndef CudaFramework_CudaModern_CudaMemSupport_hpp
#define CudaFramework_CudaModern_CudaMemSupport_hpp

#include <cassert>
#include <algorithm>
#include <random>

#include "CudaFramework/General/AssertionDebug.hpp"
#include "CudaFramework/CudaModern/CudaTypeDefs.hpp"
#include "CudaFramework/CudaModern/CudaError.hpp"
#include "CudaFramework/CudaModern/CudaRefcounting.hpp"
#include "CudaFramework/CudaModern/CudaDeviceMemory.hpp"
#include "CudaFramework/CudaModern/CudaDevice.hpp"

namespace utilCuda{

    class CudaContext;

    class CudaMemSupport {
        friend class CudaDevice;
        friend class CudaContext;
    public:
        CudaDevice& device() { return m_alloc->device(); }

        // Swap out the associated allocator.
        void setAllocator(CudaAlloc* alloc) {
            assert(alloc->device().ordinal() == m_alloc->device().ordinal());
            m_alloc.reset(alloc);
        }

        // Access the associated allocator.
        CudaAlloc* getAllocator() { return m_alloc.get(); }

        // Support for creating arrays.
        template<typename T>
        DeviceMemPtr<T> malloc(size_t count);

        template<typename T>
        DeviceMemPtr<T> malloc(const T* data, size_t count);

        template<typename T>
        DeviceMemPtr<T> malloc(const std::vector<T>& data);

        template<typename T>
        DeviceMemPtr<T> fill(size_t count, T fill);

        template<typename T>
        DeviceMemPtr<T> fillAscending(size_t count, T first, T step);

        template<typename T>
        DeviceMemPtr<T> genRandom(size_t count, T min, T max);

        template<typename T>
        DeviceMemPtr<T> sortRandom(size_t count, T min, T max);

        template<typename T, typename Func>
        DeviceMemPtr<T> genFunc(size_t count, Func f);

    protected:
        CudaMemSupport() { }
        AllocPtrType m_alloc;
    };



////////////////////////////////////////////////////////////////////////////////
// CudaMemSupport method implementations

template<typename T>
DeviceMemPtr<T> CudaMemSupport::malloc(size_t count) {

    ASSERTMSG(count > 0, "Count needs to be greater then zero");

	DeviceMemPtr<T> mem(new CudaDeviceMem<T>(m_alloc.get(),count));
	ASSERTCHECK_CUDA(m_alloc->malloc(sizeof(T) * count, (void**)&mem->m_pDevice));

#ifndef NDEBUG
	// Initialize the memory to -1 in debug mode.
	ASSERTCHECK_CUDA(cudaMemset(mem->get(), -1, count));
#endif

	return mem;
}

template<typename T>
DeviceMemPtr<T> CudaMemSupport::malloc(const T* data, size_t count) {
	DeviceMemPtr<T> mem = malloc<T>(count);
	ASSERTCHECK_CUDA(mem->fromHost(data, count));
	return mem;
}

template<typename T>
DeviceMemPtr<T> CudaMemSupport::malloc(const std::vector<T>& data) {
	DeviceMemPtr<T> mem = malloc<T>(data.size());
	if(data.size()) ASSERTCHECK_CUDA(mem->fromHost(&data[0], data.size()));
	return mem;
}

template<typename T>
DeviceMemPtr<T> CudaMemSupport::fill(size_t count, T fill) {
	std::vector<T> data(count, fill);
	return malloc(data);
}

template<typename T>
DeviceMemPtr<T> CudaMemSupport::fillAscending(size_t count, T first, T step) {
	std::vector<T> data(count);
	for(size_t i = 0; i < count; ++i)
		data[i] = first + i * step;
	return malloc(data);
}

template<typename T>
DeviceMemPtr<T> CudaMemSupport::genRandom(size_t count, T min, T max) {
	std::vector<T> data(count);

	RandomGenerator generator;
    std::uniform_int_distribution<T> distribution(min,max);

	for(size_t i = 0; i < count; ++i)
		data[i] = distribution(generator);

	return malloc(data);
}

template<typename T>
DeviceMemPtr<T> CudaMemSupport::sortRandom(size_t count, T min, T max) {
	std::vector<T> data(count);

	RandomGenerator generator;
    std::uniform_int_distribution<T> distribution(min,max);

	for(size_t i = 0; i < count; ++i)
		data[i] = distribution(generator);

	std::sort(data.begin(), data.end());
	return malloc(data);
}

template<typename T, typename Func>
DeviceMemPtr<T> CudaMemSupport::genFunc(size_t count, Func f) {
	std::vector<T> data(count);
	for(size_t i = 0; i < count; ++i)
		data[i] = f(i);

	return malloc<T>(data);
}



};
#endif
