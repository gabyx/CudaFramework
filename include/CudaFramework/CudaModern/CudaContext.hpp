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

#ifndef CudaFramework_CudaModern_CudaContext_hpp
#define CudaFramework_CudaModern_CudaContext_hpp

#include <cuda.h>
#include <cuda_runtime.h>

#include "CudaFramework/CudaModern/CudaTypeDefs.hpp"
#include "CudaFramework/CudaModern/CudaRefcounting.hpp"
#include "CudaFramework/CudaModern/CudaMatrixSupport.hpp"

#include "CudaFramework/CudaModern/CudaTimer.hpp"
#include "CudaFramework/CudaModern/CudaEvent.hpp"

namespace utilCuda{


struct ContextGroup;

// Create a context on the default stream (0) on device with idx ordinal.
ContextPtrType createCudaContextOnDevice( int ordinal, bool standardAllocator = false );

// Create a context on a new stream on device with idx ordinal
ContextPtrType createCudaContextNewStreamOnDevice(int ordinal, bool standardAllocator = false );

// Create a context and attach stream on device with idx ordinal
ContextPtrType createCudaContextAttachStreamOnDevice( int ordinal, cudaStream_t stream, bool standardAllocator=false);


class CudaContext : public CudaMatrixSupport, public ReferenceCounting<void,CudaContext> {
	friend struct ContextGroup;

	friend ContextPtrType createCudaContextOnDevice(int ordinal,bool standardAllocator);
	friend ContextPtrType createCudaContextNewStreamOnDevice(int ordinal,bool standardAllocator);
	friend ContextPtrType createCudaContextAttachStreamOnDevice(int ordinal, cudaStream_t stream,bool standardAllocator);

public:

    ~CudaContext();

	// Get the pointer to a standart context ( all standart contexts are in struct CudaContextGroup)
	static ContextPtrType StandardContext(int ordinal = -1);

	// 4KB of page-locked memory per context.
	char* pageLocked() { return m_pageLocked; }
    cudaStream_t auxStream() const { return m_auxStream; }

	int numSMs() { return device().numSMs(); }
	int archVersion() { return device().archVersion(); }
	int getPTXVersion() { return device().getPTXVersion(); }
	std::string deviceString() { return device().deviceString(); }

	cudaStream_t stream() const { return m_stream; }

	// Set this device as the active device on the thread.
	void setActive() { device().setActive(); }

	// Access the included event.
	CudaEvent& event() { return m_event; }

	// Use the included timer.
	CudaTimer& timer() { return m_timer; }
	void start() { m_timer.start(); }
	double split() { return m_timer.split(); }
	double throughput(int count, int numIterations) {
		return m_timer.throughput(count, numIterations);
	}

private:
	CudaContext(CudaDevice& device, bool newStream, bool standardAllocator);


	AllocPtrType createDefaultAlloc(CudaDevice& device);

	bool m_ownStream;
	cudaStream_t m_stream;
	cudaStream_t m_auxStream;
	CudaEvent m_event;
	CudaTimer m_timer;

	char* m_pageLocked;
};

};
#endif
