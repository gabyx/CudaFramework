// ========================================================================================
//  CudaFramework
//  Copyright (C) 2014 by Gabriel Nützi <nuetzig (at) imes (d0t) mavt (d0t) ethz (d0t) ch>
//
//  This Source Code Form is subject to the terms of the GNU GPL 3.0 licence. 
//  If a copy of the GNU GPL 3.0 was not distributed with this
//  file, you can obtain one at http://opensource.org/licenses/GPL-3.0.
// ========================================================================================

#include "CudaFramework/CudaModern/CudaContext.hpp"

#include "CudaFramework/CudaModern/CudaError.hpp"

#include "CudaFramework/CudaModern/CudaContextGroup.hpp"
#include "CudaFramework/CudaModern/CudaAlloc.hpp"

namespace utilCuda{


CudaContext::CudaContext(CudaDevice& device, bool newStream, bool standardAllocator) :
	m_event(cudaEventDisableTiming /*| cudaEventBlockingSync */),
	m_stream(0), m_pageLocked(0) {

	// Create an allocator.
	if(standardAllocator)
		m_alloc.reset(new CudaAllocSimple(device));
	else
		m_alloc = createDefaultAlloc(device);

    CHECK_CUDA_LAST
	if(newStream) ASSERTCHECK_CUDA(cudaStreamCreate(&m_stream));
	m_ownStream = newStream;

	// Allocate 4KB of page-locked memory.
	ASSERTCHECK_CUDA(cudaMallocHost(&m_pageLocked, 4096));

	// Allocate an auxiliary stream.
	ASSERTCHECK_CUDA(cudaStreamCreate(&m_auxStream));
}

CudaContext::~CudaContext() {

    CUDA_DESTRUCTOR_MESSAGE(this);

	if(m_pageLocked){
		ASSERTCHECK_CUDA(cudaFreeHost(m_pageLocked));
	}

	if(m_ownStream){
		ASSERTCHECK_CUDA(cudaStreamDestroy(m_stream));
	}

	ASSERTCHECK_CUDA(cudaStreamDestroy(m_auxStream));
}

AllocPtrType CudaContext::createDefaultAlloc(CudaDevice& device) {
	CudaAllocBuckets * alloc(new CudaAllocBuckets(device));
	size_t freeMem, totalMem;

    cudaError_t err = cudaMemGetInfo(&freeMem, &totalMem);
	CHECK_CUDA_M(err, "ERROR RETRIEVING MEM INFO FOR CUDA DEVICE" << device.ordinal() );

    // get max n such that (n*128mb < freeMem-reserve)
//    unsigned int reserve = 128 << 20;
//    unsigned int n = (freeMem-reserve) / (128<<20);
//    ASSERTMSG((n*128)<<20 <= freeMem-reserve,"Wrong capacity choosed");

	// Maintain a buffer of n*128MB with max objects of 512MB.
	alloc->setCapacity( (1024)<< 20, (512)<< 20);

	return AllocPtrType(alloc);
}


// Does activate the context!
ContextPtrType createCudaContextOnDevice(int ordinal,bool standardAllocator) {

	CudaDevice& device = CudaDevice::byOrdinal(ordinal);
	ContextPtrType context(new CudaContext(device, false, standardAllocator));

	context->setActive();

	return context;
}

// Does NOT activate the context! Need to be done seperately
ContextPtrType createCudaContextNewStreamOnDevice( int ordinal, bool standardAllocator) {
    CudaDevice& device = CudaDevice::byOrdinal(ordinal);
	ContextPtrType context(new CudaContext( device, true, standardAllocator));
	return context;
}


// Does NOT activate the context! Need to be done seperately
ContextPtrType createCudaContextAttachStreamOnDevice( int ordinal, cudaStream_t stream, bool standardAllocator) {
	ContextPtrType context(new CudaContext( CudaDevice::byOrdinal(ordinal), false, standardAllocator));
	context->m_stream = stream;
	return context;
}

};
