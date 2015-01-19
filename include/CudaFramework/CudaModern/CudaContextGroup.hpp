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

#ifndef CudaFramework_CudaModern_CudaContextGroup_hpp
#define CudaFramework_CudaModern_CudaContextGroup_hpp

#include <memory>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>

#include "CudaFramework/CudaModern/CudaTypeDefs.hpp"
#include "CudaFramework/CudaModern/CudaDevice.hpp"
#include "CudaFramework/CudaModern/CudaContext.hpp"


namespace utilCuda{


////////////////////////////////////////////////////////////////////////////////
// CudaContext

struct ContextGroup {

	std::vector< ContextPtrType > m_standardContexts;
	unsigned int m_numDevices;

	ContextGroup(){
		m_numDevices = CudaDevice::deviceCount();
		m_standardContexts.resize(m_numDevices, ContextPtrType(nullptr) );
	}

	ContextPtrType getByOrdinal(unsigned int ordinal) {
		if( !(m_standardContexts[ordinal].get()) ) {
			CudaDevice& device = CudaDevice::byOrdinal(ordinal);
			m_standardContexts[ordinal].reset(new CudaContext(device, false, true));
		}
		return m_standardContexts[ordinal];
	}

	~ContextGroup() {
		// Destruction of std::vector destructs all IntrusivePtr -> destroying all standard contextes.
        std::cout << "destroying all standard contexts" << std::endl;
	}
};

//extern std::unique_ptr<ContextGroup> contextGroup;

};



#endif

