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

#ifndef CudaFramework_CudaModern_CudaDevice_hpp
#define CudaFramework_CudaModern_CudaDevice_hpp

#include <cstdarg>
#include <string>
#include <memory>
#include <cuda.h>
#include <cuda_runtime.h>

#include "CudaFramework/CudaModern/CudaTypeDefs.hpp"
#include "CudaFramework/CudaModern/CudaError.hpp"

namespace utilCuda{





// This static global bool is to determine if destroyDeviceGroup() has been used.
// The user of this library need to call destroyDeviceGroup() in main()
// such that all devices are destroyed befor the CUDA runtime destroys it self which
// leads to cuda_Error_t "unload Cuda runtime error" if this is not called!
void destroyDeviceGroup();


struct DeviceGroup;

class CudaDevice : public NonCopyable {
	friend struct DeviceGroup;
public:
	static int deviceCount();
	static CudaDevice& byOrdinal(int ordinal);
	static CudaDevice& selected();

	// Device properties.
	const cudaDeviceProp& prop() const { return m_prop; }
	int ordinal() const { return m_ordinal; }
	int numSMs() const { return m_prop.multiProcessorCount; }
	int archVersion() const { return 100 * m_prop.major + 10 * m_prop.minor; }

	// LaunchBox properties.
	int getPTXVersion() const { return m_ptxVersion; }

	std::string deviceString()  const;
    std::string memInfoString() const;

	// Set this device as the active device on the thread.
	void setActive();

    bool hasFreeMemory() const;

private:
	CudaDevice() { }		// hide the constructor.
	int m_ordinal;
	int m_ptxVersion;
	cudaDeviceProp m_prop;
};





};

#endif
