
/******************************************************************************
 * Copyright (c) 2013, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
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
 ******************************************************************************/

/******************************************************************************
 *
 * Code and text by Sean Baxter, NVIDIA Research
 * See http://nvlabs.github.io/moderngpu for repository and documentation.
 *
 ******************************************************************************/

#ifndef CudaFramework_Kernels_JORProxVel_ReductionKernel_HostUtilities_TuningFunctions_hpp
#define CudaFramework_Kernels_JORProxVel_ReductionKernel_HostUtilities_TuningFunctions_hpp

#include <cuda_runtime.h>



// Returns (NT, VT) from the sm version.


// HPP (C++ Seite)
namespace ReductionGPU{


    template<typename Derived>
	static int2 GetLaunchParams(int sm) {
		if(sm >= 350)
			return make_int2(Derived::Sm35::NT, Derived::Sm35::VT);
		else if(sm >= 300)
			return make_int2(Derived::Sm30::NT, Derived::Sm30::VT);
		else
			return make_int2(Derived::Sm20::NT, Derived::Sm20::VT);
	}

    template<typename Derived>
    static int2 GetLaunchParams(utilCuda::CudaContext& context) {
		return GetLaunchParams<Derived>(context.getPTXVersion());
	}


}
  # endif
