// ========================================================================================
//  CudaFramework
//  Copyright (C) 2014 by Gabriel Nützi <nuetzig (at) imes (d0t) mavt (d0t) ethz (d0t) ch>
//
//  This Source Code Form is subject to the terms of the GNU GPL 3.0 licence. 
//  If a copy of the GNU GPL 3.0 was not distributed with this
//  file, you can obtain one at http://opensource.org/licenses/GPL-3.0.
// ========================================================================================

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
#ifndef CudaFramework_Kernels_JORProxVel_ReductionKernel_HostUtilities_Tuning_hpp
#define CudaFramework_Kernels_JORProxVel_ReductionKernel_HostUtilities_Tuning_hpp

#include <cuda_runtime.h>


#if __CUDA_ARCH__ >= 350
	#define MGPU_SM_TAG Sm35
#elif __CUDA_ARCH__ >= 300
	#define MGPU_SM_TAG Sm30
#elif __CUDA_ARCH__ >= 200
	#define MGPU_SM_TAG Sm20
#else
	#define MGPU_SM_TAG Sm20
#endif

#define MGPU_LAUNCH_PARAMS typename Tuning::MGPU_SM_TAG
#define MGPU_LAUNCH_BOUNDS __global__ \
	__launch_bounds__(Tuning::MGPU_SM_TAG::NT, Tuning::MGPU_SM_TAG::OCC)



template<int NT_, int VT_, int OCC_, bool HalfCapacity_, bool LdgTranspose_>
struct SegReduceTuning {
	enum {
		NT = NT_,
		VT = VT_,
		OCC = OCC_,
		HalfCapacity = HalfCapacity_,
		LdgTranspose = LdgTranspose_
	};
};


template<typename Sm20_, typename Sm30_ = Sm20_, typename Sm35_ = Sm30_>
struct LaunchBox {
	typedef Sm20_ Sm20;
	typedef Sm30_ Sm30;
	typedef Sm35_ Sm35;
};

template<size_t size>
struct SegReduceNormalTuning {
	typedef LaunchBox<
		SegReduceTuning<128, 11, 0, false, false>, // DEBUG(GABRIEL) orig: SegReduceTuning<128, 11, 0, false, false>,
		SegReduceTuning<128, 7, 0, true, false>, // DEBUG(GABRIEL) orig: SegReduceTuning<128, 7, 0, true, false>,
		SegReduceTuning<128, (size > sizeof(int)) ? 11 : 7, 0, true, true> // DEBUG(GABRIEL) orig: SegReduceTuning<128, (size > sizeof(int)) ? 11 : 7, 0, true, true>,
		> Tuning;
};

// SegReduceCSR - Preprocess


template<size_t size>
struct SegReducePreprocessTuning {
	typedef LaunchBox<
		SegReduceTuning<128, 11, 0, false, false>,
		SegReduceTuning<128, 11, 0, true, false>,
		SegReduceTuning<128, 11, 0, true, (size > 4) ? false : true>
	> Tuning;
};

// SegReduceCSR - Indirect
template<size_t size>
struct SegReduceIndirectTuning {
	typedef LaunchBox<
		SegReduceTuning<128, 11, 0, false, false>,
		SegReduceTuning<128, 7, 0, true, false>,
		SegReduceTuning<128, 7, 0, true, true>
	> Tuning;
};


# endif
