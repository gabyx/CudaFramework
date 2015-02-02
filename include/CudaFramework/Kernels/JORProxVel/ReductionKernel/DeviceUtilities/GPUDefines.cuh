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
# ifndef GPUDefines_cuh
# define GPUDefines_cuh

namespace ReductionGPU {
#ifndef MGPU_MIN
#define MGPU_MIN(x, y) (((x) <= (y)) ? (x) : (y))
#define MGPU_MAX(x, y) (((x) >= (y)) ? (x) : (y))
#define MGPU_MAX0(x) (((x) >= 0) ? (x) : 0)
#define MGPU_ABS(x) (((x) >= 0) ? (x) : (-x))

#define MGPU_DIV_UP(x, y) (((x) + (y) - 1) / (y))
#define MGPU_DIV_ROUND(x, y) (((x) + (y) / 2) / (y))
#define MGPU_ROUND_UP(x, y) ((y) * MGPU_DIV_UP(x, y))
#define MGPU_SHIFT_DIV_U(x, y) (((x) + ((1<< (y)) - 1))>> y)
#define MGPU_ROUND_UP_POW2(x, y) (((x) + (y) - 1) & ~((y) - 1))
#define MGPU_ROUND_DOWN_POW2(x, y) ((x) & ~((y) - 1))
#define MGPU_IS_POW_2(x) (0 == ((x) & ((x) - 1)))

#endif // MGPU_MIN

#ifdef _DEBUG
#define MGPU_SYNC_CHECK(s) {												\
	cudaError_t error = cudaDeviceSynchronize();							\
	if(cudaSuccess != error) {												\
		printf("CUDA ERROR %d %s\n%s:%d.\n%s\n",							\
			error, cudaGetErrorString(error), __FILE__, __LINE__, s);		\
		exit(0);															\
	}																		\
}
#else
#define MGPU_SYNC_CHECK(s)
#endif
}

#endif
