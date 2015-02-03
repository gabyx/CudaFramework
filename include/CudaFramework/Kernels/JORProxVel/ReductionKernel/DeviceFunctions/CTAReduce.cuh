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


#ifndef CudaFramework_Kernels_JORProxVel_ReductionKernel_DeviceFunctions_CTAReduce_cuh
#define CudaFramework_Kernels_JORProxVel_ReductionKernel_DeviceFunctions_CTAReduce_cuh


#include <cuda_runtime.h>
#include "DeviceUtilities/Intrinsics.cuh"


namespace ReductionGPU {





////////////////////////////////////////////////////////////////////////////////
// CTAReduce


// Count leading zeros - start from most significant bit.


template<int NT, typename Op = ReductionGPU::plus<int> >
struct CTAReduce {
    typedef typename Op::first_argument_type T;
    enum { Size = NT, Capacity = NT };
    struct Storage {
        T shared[Capacity];
    };

    __device__ static T Reduce(int tid, T x, Storage& storage, Op op = Op()) {
        storage.shared[tid] = x;
        __syncthreads();

        // Fold the data in half with each pass.
#pragma unroll
        for(int destCount = NT / 2; destCount >= 1; destCount /= 2) {
            if(tid < destCount) {
                // Read from the right half and store to the left half.
                x = op(x, storage.shared[destCount + tid]);
                storage.shared[tid] = x;
            }
            __syncthreads();
        }
        T total = storage.shared[0];
        __syncthreads();
        return total;
    }
};





////////////////////////////////////////////////////////////////////////////////
// deviceFindSegScanDelta
// Runs an inclusive max-index scan over binary inputs.

template<int NT>
__device__ int deviceFindSegScanDelta(int tid,
                                      bool flag,
                                      int* delta_shared) {
    const int NumWarps = NT / 32;

    int warp = tid / 32;
    int lane = 31 & tid;
    uint warpMask = 0xffffffff>> (31 - lane);		// inclusive search
    uint ctaMask = 0x7fffffff>> (31 - lane);		// exclusive search

    uint warpBits = __ballot(flag);
    delta_shared[warp] = warpBits;
    __syncthreads();

    if(tid < NumWarps) {
        uint ctaBits = __ballot(0 != delta_shared[tid]);
        int warpSegment = 31 - clz(ctaMask & ctaBits);
        int start = (-1 != warpSegment) ?
                    (31 - clz(delta_shared[warpSegment]) + 32 * warpSegment) : 0;
        delta_shared[NumWarps + tid] = start;
    }
    __syncthreads();

    // Find the closest flag to the left of this thread within the warp.
    // Include the flag for this thread.
    int start = 31 - clz(warpMask & warpBits);
    if(-1 != start) start += ~31 & tid;
    else start = delta_shared[NumWarps + warp];
    __syncthreads();

    return tid - start;
}

////////////////////////////////////////////////////////////////////////////////
// CTASegScan

template<int NT, typename _Op = ReductionGPU::plus<int> >
struct CTASegScan {
    typedef _Op Op;
    typedef typename Op::result_type T;
    enum { NumWarps = NT / 32, Size = NT, Capacity = 2 * NT };
    union Storage {
        int delta[NumWarps];
        T values[Capacity];
    };

    // Each thread passes the reduction of the LAST SEGMENT that it covers.
    // flag is set to true if there's at least one segment flag in the thread.
    // SegScan returns the reduction of values for the first segment in this
    // thread over the preceding threads.
    // Return the value init for the first thread.

    // When scanning single elements per thread, interpret the flag as a BEGIN
    // FLAG. If tid's flag is set, its value belongs to thread tid + 1, not
    // thread tid.

    // The function returns the reduction of the last segment in the CTA.

    __device__ static T SegScanDelta(int tid,
                                     int tidDelta,
                                     T x,
                                     Storage& storage,
                                     T* carryOut,
                                     T identity = (T)0,
                                     Op op = Op()) {

        // Run an inclusive scan
        int first = 0;
        storage.values[first + tid] = x;
        __syncthreads();

#pragma unroll
        for(int offset = 1; offset < NT; offset += offset) {
            if(tidDelta >= offset)
                x = op(storage.values[first + tid - offset], x);
            first = NT - first;
            storage.values[first + tid] = x;
            __syncthreads();
        }

        // Get the exclusive scan.
        x = tid ? storage.values[first + tid - 1] : identity;
        *carryOut = storage.values[first + NT - 1];
        __syncthreads();
        return x;
    }

    __device__ static T SegScan(int tid, T x, bool flag, Storage& storage,
                                T* carryOut, T identity = (T)0, Op op = Op()) {

        // Find the left-most thread that covers the first segment of this
        // thread.
        int tidDelta = deviceFindSegScanDelta<NT>(tid, flag, storage.delta);

        return SegScanDelta(tid, tidDelta, x, storage, carryOut, identity, op);
    }
};









struct SegReduceRange {
    int begin;
    int end;
    int total;
    bool flushLast;
};

__device__ SegReduceRange deviceShiftRange(int limit0,
                                           int limit1) {
    SegReduceRange range;
    range.begin = 0x7fffffff & limit0;
    range.end = 0x7fffffff & limit1;
    range.total = range.end - range.begin;
    range.flushLast = 0 == (0x80000000 & limit1);
    range.end += !range.flushLast;
    return range;
}

////////////////////////////////////////////////////////////////////////////////
// After loading CSR terms into shared memory, each thread binary searches
// (upper-bound) to find its starting point. Each thread then walks forward,
// emitting the csr0-relative row indices to register.

template<int NT, int VT>
__device__ int DeviceExpandCsrRows(int tidOffset,
                                   int* csr_shared,
                                   int numRows,
                                   int end,
                                   int rows[VT + 1],
                                   int rowStarts[VT]) {

    // Each thread binary searches for its starting row.
    int row = binarySearch<MgpuBoundsUpper>(csr_shared, numRows, tidOffset,
                                            ReductionGPU::less<int>()) - 1;

    // Each thread starts at row and scans forward, emitting row IDs into
    // register. Store the CTA-local row index (starts at 0) to rows and the
    // start of the row (globally) to rowStarts.
    int curOffset = csr_shared[row];
    int nextOffset = (row + 1 < numRows) ? csr_shared[row + 1] : end;

    rows[0] = row;
    rowStarts[0] = curOffset;
    int endFlags = 0;

#pragma unroll
    for(int i = 1; i <= VT; ++i) {
        // Advance the row cursor when the iterator hits the next row offset.
        if(tidOffset + i == nextOffset) {
            // Set an end flag when the cursor advances to the next row.
            endFlags |= 1<< (i - 1);

            // Advance the cursor and load the next row offset.
            ++row;
            curOffset = nextOffset;
            nextOffset = (row + 1 < numRows) ? csr_shared[row + 1] : end;
        }
        rows[i] = row;
        if(i < VT) rowStarts[i] = curOffset;
    }
    __syncthreads();

    return endFlags;
}

////////////////////////////////////////////////////////////////////////////////
// DeviceSegReducePrepare
// Expand non-empty interval of CSR elements into row indices. Compute end-flags
// by comparing adjacent row IDs.

// DeviceSegReducePrepare may be called either by a pre-processing kernel or by
// the kernel that actually evaluates the segmented reduction if no preprocesing
// is desired.

struct SegReduceTerms {
    int endFlags;
    int tidDelta;
};

template<int NT, int VT>
__device__ SegReduceTerms DeviceSegReducePrepare(int* csr_shared,
                                                 int numRows,
                                                 int tid,
                                                 int gid,
                                                 bool flushLast,
                                                 int rows[VT + 1],
                                                 int rowStarts[VT]) {

    // Pass a sentinel (end) to point to the next segment start. If we flush,
    // this is the end of this tile. Otherwise it is INT_MAX
    int endFlags = DeviceExpandCsrRows<NT, VT>(gid + VT * tid, csr_shared,
                   numRows, flushLast ? (gid + NT * VT) : INT_MAX, rows, rowStarts);

    // Find the distance to to scan to compute carry-in for each thread. Use the
    // existance of an end flag anywhere in the thread to determine if carry-out
    // values from the left should propagate through to the right.
    int tidDelta = deviceFindSegScanDelta<NT>(tid, rows[0] != rows[VT],
                   csr_shared);

    SegReduceTerms terms = { endFlags, tidDelta };
    return terms;
}



////////////////////////////////////////////////////////////////////////////////
// CTASegReduce
// Core segmented reduction code. Supports fast-path and slow-path for intra-CTA
// segmented reduction. Stores partials to global memory.
// Callers feed CTASegReduce::reduceToGlobal values in thread order.

template<int NT, int VT, bool HalfCapacity, typename T, typename Op>
struct CTASegReduce {
    typedef CTASegScan<NT, Op> SegScan;

    enum {
        NV = NT * VT,
        Capacity = HalfCapacity ? (NV / 2) : NV
    };

    union Storage {
        typename SegScan::Storage segScanStorage;
        T values[Capacity];
    };

    template<typename DestIt,typename PREC>
    __device__ static void reduceToGlobal(  const int rows[VT + 1],
                                            int total,
                                            int tidDelta,
                                            int startRow,
                                            int block,
                                            int tid,
                                            PREC data[VT],
                                            DestIt destGlobal,
                                            PREC* carryOut_global,
                                            PREC identity,
                                            Op op,
                                            Storage& storage) {

        // Run a segmented scan within the thread.
        T x, localScan[VT];
#pragma unroll
        for(int i = 0; i < VT; ++i) {
            x = i ? op(x, data[i]) : data[i];
            localScan[i] = x;
            if(rows[i] != rows[i + 1]) x = identity;
        }

        // Run a parallel segmented scan over the carry-out values to compute
        // carry-in.
        T carryOut;
        T carryIn = SegScan::SegScanDelta(tid,
                                          tidDelta,
                                          x,
                                          storage.segScanStorage,
                                          &carryOut,
                                          identity,
                                          op);

        // Store the carry-out for the entire CTA to global memory.
        if(!tid) carryOut_global[block] = carryOut;

        destGlobal += startRow;
        if(HalfCapacity && total > Capacity) {
            // Add carry-in to each thread-local scan value. Store directly
            // to global.
#pragma unroll
            for(int i = 0; i < VT; ++i) {
                // Add the carry-in to the local scan.
                T x2 = op(carryIn, localScan[i]);

                // Store on the end flag and clear the carry-in.
                if(rows[i] != rows[i + 1]) {
                    carryIn = identity;
                    destGlobal[rows[i]] = x2;
                }
            }
        } else {
            // All partials fit in shared memory. Add carry-in to each thread-
            // local scan value.
#pragma unroll
            for(int i = 0; i < VT; ++i) {
                // Add the carry-in to the local scan.
                T x2 = op(carryIn, localScan[i]);

                // Store reduction when the segment changes and clear the
                // carry-in.
                if(rows[i] != rows[i + 1]) {
                    storage.values[rows[i]] = x2;
                    carryIn = identity;
                }
            }
            __syncthreads();

            // Cooperatively store reductions to global memory.
            for(int index = tid; index < total; index += NT)
                destGlobal[index] = storage.values[index];
            __syncthreads();
        }
    }
};



} // namespace ReductionGPU
#endif // CTAReduce_cuh
