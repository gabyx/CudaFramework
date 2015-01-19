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

#ifndef CudaFramework_CudaModern_CudaAlloc_hpp
#define CudaFramework_CudaModern_CudaAlloc_hpp

#include <list>
#include <map>

#include <cuda.h>
#include <driver_types.h>

#include "CudaFramework/CudaModern/CudaError.hpp"

#include "CudaFramework/CudaModern/CudaRefcounting.hpp"

////////////////////////////////////////////////////////////////////////////////
// CudaMemSupport
// Convenience functions for allocating device memory and copying to it from
// the host. These functions are factored into their own class for clarity.
// The class is derived by CudaContext.
namespace utilCuda{

    class CudaDevice;
    class CudaContext;

////////////////////////////////////////////////////////////////////////////////
// Customizable allocator.

// CudaAlloc is the interface class all allocator accesses. Users may derive
// this, implement custom allocators, and set it to the device with
// CudaDevice::setAllocator.

    class CudaAlloc : public ReferenceCounting<void,CudaAlloc> {
        public:
            virtual cudaError_t malloc(size_t size, void** p) = 0;
            virtual cudaError_t mallocPitch(void** p, size_t *pitch, size_t width, size_t height) = 0;
            virtual bool free(void* p) = 0;
            virtual void clear() = 0;

            virtual ~CudaAlloc() { }

            CudaDevice& device() { return m_device; }

        protected:
            CudaAlloc(CudaDevice& device) : m_device(device) { }
            CudaDevice& m_device;
    };

    // A concrete class allocator that simply calls cudaMalloc and cudaFree.
    class CudaAllocSimple : public CudaAlloc {
    public:
        CudaAllocSimple(CudaDevice& device) : CudaAlloc(device) { }

        virtual cudaError_t malloc(size_t size, void** p);
        virtual cudaError_t mallocPitch(void** p, size_t *pitch, size_t width, size_t height);
        virtual bool free(void* p);
        virtual void clear() { }
        virtual ~CudaAllocSimple() {
            CUDA_DESTRUCTOR_MESSAGE(this);
        }
    };

    // A concrete class allocator that uses exponentially-spaced buckets and an LRU
    // (Least recently used) http://de.wikipedia.org/wiki/Least_recently_used
    // to reuse allocations. This is the default allocator. It is shared between
    // all contexts on the device.
    // The capacity which is set from outside (at the beginning) is never automatically increased!
    class CudaAllocBuckets : public CudaAlloc {
    public:
        CudaAllocBuckets(CudaDevice& device);
        virtual ~CudaAllocBuckets();

        virtual cudaError_t malloc(size_t size, void** p);
        virtual cudaError_t mallocPitch(void** p, size_t *pitch, size_t width, size_t height);
        virtual bool free(void* p);
        virtual void clear();

        size_t allocated() const { return m_allocated; }
        size_t committed() const { return m_committed; }
        size_t capacity() const { return m_capacity; }

        bool sanityCheck() const;

        // Sets the capacity and compacts the buffer
        void setCapacity(size_t capacity, size_t maxObjectSize) {
            m_capacity = capacity;
            m_maxObjectSize = maxObjectSize;
            clear();
        }

    private:
        static const int NumBuckets = 84;
        static const size_t BucketSizes[NumBuckets];

        struct MemNode;
        typedef std::list<MemNode> MemList;
        typedef std::map<void*, MemList::iterator> AddressMap;
        typedef std::multimap<int, MemList::iterator> PriorityMap;

        struct MemNode {
            AddressMap::iterator address;
            PriorityMap::iterator priority;
            int bucket;
        };

        void compact(size_t extra);
        void freeNode(MemList::iterator memIt);
        int locateBucket(size_t size) const;

        AddressMap m_addressMap;             ///< Address to MemNode
        PriorityMap m_priorityFreeMem;       ///< Priority to a MemNode (only the CACHED MemNodes which are not used!)
        MemList m_memLists[NumBuckets + 1];  ///< For each bucket size a memory list of MemNode's, the uncached bucket [NumBuckets] is the special bucket for all uncached memNodes!

        // The UNCACHED bucket [NumBuckets] does not belong to the buffer!
        // m_allocated counts all allocated bytes of CACHED memNodes in the buffer
        // m_committed counts all used/commited bytes of CACHED memNodes in the buffer (cant free these!)
        size_t m_maxObjectSize, m_capacity, m_allocated, m_committed;

        int m_counter;
    };


};

#endif
