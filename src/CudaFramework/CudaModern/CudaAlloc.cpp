

#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include "CudaFramework/General/AssertionDebug.hpp"

#include "CudaFramework/CudaModern/CudaError.hpp"
#include "CudaFramework/CudaModern/CudaDevice.hpp"
#include "CudaFramework/CudaModern/CudaAlloc.hpp"


namespace utilCuda{

////////////////////////////////////////////////////////////////////////////////
// CudaAllocSimple

cudaError_t CudaAllocSimple::mallocPitch(void** p, size_t *pitch, size_t widthInBytes, size_t height){

    cudaError_t error = cudaSuccess;
	*p = 0;
	if(widthInBytes*height){
        error = cudaMallocPitch(p, pitch,widthInBytes,height);
	}
	return error;

}

cudaError_t CudaAllocSimple::malloc(size_t size, void** p) {
	cudaError_t error = cudaSuccess;
	*p = 0;
	if(size){
        error = cudaMalloc(p, size);
	}
	return error;
}

bool CudaAllocSimple::free(void* p) {
	cudaError_t error = cudaSuccess;
	if(p) error = cudaFree(p);
	return cudaSuccess == error;
}

////////////////////////////////////////////////////////////////////////////////
// CudaAllocBuckets

/**
  _   ============================== 0
  |
  |       Used Memory (committed/used)
  |       not in priority queue
  |
  |   ============================== m_commited
  |
  |          Priority Queue
  |          (uncomitted/unused)
  |          This space gets freed
  |          in function compact();
  |
  |
  |   ==========                    m_allocated < m_capacity
 size
             Uncached (used)
              (not counted)
      ==============================

*/

CudaAllocBuckets::CudaAllocBuckets(CudaDevice& device) : CudaAlloc(device) {
	m_maxObjectSize = m_capacity = m_allocated = m_committed = 0;
	m_counter = 0;
}

CudaAllocBuckets::~CudaAllocBuckets() {
    CUDA_DESTRUCTOR_MESSAGE(this);
	setCapacity(0, 0);
	assert(!m_allocated);
}

bool CudaAllocBuckets::sanityCheck() const {
	// Iterate through all allocated objects and verify sizes.
	size_t allocatedCount = 0, committedCount = 0;
	for(AddressMap::const_iterator i = m_addressMap.begin();
		i != m_addressMap.end(); ++i) {

		int bucket = i->second->bucket;
		size_t size = (bucket < NumBuckets) ? BucketSizes[bucket] : 0;
		allocatedCount += size;

		if(i->second->priority == m_priorityFreeMem.end())
			committedCount += size;
	}

	return allocatedCount == m_allocated && committedCount == m_committed;
}

cudaError_t CudaAllocBuckets::mallocPitch(void** p, size_t *pitch, size_t widthInBytes, size_t height){

    // Calculate the proper pitch and hand to normal malloc function which might choose a bucket which is already
    // allocated to 256 bytes boundary (sepcification of cudaMallo) or calls cudaMalloc which allocates then the self-padded
    // Memory
    // We cant use cudaMallocPitch because this allocates already memory and there is no function so far
    // to get the pitch only.

    // use textureAlignment (found on the internet hmmm... this seems wrong)
    //size_t align = m_device.prop().textureAlignment;
    //*pitch = ((widthInBytes + align - 1)/align) * align;

    // use Cuda Programming Guide Alignmenet (which should be the best i think)
    // Upper closest multible of 32/64/128
    //size_t upperMultOf32 = ((widthInBytes + 32 - 1)/32)*32;   //  widthInBytes/32 + 1
    *pitch = std::min(
    					std::min( ((widthInBytes + 32 - 1)>>5)<<5 , ((widthInBytes + 64 - 1)>>6)<<6 ),
    				    ((widthInBytes + 128 - 1)>>7)<<7
    				);

    // Still try cudaMallocPitch and call with nullptr to see what happens!
    // sadly it does not work ....
    // Maybe this is
    //    CHECK_CUDA(cudaMallocPitch(nullptr, pitch,widthInBytes,height))

    int allocSize = *pitch * height ;
    return malloc(allocSize, p );
}


cudaError_t CudaAllocBuckets::malloc(size_t size, void** p) {

	// Locate the bucket index and adjust the size of the allocation to the
	// bucket size.
	size_t allocSize = size;
	size_t commitSize = 0;
	int bucket = locateBucket(size);

	// if the fitted bucketIdx is a feasible bucket, adjust allocSize and commitSize
	// if uncached bucket bucketIdx=NumBuckets => commitSize = 0;
	if(bucket < NumBuckets)
		allocSize = commitSize = BucketSizes[bucket];


	// Peel off an already-allocated node and reuse it. ===========================================
	// We always take the front memNode (if it is not used = not in priorityMap)
	// the memNode list is organized in the fashion that the free nodes are at the beginngin, the used at the end!

	MemList& list = m_memLists[bucket]; // get the memory list for this bucketidx (uncached bucket bucketIdx=NumBuckets )!

	// if list is not empty and the priority of the front memNode  is not used (in the priority map)
	if(list.size() && list.front().priority != m_priorityFreeMem.end()) {

		MemList::iterator memIt = list.begin(); // take front memNode

        // Make memNode used!! (such that it cannot get deleted (removed from priority map!!)
		m_priorityFreeMem.erase(memIt->priority);
		memIt->priority = m_priorityFreeMem.end(); // invalidate iterator

        //
		list.splice(list.end(), list, memIt); // positionate used memNode always the end of the list
		m_committed += commitSize;            // overall used (commited) size of all buckets increases

		*p = memIt->address->first; // set address of pointer

		// memory has been found in bucket bucketNr with size commitSize
		return cudaSuccess;
	}
    // ======================================================================================

    // No memNode found to reuses, allocate some new node for this bucket
    // (bucketIdx or special size bucketIdx = NumBuckets)

	// Shrink if this allocation would put us over the limit.
	// Does not do anything for uncached data, for cached data compact such that commitSize + m_alloc < m_capacity (if possible)
	compact(commitSize);

	cudaError_t error = cudaSuccess;
	*p = nullptr;
	if(size) error = cudaMalloc(p, allocSize);

    //reduces the capacity about 10% of the buffer, continouously, do not know for what this loop might be any good!
    // but is works
	while((cudaErrorMemoryAllocation == error) && (m_committed < m_allocated)) {
		setCapacity(m_capacity - m_capacity / 10, m_maxObjectSize);
		error = cudaMalloc(p, size);
	}
	if(cudaSuccess != error) return error;

    // Insert a new memNode in the bucket (bucketIdx or special size bucketIdx = NumBuckets)
    auto & memList = m_memLists[bucket];
	MemList::iterator memIt = memList.insert(memList.end(), MemNode());
	memIt->bucket = bucket;
	memIt->address = m_addressMap.insert(std::make_pair(*p, memIt)).first;
	memIt->priority = m_priorityFreeMem.end(); // make used
	m_allocated += commitSize; // add zero for uncached node
	m_committed += commitSize; // add zero for uncached node

	assert(sanityCheck());

	return cudaSuccess;
}



bool CudaAllocBuckets::free(void* p) {
	AddressMap::iterator it = m_addressMap.find(p);
	if(it == m_addressMap.end()) {
		// If the pointer was not found in the address map, cudaFree it anyways
		// but return false.
		if(p) cudaFree(p);
		return false;
	}

	// Because we're freeing a page, it had better not be in the priority queue.
	MemList::iterator memIt = it->second;
	assert(memIt->priority == m_priorityFreeMem.end()); // assert that it is marked as used!

	// Always free allocations larger than the largest bucket
	it->second->priority = m_priorityFreeMem.insert(
		std::make_pair(m_counter++ - memIt->bucket, memIt));

	// freed nodes are moved to the front, committed nodes are moved to the
	// end.
	int bucket = memIt->bucket;
	size_t commitSize = (bucket < NumBuckets) ? BucketSizes[bucket] : 0;

	MemList& list = m_memLists[bucket];
	list.splice(list.begin(), list, memIt);
	m_committed -= commitSize;

	// Delete data that's not cached.
	if(NumBuckets == bucket)
		freeNode(memIt);

	compact(0);
	return true;
}

void CudaAllocBuckets::clear() {
	compact(m_allocated);
}

void CudaAllocBuckets::freeNode(CudaAllocBuckets::MemList::iterator memIt) {
    // Routine to free the node!

	if(memIt->address->first) cudaFree(memIt->address->first);

	int bucket = memIt->bucket;

	size_t commitSize = (bucket < NumBuckets) ? BucketSizes[bucket] : 0;

	m_addressMap.erase(memIt->address); // erase in address map

	if(memIt->priority != m_priorityFreeMem.end())
		m_priorityFreeMem.erase(memIt->priority);
	else
        // memIt->priority at end = commited
		m_committed -= commitSize;

	m_allocated -= commitSize;

    // erase memNode in its memNode list of bucket
	m_memLists[bucket].erase(memIt);

	assert(sanityCheck());
}

void CudaAllocBuckets::compact(size_t extra) {

    // m_allocated > m_committed => m_priorityFreeMem is not empty
    // and we still have some cached nodes we can free!

	while(m_allocated + extra > m_capacity && m_allocated > m_committed) {
		// Walk the priority queue from beginning to end removing nodes.
		// Removing nodes with lowest priority first
		// m_allocated decreases,  m_committed does not change!
		MemList::iterator memIt = m_priorityFreeMem.begin()->second;
		freeNode(memIt);
	}

}

// Exponentially spaced buckets.
const size_t CudaAllocBuckets::BucketSizes[CudaAllocBuckets::NumBuckets] = {
	       256,        512,       1024,       2048,       4096,       8192,
	     12288,      16384,      24576,      32768,      49152,      65536,
	     98304,     131072,     174848,     218624,     262144,     349696,
	    436992,     524288,     655360,     786432,     917504,    1048576,
	   1310720,    1572864,    1835008,    2097152,    2516736,    2936064,
	   3355648,    3774976,    4194304,    4893440,    5592576,    6291456,
	   6990592,    7689728,    8388608,    9786880,   11184896,   12582912,
	  13981184,   15379200,   16777216,   18874368,   20971520,   23068672,
	  25165824,   27262976,   29360128,   31457280,   33554432,   36910080,
	  40265472,   43620864,   46976256,   50331648,   53687296,   57042688,
	  60398080,   63753472,   67108864,   72701440,   78293760,   83886080,
	  89478656,   95070976,  100663296,  106255872,  111848192,  117440512,
	 123033088,  128625408,  134217728,  143804928,  153391872,  162978816,
	 172565760,  182152704,  191739648,  201326592,  210913792,  220500736
};

int CudaAllocBuckets::locateBucket(size_t size) const {
	if(size > m_maxObjectSize || size > BucketSizes[NumBuckets - 1])
		return NumBuckets; // return infeasible bucketIdx NumBuckets

    // Size is the lower bound (searching the bucketIdx which just fits)
	return (int)(std::lower_bound(BucketSizes, BucketSizes + NumBuckets, size) - BucketSizes);
}
};
