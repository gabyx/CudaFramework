//#pragma once



#include "CudaFramework/Kernels/JORProxVel/ReductionKernel/KernelsWrappers/PartitionKernel.cuh"
#include "CudaFramework/Kernels/JORProxVel/ReductionKernel/KernelsWrappers/ReductionKernel.cuh"
#include "CudaFramework/Kernels/JORProxVel/ReductionKernel/KernelsWrappers/ReduceSpineKernels.cuh"

namespace ReductionGPU {


template __host__ void partitionCsrSegRedWrap(int count,
                                              int nv,
                                              int* csrGlobal,
                                              int numRows,
                                              int numPartitions,
                                              cudaStream_t stream,
                                              int* limitsDevice,
                                              float* elapsedTime);



template __host__ void segRedCsrWrap<false,SegReduceNormalTuning<sizeof(int)>::Tuning, ReductionGPU::Operation::PLUS >(  int launch_x,
                                                                                             int numBlocks,
                                                                                             int count,
                                                                                             int* csrGlobal,
                                                                                             cudaStream_t stream,
                                                                                             int* limitsDevice,
                                                                                             int* carryOutDevice,
                                                                                             int* destGlobal,
                                                                                             int* dataGlobal,
                                                                                             int identity,
                                                                                             float* elapsedTime);

/**

template __host__ void segRedSpineWrap1Kernel1<128>(const int* limitsGlobal, int count,
        int* destGlobal, const int* carryInGlobal,int identity,
        int* carryOut_global,cudaStream_t stream,int numBlocks) ;

template __host__ void segRedSpineWrap2<128>(const int* limitsGlobal, int count,
        int* destGlobal, const int* carryInGlobal,int identity,cudaStream_t stream,int numBlocks) ;

**/
    template __host__ void segRedSpineWrap<int,ReductionGPU::Operation::PLUS>(const int* limitsGlobal,
                                           int count,
	                                       int* destGlobal,
	                                       const int* carryInGlobal,
	                                       int identity,
	                                       cudaStream_t stream,
	                                       int* carry,
	                                       float* elapsedTime);






   template __host__ void segRedCsrWrap<false,SegReduceNormalTuning<sizeof(double)>::Tuning,ReductionGPU::Operation::PLUS>(  int launch_x,
                                                                                             int numBlocks,
                                                                                             int count,
                                                                                             int* csrGlobal,
                                                                                             cudaStream_t stream,
                                                                                             int* limitsDevice,
                                                                                             double* carryOutDevice,
                                                                                             double* destGlobal,
                                                                                             double* dataGlobal,
                                                                                             double identity,
                                                                                             float* elapsedTime);

    template __host__ void segRedSpineWrap<double,ReductionGPU::Operation::PLUS>( const int* limitsGlobal,
                                            int count,
	                                        double* destGlobal,
	                                        const double* carryInGlobal,
	                                        double identity,
	                                        cudaStream_t stream,
	                                        double* carry,
	                                        float* elapsedTime);
}

/*template __host__ void partitionCsrSegRedWrap(int count,
                                              int nv,
                                              int* csrGlobal,
                                              int numRows,
                                              int numPartitions,
                                              cudaStream_t stream,
                                              int* limitsDevice,
                                              float* elapsedTime);



template __host__ void segRedCsrWrap<false,SegReduceNormalTuning<sizeof(int)>::Tuning, ReductionGPU::Operation::PLUS >(  int launch_x,
                                                                                             int numBlocks,
                                                                                             int count,
                                                                                             int* csrGlobal,
                                                                                             cudaStream_t stream,
                                                                                             int* limitsDevice,
                                                                                             int* carryOutDevice,
                                                                                             int* destGlobal,
                                                                                             int* dataGlobal,
                                                                                             int identity,
                                                                                             float* elapsedTime);


    template __host__ void segRedSpineWrap<int,ReductionGPU::Operation::PLUS>(const int* limitsGlobal,
                                           int count,
	                                       int* destGlobal,
	                                       const int* carryInGlobal,
	                                       int identity,
	                                       cudaStream_t stream,
	                                       int* carry,
	                                       float* elapsedTime);






   template __host__ void segRedCsrWrap<false,SegReduceNormalTuning<sizeof(double)>::Tuning,ReductionGPU::Operation::PLUS>(  int launch_x,
                                                                                             int numBlocks,
                                                                                             int count,
                                                                                             int* csrGlobal,
                                                                                             cudaStream_t stream,
                                                                                             int* limitsDevice,
                                                                                             double* carryOutDevice,
                                                                                             double* destGlobal,
                                                                                             double* dataGlobal,
                                                                                             double identity,
                                                                                             float* elapsedTime);

    template __host__ void segRedSpineWrap<double,ReductionGPU::Operation::PLUS>( const int* limitsGlobal,
                                            int count,
	                                        double* destGlobal,
	                                        const double* carryInGlobal,
	                                        double identity,
	                                        cudaStream_t stream,
	                                        double* carry,
	                                        float* elapsedTime);*/
