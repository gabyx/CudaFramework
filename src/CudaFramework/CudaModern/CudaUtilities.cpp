// ========================================================================================
//  CudaFramework
//  Copyright (C) 2014 by Gabriel Nützi <nuetzig (at) imes (d0t) mavt (d0t) ethz (d0t) ch>
//
//  This Source Code Form is subject to the terms of the GNU GPL 3.0 licence. 
//  If a copy of the GNU GPL 3.0 was not distributed with this
//  file, you can obtain one at http://opensource.org/licenses/GPL-3.0.
// ========================================================================================

#include "CudaFramework/CudaModern/CudaUtilities.hpp"

#include "CudaFramework/CudaModern/CudaError.hpp"


#include <iostream>
#include <chrono>
#include <ostream>
#include <cuda_runtime.h>


#define START_TIMER(start)  auto start = std::chrono::high_resolution_clock::now();
#define STOP_TIMER( count, start)  \
    auto count = std::chrono::duration<double,std::milli>(std::chrono::high_resolution_clock::now() - start ).count();


namespace utilCuda{

    void printAllCudaDeviceSpecs()
    {
        int ndevices;
        cudaGetDeviceCount(&ndevices);

        cudaDeviceProp props;

        for(int i=0;i<ndevices;i++){
            CHECK_CUDA(cudaGetDeviceProperties(&props, i));
            writeCudaDeviceProbs(std::cout, props,i);
        }
    }

    void writeCudaDeviceProbs(std::ostream & file, const cudaDeviceProp & probs, int devicenumber){
        file << "#==================== DEVICE PROBS FOR DEVICE "<< devicenumber << "========================" << std::endl;
        file << "#  Device Name: "<<  probs.name <<std::endl;
        file << "#  Total Global Memory: "<<  probs.totalGlobalMem<<std::endl;
        file << "#  Shared Memory/Block: "<< probs.sharedMemPerBlock<<std::endl;
        file << "#  Registers (32bit)/Block: "<<  probs.regsPerBlock<<std::endl;
        file << "#  Warp Size: "<< probs.warpSize<<std::endl;
        file << "#  Memory Pitch: "<<  probs.memPitch<<std::endl;
        file << "#  Max. Threads/Block: "<<  probs.maxThreadsPerBlock<<std::endl;
        file << "#  Max. Threads Dim: "<<  probs.maxThreadsDim[0] <<","<< probs.maxThreadsDim[1] <<"," << probs.maxThreadsDim[2] <<std::endl;
        file << "#  Max. Grid Size: "<<  probs.maxGridSize[0] <<","<< probs.maxGridSize[1] <<"," << probs.maxGridSize[2] <<std::endl;
        file << "#  Total Const Memory: "<<  probs.totalConstMem<<std::endl;
        file << "#  Compute Capability: "<<  probs.major <<","<< probs.minor<<std::endl;
        file << "#  Clock Rate: "<<  probs.clockRate<<std::endl;
        file << "#  Texture Alignment: " << probs.textureAlignment<<std::endl;
        file << "#  Devide Overlap: " << probs.deviceOverlap<<std::endl;
        file << "#  Multiprocessor Count: " <<  probs.multiProcessorCount<<std::endl;
        file << "#  Kernel Exec. Timeout: " <<  probs.kernelExecTimeoutEnabled<<std::endl;
        file << "#  Integrated GPU: " <<  probs.integrated<<std::endl;
        file << "#  Can map host memory GPU: " <<  probs.canMapHostMemory<<std::endl;
        file << "#  Compute Mode: " <<  probs.computeMode<<std::endl;
        file << "#  Concurrent Kernels: " <<  probs.concurrentKernels<<std::endl;
        file << "#  ECC Enabled: " <<  probs.ECCEnabled<<std::endl;
        file << "#  PCI Bus ID: " <<  probs.pciBusID<<std::endl;
        file << "#  PCI Device: " <<  probs.pciDeviceID<<std::endl;
        file << "#  TCC Driver: " <<  probs.tccDriver<<std::endl;
        file << "#  Async Engine Count: " << probs.asyncEngineCount<<std::endl;
        file << "#  Unified Adressing: " <<  probs.unifiedAddressing<<std::endl;
        file << "#  Mem. Clock Rate: " <<  probs.memoryClockRate<<std::endl;
        file << "#  Mem. Bus Width:" <<  probs.memoryBusWidth<<std::endl;
        file << "#  L2 Cache Size: " <<  probs.l2CacheSize<<std::endl;
        file << "#  Max Threads per MultiProcessor: " <<  probs.maxThreadsPerMultiProcessor<<std::endl;
        file << "#========================================================================" << std::endl;
    }

    namespace memoryTransferTest{

        double testHostToDeviceTransfer(unsigned int memSize, bandwithData::memoryMode memMode, bool wc)
        {
           unsigned int timer = 0;
           float elapsedTimeInMs = 0.0f;
           double bandwidthInMBs = 0.0f;
           cudaEvent_t start, stop;


           CHECK_CUDA( cudaEventCreate( &start ) );
           CHECK_CUDA( cudaEventCreate( &stop ) );

           //allocate host memory
           unsigned char *h_odata = NULL;
           if( bandwithData::PINNED == memMode )
           {
        #if CUDART_VERSION >= 2020
              //pinned memory mode - use special function to get OS-pinned memory
              CHECK_CUDA( cudaHostAlloc( (void**)&h_odata, memSize, (wc) ? cudaHostAllocWriteCombined : 0 ) );
        #else
              //pinned memory mode - use special function to get OS-pinned memory
              CHECK_CUDA( cudaMallocHost( (void**)&h_odata, memSize ) );
        #endif
           }
           else
           {
              //pageable memory mode - use malloc
              h_odata = (unsigned char *)malloc( memSize );
           }
           unsigned char *h_cacheClear1 = (unsigned char *)malloc( bandwithData::cacheClearSize );
           unsigned char *h_cacheClear2 = (unsigned char *)malloc( bandwithData::cacheClearSize );

           //initialize the memory
           for(unsigned int i = 0; i < memSize/sizeof(unsigned char); i++)
           {
              h_odata[i] = (unsigned char) (i & 0xff);
           }
           for(unsigned int i = 0; i < bandwithData::cacheClearSize / sizeof(unsigned char); i++)
           {
              h_cacheClear1[i] = (unsigned char) (i & 0xff);
              h_cacheClear2[i] = (unsigned char) (0xff - (i & 0xff));
           }

           //allocate device memory
           unsigned char* d_idata;
           CHECK_CUDA( cudaMalloc( (void**) &d_idata, memSize));

           START_TIMER(startCPU);
           CHECK_CUDA( cudaEventRecord( start, 0 ) );

           //copy host memory to device memory
           if( bandwithData::PINNED == memMode )
           {
              for(unsigned int i = 0; i < bandwithData::memcpyIterations; i++)
              {
                 CHECK_CUDA( cudaMemcpyAsync( d_idata, h_odata, memSize,
                    cudaMemcpyHostToDevice, 0) );
              }
           }
           else {
              for(unsigned int i = 0; i < bandwithData::memcpyIterations; i++)
              {
                 CHECK_CUDA( cudaMemcpy( d_idata, h_odata, memSize,
                    cudaMemcpyHostToDevice) );
              }
           }

           CHECK_CUDA( cudaEventRecord( stop, 0 ) );
           CHECK_CUDA( cudaThreadSynchronize() );
           //total elapsed time in ms
           STOP_TIMER(count,startCPU);

           CHECK_CUDA( cudaEventElapsedTime( &elapsedTimeInMs, start, stop ) );
           if ( bandwithData::PINNED != memMode || bandwithData::bDontUseGPUTiming )
           {
              elapsedTimeInMs = count;
           }

           //calculate bandwidth in MB/s
           bandwidthInMBs = (1e3 * memSize * (float)bandwithData::memcpyIterations) /
              (elapsedTimeInMs * (float)(1 << 20));

           //clean up memory
           CHECK_CUDA( cudaEventDestroy(stop) );
           CHECK_CUDA( cudaEventDestroy(start) );

           if( bandwithData::PINNED == memMode )
           {
              CHECK_CUDA( cudaFreeHost(h_odata) );
           }
           else
           {
              free(h_odata);
           }
           free(h_cacheClear1);
           free(h_cacheClear2);
           CHECK_CUDA(cudaFree(d_idata));

           return bandwidthInMBs;
        }


        double testDeviceToHostTransfer(unsigned int memSize, bandwithData::memoryMode memMode, bool wc)
        {
           unsigned int timer = 0;
           float elapsedTimeInMs = 0.0;
           double bandwidthInMBs = 0.0;
           unsigned char *h_idata = NULL;
           unsigned char *h_odata = NULL;
           cudaEvent_t start, stop;


           CHECK_CUDA  ( cudaEventCreate( &start ) );
           CHECK_CUDA  ( cudaEventCreate( &stop ) );

           //allocate host memory
           if( bandwithData::PINNED == memMode )
           {
              //pinned memory mode - use special function to get OS-pinned memory
        #if CUDART_VERSION >= 2020
              CHECK_CUDA( cudaHostAlloc( (void**)&h_idata, memSize, (wc) ? cudaHostAllocWriteCombined : 0 ) );
              CHECK_CUDA( cudaHostAlloc( (void**)&h_odata, memSize, (wc) ? cudaHostAllocWriteCombined : 0 ) );
        #else
              CHECK_CUDA( cudaMallocHost( (void**)&h_idata, memSize ) );
              CHECK_CUDA( cudaMallocHost( (void**)&h_odata, memSize ) );
        #endif
           }
           else
           {
              //pageable memory mode - use malloc
              h_idata = (unsigned char *)malloc( memSize );
              h_odata = (unsigned char *)malloc( memSize );
           }
           //initialize the memory
           for(unsigned int i = 0; i < memSize/sizeof(unsigned char); i++)
           {
              h_idata[i] = (unsigned char) (i & 0xff);
           }

           // allocate device memory
           unsigned char* d_idata;
           CHECK_CUDA( cudaMalloc( (void**) &d_idata, memSize));

           //initialize the device memory
           CHECK_CUDA( cudaMemcpy( d_idata, h_idata, memSize,
              cudaMemcpyHostToDevice) );

           //copy data from GPU to Host
           START_TIMER(startCPU);
           CHECK_CUDA( cudaEventRecord( start, 0 ) );
           if( bandwithData::PINNED == memMode )
           {
              for( unsigned int i = 0; i < bandwithData::memcpyIterations; i++ )
              {
                 CHECK_CUDA( cudaMemcpyAsync( h_odata, d_idata, memSize,
                    cudaMemcpyDeviceToHost, 0) );
              }
           }
           else
           {
              for( unsigned int i = 0; i < bandwithData::memcpyIterations; i++ )
              {
                 CHECK_CUDA( cudaMemcpy( h_odata, d_idata, memSize,
                    cudaMemcpyDeviceToHost) );
              }
           }
           CHECK_CUDA( cudaEventRecord( stop, 0 ) );

           // make sure GPU has finished copying
           CHECK_CUDA( cudaThreadSynchronize() );
           //get the the total elapsed time in ms
           STOP_TIMER(count,startCPU);
           CHECK_CUDA( cudaEventElapsedTime( &elapsedTimeInMs, start, stop ) );
           if( bandwithData::PINNED != memMode || bandwithData::bDontUseGPUTiming )
           {
              elapsedTimeInMs =  count;
           }

           //calculate bandwidth in MB/s
           bandwidthInMBs = (1e3 * memSize * (float)bandwithData::memcpyIterations) /
              (elapsedTimeInMs * (float)(1 << 20));

           //clean up memory
           CHECK_CUDA( cudaEventDestroy(stop) );
           CHECK_CUDA( cudaEventDestroy(start) );

           if( bandwithData::PINNED == memMode )
           {
              CHECK_CUDA( cudaFreeHost(h_idata) );
              CHECK_CUDA( cudaFreeHost(h_odata) );
           }
           else
           {
              free(h_idata);
              free(h_odata);
           }
           CHECK_CUDA(cudaFree(d_idata));

           return bandwidthInMBs;
        }


        double testMalloc(unsigned int memSize)
        {
           unsigned int timer = 0;
           float elapsedTimeInMs = 0.0;
           double bandwidthInMBs = 0.0;
           cudaEvent_t start, stop;

           CHECK_CUDA  ( cudaEventCreate( &start ) );
           CHECK_CUDA  ( cudaEventCreate( &stop ) );

           // allocate device memory
           unsigned char* d_idata;

           for(int i = 0 ; i< bandwithData::memcpyIterations;i++){
              START_TIMER(startCPU);
              CHECK_CUDA( cudaEventRecord( start, 0 ) );
              CHECK_CUDA( cudaMalloc( (void**) &d_idata, memSize));
              CHECK_CUDA( cudaEventRecord( stop, 0 ) );
              // make sure GPU has finished copying
              CHECK_CUDA( cudaThreadSynchronize() );
              //get the the total elapsed time in ms
              STOP_TIMER(count,startCPU);
              CHECK_CUDA( cudaEventElapsedTime( &elapsedTimeInMs, start, stop ) );
              if( bandwithData::bDontUseGPUTiming )
              {
                 elapsedTimeInMs =  (float) count;
              }

              //calculate bandwidth in MB/s
              bandwidthInMBs += (1e3 * memSize) /
                 (elapsedTimeInMs * (float)(1 << 20));
              CHECK_CUDA(cudaFree(d_idata));
           }

           //clean up memory
           CHECK_CUDA( cudaEventDestroy(stop) );
           CHECK_CUDA( cudaEventDestroy(start) );



           return bandwidthInMBs / bandwithData::memcpyIterations;
        }

        void measureCopySpeeds(){

           std::cout << "Copy speed test to GPU =====================================" <<std::endl;

           std::cout << "Pagable Memory: Host-->Device: \t" << testHostToDeviceTransfer(bandwithData::default_size,bandwithData::PAGEABLE,false) * (double)(1 << 20) << " B/s" << std::endl;
           std::cout << "Pagable Memory: Device-->Host: \t" << testDeviceToHostTransfer(bandwithData::default_size,bandwithData::PAGEABLE,false) * (double)(1 << 20)<< " B/s" << std::endl;
           std::cout << "Pinned Memory: Host-->Device: \t" << testHostToDeviceTransfer(bandwithData::default_size,bandwithData::PINNED,false) * (double)(1 << 20)<< " B/s" << std::endl;
           std::cout << "Pinned Memory: Device-->Host: \t" << testDeviceToHostTransfer(bandwithData::default_size,bandwithData::PINNED,false) * (double)(1 << 20)<< " B/s" << std::endl;

           std::cout << "Pagable Memory: Malloc: \t" << testMalloc(bandwithData::default_size) * (double)(1 << 20)<< " B/s" << std::endl;
        }

    };

};
