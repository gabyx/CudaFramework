// ========================================================================================
//  CudaFramework
//  Copyright (C) 2014 by Gabriel Nützi <nuetzig (at) imes (d0t) mavt (d0t) ethz (d0t) ch>
//
//  This Source Code Form is subject to the terms of the GNU GPL 3.0 licence. 
//  If a copy of the GNU GPL 3.0 was not distributed with this
//  file, you can obtain one at http://opensource.org/licenses/GPL-3.0.
// ========================================================================================
#ifndef CudaFramework_General_GPUDefines_hpp
#define CudaFramework_General_GPUDefines_hpp

#warning "This file should not be used anymore, it is only usefull for some old kernel for newer implementations please use the CUDAContext to get theses values!"

// GTX 580 3 GB Defines for easy access!
#define GPU_nMultiprocessors 16
#define GPU_WarpSize 32
#define GPU_RegsPerBlock 32768
#define GPU_ThreadsPerBlock 1024
#define GPU_ThreadDimX 1024
#define GPU_ThreadDimY 1024
#define GPU_ThreadDimZ 64
#define GPU_GridDimX 65535
#define GPU_GridDimY 65535
#define GPU_GridDimZ 65535

#define GPU_SharedPerBlock 49152
#define GPU_TotalGlobal 3220766720 // or 1543045120 for the 1.5gb version

#define GPU_PAGABLE_HOST_TO_DEVICE_SPEED 3.46777e9  // Bytes/s
#define GPU_PAGABLE_DEVICE_TO_HOST_SPEED 3.45026e9
#define GPU_PINNED_HOST_TO_DEVICE_SPEED  5.98451e9  // Bytes/s
#define GPU_PINNED_DEVICE_TO_HOST_SPEED  6.63702e9
#define GPU_MALLOC_SPEED 2.30799e9

#define GPU_FLOAT_SPEED 1482.0  // Gflops
#define GPU_DOUBLE_SPEED 197.5  // Gflops


/**
Compute Capability                                                            1.0            1.1                1.2                1.3               2.0                2.1
SM Version                                                                    sm_10          sm_11              sm_12              sm_13             sm_20              m_21
Threads / Warp                                                                32             32                 32                 32                32                 32
Warps / Multiprocessor                                                        24             24                 32                 32                48                 48
Threads / Multiprocessor                                                      768            768                1024               1024              1536               1536
Thread Blocks / Multiprocessor                                                8              8                  8                  8                 8                  8
Max Shared Memory / Multiprocessor (bytes)                                    16384          16384              16384              16384             49152              49152
Register File Size                                                            8192           8192               16384              16384             32768              32768
Register Allocation Unit Size                                                 256            256                512                512               64                 64
Allocation Granularity                                                        block          block              block              block             warp               warp
Shared Memory Allocation Unit Size                                            512            512                512                512               128                128
Warp allocation granularity (for registers)                                   2              2                  2                  2
Max Thread Block Size                                                         512            512                512                512               1024               1024

Shared Memory Size Configurations (bytes)                                     16384          16384              16384              16384             49152              49152
[note: default at top of list]                                                                                                                       16384              16384

Warp register allocation granularities                                                                                                               64                 64
[note: default at top of list]    21,22,29,30,37,38,45,46,                                                                                           128                128
*/


#endif
