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

#ifndef CudaFramework_CudaModern_CudaDeviceGroup_hpp
#define CudaFramework_CudaModern_CudaDeviceGroup_hpp

#include <memory>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>

#include "CudaFramework/General/AssertionDebug.hpp"
#include "CudaFramework/CudaModern/CudaError.hpp"
#include "CudaFramework/CudaModern/CudaDevice.hpp"


namespace utilCuda{

    extern cudaError_t getCudaCompilerVersion(unsigned int ordinal, cudaFuncAttributes & attr);

    // Global extern struct which contains all devices!
    // The user of this library need to call destroyDeviceGroup() in main()
    // such that all devices are destroyed befor the CUDA runtime destroys it self!
    class DeviceGroup;
    extern std::unique_ptr<DeviceGroup> deviceGroup;


    class DeviceGroup {

        friend void destroyDeviceGroup();

    private:

        // This static global bool is to determine if destroyDeviceGroup() has been used.
        // The user of this library need to call destroyDeviceGroup() in main()
        // such that all devices are destroyed befor the CUDA runtime destroys it self which
        // leads to cuda_Error_t "unload Cuda runtime error" if this is not called!
        static bool destroyDeviceGroupCalled;

        int numCudaDevices;
        CudaDevice** cudaDevices;

    public:
        DeviceGroup() {
            numCudaDevices = -1;
            cudaDevices = 0;
        }

        int getDeviceCount() {
            if(-1 == numCudaDevices) {
                cudaError_t error = cudaGetDeviceCount(&numCudaDevices);
                if(cudaSuccess != error || numCudaDevices <= 0) {
                    ERRORMSG_CUDA("ERROR ENUMERATING CUDA DEVICES. Exiting.");
                }
                cudaDevices = new CudaDevice*[numCudaDevices];
                std::memset(cudaDevices, 0, sizeof(CudaDevice*) * numCudaDevices);
            }
            return numCudaDevices;
        }

        CudaDevice* getByOrdinal(int ordinal) {
            if(ordinal >= getDeviceCount()) return 0;

            if(!cudaDevices[ordinal]) {
                // Retrieve the device properties.
                CudaDevice* device = cudaDevices[ordinal] = new CudaDevice;
                device->m_ordinal = ordinal;
                cudaError_t error = cudaGetDeviceProperties(&device->m_prop, ordinal);
                if(cudaSuccess != error) {
                    ERRORMSG_CUDA("FAILURE TO CREATE CUDA DEVICE: " << ordinal);
                }

                // Get the compiler version for this device.
                cudaFuncAttributes attr;
                if(cudaSuccess == getCudaCompilerVersion(ordinal,attr))
                    device->m_ptxVersion = 10 * attr.ptxVersion;
                else {
                    printf("NOT COMPILED WITH COMPATIBLE PTX VERSION FOR DEVICE"
                        " %d\n", ordinal);
                    // The module wasn't compiled with support for this device.
                    device->m_ptxVersion = 0;
                }
            }
            return cudaDevices[ordinal];
        }

        ~DeviceGroup() {
            CUDA_DESTRUCTOR_MESSAGE(this);
            //make sure this constructor has been called prior to the end of main!
            if(destroyDeviceGroupCalled==false){
                ERRORMSG_CUDA("Please properly destroy all devices with destroyDeviceGroup()!" << std::endl <<
                              "Without this we would destroy the devices now, which will result in \"unload cuda runtime error\" ")
            }
            if(cudaDevices) {
                for(int i = 0; i < numCudaDevices; ++i)
                    delete cudaDevices[i];
                delete [] cudaDevices;
            }
            ASSERTCHECK_CUDA(cudaDeviceReset());
        }
    };


};



#endif

