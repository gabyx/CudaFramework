#include "CudaFramework/CudaModern/CudaDevice.hpp"

#include "CudaFramework/CudaModern/CudaDeviceGroup.hpp"

#include "CudaFramework/CudaModern/CudaError.hpp"

#include "CudaFramework/CudaModern/CudaUtilities.hpp"


namespace utilCuda{


    // Helper function to destroy all devices in the device group!
    // The user of this library need to call destroyDeviceGroup() in main()
    // such that all devices are destroyed befor the CUDA runtime destroys it self which
    // leads to cuda_Error_t "unload Cuda runtime error" if this is not called!
    void destroyDeviceGroup(){
        DeviceGroup::destroyDeviceGroupCalled = true;
        deviceGroup.reset();
    }




////////////////////////////////////////////////////////////////////////////////
// CudaDevice
int CudaDevice::deviceCount() {
	if(!utilCuda::deviceGroup.get())
		deviceGroup.reset(new DeviceGroup);
	return deviceGroup->getDeviceCount();
}

CudaDevice& CudaDevice::byOrdinal(int ordinal) {
	if(ordinal < 0 || ordinal >= deviceCount()) {
		ERRORMSG_CUDA("CODE REQUESTED INVALID CUDA DEVICE " << ordinal);
	}
	return *deviceGroup->getByOrdinal(ordinal);
}

CudaDevice& CudaDevice::selected() {
	int ordinal;
	cudaError_t error = cudaGetDevice(&ordinal);
	if(cudaSuccess != error) {
		ERRORMSG_CUDA("ERROR RETRIEVING CUDA DEVICE ORDINAL");
	}
	return byOrdinal(ordinal);
}

void CudaDevice::setActive() {
	cudaError_t error = cudaSetDevice(m_ordinal);
	if(cudaSuccess != error) {
		ERRORMSG_CUDA("ERROR SETTING CUDA DEVICE TO ORDINAL " << m_ordinal);
	}
}


std::string CudaDevice::memInfoString() const{
    size_t freeMem, totalMem;
	std::stringstream s;

	cudaError_t error = cudaMemGetInfo(&freeMem, &totalMem);
	if(cudaSuccess != error) {
        ERRORMSG_CUDA("ERROR RETRIEVING MEM INFO FOR CUDA DEVICE " << m_ordinal);
	}

	double memBandwidth = (m_prop.memoryClockRate * 1000.0) * (m_prop.memoryBusWidth / 8 * 2) / 1.0e9;

    s << "=================== Memory Info for DEVICE "<< m_ordinal << "===========================" <<std::endl
    <<  "#  Free Mem [mb] : " << (double)freeMem/(1<<20) <<  " of " <<  (double)totalMem/(1<<20) <<std::endl
    <<  "#  MemFreq [Mhz] : " <<m_prop.memoryClockRate/1000.0<< " x " << m_prop.memoryBusWidth
                              << " bits -> ( "<<memBandwidth<<" GB/s )"<< std::endl
    <<   "======================================================================="<< std::endl;
    return s.str();
}

std::string CudaDevice::deviceString() const {
    std::stringstream s;
    s << memInfoString();
    utilCuda::writeCudaDeviceProbs(s,m_prop,m_ordinal);

	return s.str();
}

bool CudaDevice::hasFreeMemory() const {
	size_t freeMem, totalMem;

	cudaError_t error = cudaMemGetInfo(&freeMem, &totalMem);
	if(cudaSuccess != error) {
        ERRORMSG_CUDA("ERROR RETRIEVING MEM INFO FOR CUDA DEVICE " << m_ordinal);
	}

	return freeMem < totalMem;
}

};
