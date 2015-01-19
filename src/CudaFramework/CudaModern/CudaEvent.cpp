
#include "CudaFramework/CudaModern/CudaEvent.hpp"
#include "CudaFramework/CudaModern/CudaError.hpp"

#include <cuda_runtime.h>

namespace utilCuda{

    CudaEvent::CudaEvent() {
        ASSERTCHECK_CUDA(cudaEventCreate(&m_event));
    }
    CudaEvent::CudaEvent(unsigned int flags) {
        ASSERTCHECK_CUDA(cudaEventCreateWithFlags(&m_event, flags));
    }
    CudaEvent::~CudaEvent() {
        ASSERTCHECK_CUDA(cudaEventDestroy(m_event));
    }

};
