// ========================================================================================
//  CudaFramework
//  Copyright (C) 2014 by Gabriel Nützi <nuetzig (at) imes (d0t) mavt (d0t) ethz (d0t) ch>
//
//  This Source Code Form is subject to the terms of the GNU GPL 3.0 licence. 
//  If a copy of the GNU GPL 3.0 was not distributed with this
//  file, you can obtain one at http://opensource.org/licenses/GPL-3.0.
// ========================================================================================


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
