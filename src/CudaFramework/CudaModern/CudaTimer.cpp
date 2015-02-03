// ========================================================================================
//  CudaFramework
//  Copyright (C) 2014 by Gabriel Nützi <nuetzig (at) imes (d0t) mavt (d0t) ethz (d0t) ch>
//
//  This Source Code Form is subject to the terms of the GNU GPL 3.0 licence. 
//  If a copy of the GNU GPL 3.0 was not distributed with this
//  file, you can obtain one at http://opensource.org/licenses/GPL-3.0.
// ========================================================================================

#include "CudaFramework/CudaModern/CudaTimer.hpp"

#include <cuda_runtime.h>

namespace utilCuda{
////////////////////////////////////////////////////////////////////////////////
// CudaTimer

void CudaTimer::start() {
	cudaEventRecord(m_start);
	cudaDeviceSynchronize();
}
double CudaTimer::split() {
	cudaEventRecord(m_end);
	cudaDeviceSynchronize();
	float t;
	cudaEventElapsedTime(&t, m_start, m_end);
	m_start.swap(m_end);
	return (t / 1000.0);
}
double CudaTimer::throughput(int count, int numIterations) {
	double elapsed = split();
	return (double)numIterations * count / elapsed;
}

};
