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
