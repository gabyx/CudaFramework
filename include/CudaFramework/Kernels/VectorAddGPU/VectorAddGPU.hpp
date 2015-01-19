// ========================================================================================
//  CudaFramework
//  Copyright (C) 2014 by Gabriel Nützi <nuetzig (at) imes (d0t) mavt (d0t) ethz (d0t) ch>
//
//  This Source Code Form is subject to the terms of the GNU GPL 3.0 licence.
//  If a copy of the GNU GPL 3.0 was not distributed with this
//  file, you can obtain one at http://opensource.org/licenses/GPL-3.0.
// ========================================================================================

#ifndef CudaFramework_Kernels_VectorAddGPU_VectorAddGPU_hpp
#define CudaFramework_Kernels_VectorAddGPU_VectorAddGPU_hpp

#include <fstream>

#include <cuda_runtime.h>

namespace vectorAddGPU{
   template<typename PREC>
   void vectorAdd_kernelWrap( PREC *Cdev, PREC * Adev, PREC * Bdev,  int NB,  const dim3 &threads, const dim3 &blocks);

   template<typename PREC>
   void vectorAddShared_kernelWrap( PREC *Cdev, PREC * Adev, PREC * Bdev,  int NB,  const dim3 &threads, const dim3 &blocks);



   void randomVectorAdd(int kernel);
   void performanceTestVectorAdd(std::ostream & data, std::ostream & log, int kernel);


}
#endif
