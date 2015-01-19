// ========================================================================================
//  CudaFramework
//  Copyright (C) 2014 by Gabriel Nützi <nuetzig (at) imes (d0t) mavt (d0t) ethz (d0t) ch>
//
//  This Source Code Form is subject to the terms of the GNU GPL 3.0 licence.
//  If a copy of the GNU GPL 3.0 was not distributed with this
//  file, you can obtain one at http://opensource.org/licenses/GPL-3.0.
// ========================================================================================

#ifndef CudaFramework_Kernels_TestsGPU_TestsGPU_hpp
#define CudaFramework_Kernels_TestsGPU_TestsGPU_hpp



namespace testsGPU{

template<typename PREC> void branchTest_kernelWrap(PREC * a);
template<typename PREC> void strangeCudaBehaviour_wrap(PREC* output, PREC * mu, PREC * d, PREC * t, PREC * input);


void branchTest();

void cudaStrangeBehaviour();

}

#endif
