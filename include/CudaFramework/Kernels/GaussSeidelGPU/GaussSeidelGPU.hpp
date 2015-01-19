// ========================================================================================
//  CudaFramework
//  Copyright (C) 2014 by Gabriel Nützi <nuetzig (at) imes (d0t) mavt (d0t) ethz (d0t) ch>
//
//  This Source Code Form is subject to the terms of the GNU GPL 3.0 licence.
//  If a copy of the GNU GPL 3.0 was not distributed with this
//  file, you can obtain one at http://opensource.org/licenses/GPL-3.0.
// ========================================================================================

#ifndef CudaFramework_Kernels_GaussSeidelGPU_GaussSeidelGPU_hpp
#define CudaFramework_Kernels_GaussSeidelGPU_GaussSeidelGPU_hpp

#include "CudaFramework/CudaModern/CudaMatrix.hpp"

namespace gaussSeidelGPU {

template<typename TCudaMatrix>
void blockGaussSeidelStepA_kernelWrap( TCudaMatrix &G_dev, TCudaMatrix &c_dev, TCudaMatrix &t_dev, TCudaMatrix &x_old_dev,  int j_g);

template<typename TCudaMatrix>
void blockGaussSeidelStepANoDivision_kernelWrap( TCudaMatrix &T_dev, TCudaMatrix &d_dev, TCudaMatrix &t_dev, TCudaMatrix &x_old_dev,  int j_g, bool* convergedFlag_dev, typename TCudaMatrix::PREC absTOL, typename TCudaMatrix::PREC relTOL);


template<typename TCudaMatrix>
void blockGaussSeidelStepACorrect_kernelWrap( TCudaMatrix &G_dev, TCudaMatrix &c_dev, TCudaMatrix &t_dev, TCudaMatrix &x_old_dev,  int j_g);

template<typename TCudaMatrix>
void blockGaussSeidelStepACorrectNoDivision_kernelWrap( TCudaMatrix &T_dev, TCudaMatrix &d_dev, TCudaMatrix &t_dev, TCudaMatrix &x_old_dev,  int j_g);
template<typename TCudaMatrix>
void blockGaussSeidelStepACorrectNoDivision_kernelWrap( TCudaMatrix &T_dev, TCudaMatrix &d_dev, TCudaMatrix &t_dev, TCudaMatrix &x_old_dev,  int j_g, bool* convergedFlag_dev, typename TCudaMatrix::PREC absTOL, typename TCudaMatrix::PREC relTOL);

template<typename TCudaMatrix>
void blockGaussSeidelStepB_kernelWrap( TCudaMatrix &G_dev, TCudaMatrix &c_dev, TCudaMatrix &t_dev, TCudaMatrix &x_old_dev,  int j_g);

void gaussSeidelTest();
void gaussSeidelTestNoDivision();
void gaussSeidelTestNoDivisionWithError();

}
#endif
