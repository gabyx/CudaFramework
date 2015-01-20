// ========================================================================================
//  CudaFramework
//  Copyright (C) 2014 by Gabriel Nützi <nuetzig (at) imes (d0t) mavt (d0t) ethz (d0t) ch>
//
//  This Source Code Form is subject to the terms of the GNU GPL 3.0 licence.
//  If a copy of the GNU GPL 3.0 was not distributed with this
//  file, you can obtain one at http://opensource.org/licenses/GPL-3.0.
// ========================================================================================

#ifndef CudaFramework_Kernels_ProxGPU_ProxGPU_hpp
#define CudaFramework_Kernels_ProxGPU_ProxGPU_hpp

#include <stdio.h>
#include <iostream>
#include <fstream>

#include <stdlib.h>
#include <time.h>

#include "CudaFramework/General/CPUTimer.hpp"
#include "CudaFramework/CudaModern/CudaError.hpp"
#include "CudaFramework/General/Utilities.hpp"
#include "CudaFramework/CudaModern/CudaMatrix.hpp"
#include "CudaFramework/CudaModern/CudaMatrixUtilities.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <fstream>

#include "ProxFunctions.hpp"

#include "CudaFramework/Kernels/MatrixVectorMultGPU/MatrixVectorMultGPU.hpp"
#include "CudaFramework/Kernels/VectorAddGPU/VectorAddGPU.hpp"



namespace proxGPU{


   /** @defgroup JOR Prox with cuBlas
   * @brief An implementation of the JOR prox for the contact ordered case. The multiplication is done with cuBlas and the prox is a vector kernel.
   */
   /* @{ */


   template<typename TCudaMatrix>
    void proxContactOrdered_RPlusAndDisk_2threads_kernelWrap(TCudaMatrix &mu_dev,
                                                                     TCudaMatrix &proxTerm_dev);


   template<typename TConvexSet , typename TCudaMatrix>
    void proxContactOrdered_1threads_kernelWrap(TCudaMatrix &mu_dev,
                                                        TCudaMatrix &proxTerm_dev);
   template<typename TConvexSet , typename TCudaMatrix>
    void proxContactOrderedWORSTCASE_1threads_kernelWrap(TCudaMatrix &mu_dev,
                                                                 TCudaMatrix &proxTerm_dev);

   template<typename TConvexSet , typename TCudaMatrix>
    void proxContactOrdered_1threads_kernelWrap(TCudaMatrix &mu_dev,
                                                        TCudaMatrix &proxTerm_dev,
                                                        TCudaMatrix &d_dev );

   template< typename TJorProxKernelSettings , typename TCudaMatrix>
    void jorProxContactOrdered_1threads_kernelWrap( TCudaMatrix &mu_dev,
                                                            TCudaMatrix &y_dev,
                                                            typename TCudaMatrix::PREC alpha,
                                                            TCudaMatrix &A_dev,
                                                            TCudaMatrix &x_dev,
                                                            typename TCudaMatrix::PREC beta,
                                                            TCudaMatrix &b_dev);

    template<typename TSorProxKernelSettings , typename TCudaMatrix>
    void sorProxContactOrdered_1threads_StepA_kernelWrap( TCudaMatrix &mu_dev,
                                                            TCudaMatrix &x_new_dev,
                                                            TCudaMatrix &T_dev,
                                                            TCudaMatrix &d_dev,
                                                            TCudaMatrix &t_dev,
                                                            int kernelAIdx,
                                                            bool * convergedFlag_dev,
                                                            typename TCudaMatrix::PREC absTOL, typename TCudaMatrix::PREC relTOL
                                                            );

    template<typename TSorProxKernelSettings , typename TCudaMatrix>
    void sorProx_StepB_kernelWrap(   TCudaMatrix &t_dev,
                                             TCudaMatrix &T_dev,
                                             TCudaMatrix &x_new_dev,
                                             int kernelAIdx
                                             );

   template<typename TRelaxedSorProxKernelSettings , typename TCudaMatrix>
    void sorProxRelaxed_StepA_kernelWrap(  TCudaMatrix &mu_dev,
                                                   TCudaMatrix &x_new_dev,
                                                   TCudaMatrix &t_dev,
                                                   TCudaMatrix &d_dev,
                                                   int kernelAIdx);

    template<typename TRelaxedSorProxKernelSettings , typename TCudaMatrix>
    void sorProxRelaxed_StepB_kernelWrap(  TCudaMatrix &t_dev,
                                                   TCudaMatrix &T_dev,
                                                   TCudaMatrix &x_new_dev,
                                                   int kernelAIdx);



    template<typename TCudaMatrix>
    void convergedEach_kernelWrap(TCudaMatrix &x_new_dev,
                                          TCudaMatrix &x_old_dev,
                                          bool * convergedFlag_dev,
                                          typename TCudaMatrix::PREC absTOL,
                                          typename TCudaMatrix::PREC relTOL);

   /** @} */

}
#endif
