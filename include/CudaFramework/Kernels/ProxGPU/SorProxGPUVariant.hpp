// ========================================================================================
//  CudaFramework
//  Copyright (C) 2014 by Gabriel Nützi <nuetzig (at) imes (d0t) mavt (d0t) ethz (d0t) ch>
//
//  This Source Code Form is subject to the terms of the GNU GPL 3.0 licence.
//  If a copy of the GNU GPL 3.0 was not distributed with this
//  file, you can obtain one at http://opensource.org/licenses/GPL-3.0.
// ========================================================================================

#ifndef CudaFramework_Kernels_ProxGPU_SorProxGPUVariant_hpp
#define CudaFramework_Kernels_ProxGPU_SorProxGPUVariant_hpp

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include <algorithm>

#include <Eigen/Dense>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <tinyformat/TinyFormatInclude.hpp>

#include "CudaFramework/General/CPUTimer.hpp"
#include "CudaFramework/General/StaticAssert.hpp"
#include "CudaFramework/General/TypeTraitsHelper.hpp"
#include "CudaFramework/CudaModern/CudaError.hpp"
#include "CudaFramework/General/Utilities.hpp"
#include "CudaFramework/CudaModern/CudaMatrix.hpp"
#include "CudaFramework/CudaModern/CudaMatrixUtilities.hpp"


#include "CudaFramework/General/FlopsCounting.hpp"

#include "CudaFramework/Kernels/ProxGPU/ProxSettings.hpp"
#include "CudaFramework/Kernels/ProxGPU/ProxGPU.hpp"
#include "CudaFramework/Kernels/ProxGPU/ProxKernelSettings.hpp"

#include "ConvexSets.hpp"

/**
* @addtogroup ProxTestVariant
* @defgroup SorProxGPUVariant Sor Prox GPUVariants
* @detailed VariantId specifies which variant is launched:
* Here the different variants have been included in one class!
* To be more flexible we can also completely reimplement the whole class for another GPUVariant!
* @{
*/

using namespace TypeTraitsHelper;



//Specialisation
template<typename TSorProxGPUVariantSettingsWrapper>
class SorProxGPUVariant<TSorProxGPUVariantSettingsWrapper, ConvexSets::RPlusAndDisk> {
public:

    DEFINE_SorProxGPUVariant_SettingsWrapper(TSorProxGPUVariantSettingsWrapper);

    SorProxGPUVariant()
        :Matlab(Eigen::FullPrecision, 0, ", ", ";\n", "", "", "[", "]"), m_nMaxIterations(nMaxIterations), pConvergedFlag_dev(NULL) {

    }
    ~SorProxGPUVariant() {

    }

    void setSettings(unsigned int iterations, PREC absTol, PREC relTol) {
        m_nMaxIterations = iterations;
        m_absTOL = absTol;
        m_relTOL = relTol;
    }
    /** Check settings at runtime, static settings are already checked at compile time*/
    bool checkSettings(int gpuID) {
        switch(VariantId) {
        case 1:
            return SorProxSettings1::checkSettings(gpuID);
        case 2:
            return SorProxSettings2::checkSettings(gpuID);
        case 3:
            return SorProxSettings3::checkSettings(gpuID);
        case 4:
            return SorProxSettings4::checkSettings(gpuID);
        case 5:
            return SorProxSettings5::checkSettings(gpuID);
        default:
            ERRORMSG("No settings check specified for variant: " << VariantId << std::endl)
            return false;
        }
    }

    typedef typename ManualOrDefault<  IsEqual<VariantId,1>::result, TKernelSettings,SorProxSettings1RPlusAndDisk>::TValue SorProxSettings1;
    typedef typename ManualOrDefault<  IsEqual<VariantId,2>::result, TKernelSettings,SorProxSettings2RPlusAndDisk>::TValue SorProxSettings2;
    typedef typename ManualOrDefault<  IsEqual<VariantId,3>::result, TKernelSettings,SorProxSettings1RPlusAndDisk>::TValue SorProxSettings3;
    typedef typename ManualOrDefault<  IsEqual<VariantId,4>::result, TKernelSettings,SorProxSettings1RPlusAndDisk>::TValue SorProxSettings4;

    typedef typename ManualOrDefault<  IsEqual<VariantId,5>::result, TKernelSettings,RelaxedSorProxSettings1RPlusAndDisk >::TValue SorProxSettings5;

    static std::string getVariantName() {
        std::stringstream s;
        switch(VariantId) {
        case 1:
            s<< "Full SOR";
            break;
        case 2:
            s<< "Full SOR";
            break;
        case 3:
            s<< "Full SOR";
            break;
        case 4:
            s<< "Full SOR";
            break;
        case 5:
            s<< "Relaxed SOR";
            break;
        default:
            s<< "NO DESCRIPTION: probably wrong VariantId selected!";
            break;
        }
        s << ((bAbortIfConverged)? std::string("[Convergence Check]") : std::string(""));
        return s.str();
    }

    static std::string getVariantDescriptionShort() {
        std::stringstream s;
        switch(VariantId) {
        case 1:
            s<< "[Step A][Step B]";
            break;
        case 2:
            s<< "[Step A][Step B]";
            break;
        case 3:
            s<< "[Step A]";
            break;
        case 4:
            s<< "[Step B]";
            break;
        case 5:
            s<< "[Step A][Step B]";
            break;
        default:
            s<< "NO DESCRIPTION: probably wrong VariantId selected!";
            break;
        }
        s << ((bAbortIfConverged)? std::string("[Convergence Check]") : std::string(""));
        return s.str();
    }

    static std::string getVariantDescriptionLong() {
        std::stringstream s;
        switch(VariantId) {
        case 1:
            s << SOR_PROXKERNEL_SETTINGS_STR(SorProxSettings1);
            break;
        case 2:
            s << SOR_PROXKERNEL_SETTINGS_STR(SorProxSettings2);
            break;
        case 3:
            s << SOR_PROXKERNEL_A_SETTINGS_STR(SorProxSettings3);
            break;
        case 4:
            s << SOR_PROXKERNEL_B_SETTINGS_STR(SorProxSettings4);
            break;
        case 5:
            s << RELAXED_SOR_PROXKERNEL_SETTINGS_STR(SorProxSettings5);
            break;
        default:
            s << "NO DESCRIPTION: probably wrong VariantId selected!";
            break;
        }
        return s.str() + std::string(", Matrix T aligned: ") + ((alignMatrix)? std::string("on") : std::string("off"));;
    }

    double getNOps() {
        int loops;
        double matrixMultplyFLOPS;
        switch(VariantId) {
        case 1:
            return  evaluateProxTermSOR_FLOPS(T_dev.m_M,T_dev.m_N) + proxContactOrdered_RPlusAndDisk_1threads_kernel_FLOPS(T_dev.m_M / SorProxSettings1::ProxPackageSize);
        case 2:
            return  evaluateProxTermSOR_FLOPS(T_dev.m_M,T_dev.m_N) + proxContactOrdered_RPlusAndDisk_1threads_kernel_FLOPS(T_dev.m_M / SorProxSettings2::ProxPackageSize);
        case 3:
            loops = ((T_dev.m_M + ( SorProxSettings3::BlockDimKernelA - 1 ) ) / SorProxSettings3::BlockDimKernelA) ;
            matrixMultplyFLOPS = (loops -1 ) *  (2* SorProxSettings3::BlockDimKernelA *SorProxSettings3::BlockDimKernelA - SorProxSettings3::BlockDimKernelA) ;
            matrixMultplyFLOPS +=    (2*( T_dev.m_M -(loops-1)*SorProxSettings3::BlockDimKernelA)*( T_dev.m_M -(loops-1)*SorProxSettings3::BlockDimKernelA) - ( T_dev.m_M -(loops-1)*SorProxSettings3::BlockDimKernelA));  // So many matrix multiplyses n
            return   matrixMultplyFLOPS + T_dev.m_M + proxContactOrdered_RPlusAndDisk_1threads_kernel_FLOPS(T_dev.m_M / SorProxSettings3::ProxPackageSize);
        case 4:
            loops = ((T_dev.m_M + ( SorProxSettings4::BlockDimKernelA -1 ) ) / SorProxSettings4::BlockDimKernelA) ;
            matrixMultplyFLOPS = 2*T_dev.m_M*T_dev.m_N - T_dev.m_M; // The whole matrix, subtract kernel a below!
            matrixMultplyFLOPS -=    (loops -1 ) *  (2* SorProxSettings4::BlockDimKernelA *SorProxSettings4::BlockDimKernelA - SorProxSettings4::BlockDimKernelA) ;
            matrixMultplyFLOPS -=    (2*( T_dev.m_M -(loops-1)*SorProxSettings4::BlockDimKernelA)*( T_dev.m_M -(loops-1)*SorProxSettings4::BlockDimKernelA) - ( T_dev.m_M -(loops-1)*SorProxSettings4::BlockDimKernelA));  // So many matrix multiplyses n
            return matrixMultplyFLOPS;
        case 5:
            return evaluateProxTermSOR_FLOPS(T_dev.m_M,T_dev.m_N) + proxContactOrdered_RPlusAndDisk_1threads_kernel_FLOPS(T_dev.m_M / SorProxSettings5::ProxPackageSize);
        default:
            return 0;
        }
    }

    double getBytesReadWrite() {
        int loops;
        double matrixMultplyReadWrite;
        double readWriteNumbers;
        switch(VariantId) {
        case 1:
            loops = ((T_dev.m_M + ( SorProxSettings1::BlockDimKernelA -1 ) ) / SorProxSettings1::BlockDimKernelA) ;
            readWriteNumbers =  /*T_dev (overall):*/ T_dev.m_M*T_dev.m_N + /*KERNEL A*/ /*t_dev,y_dev :*/ (2*T_dev.m_M+T_dev.m_M) + /*d_dev: */ T_dev.m_M    +   /*KERNEL B*/ /*t_dev:*/ (loops-1) * (T_dev.m_M)   + /*x*/ T_dev.m_M;
            return  readWriteNumbers * sizeof(PREC);
        case 2:
            loops = ((T_dev.m_M + ( SorProxSettings2::BlockDimKernelA -1 ) ) / SorProxSettings2::BlockDimKernelA) ;

            readWriteNumbers =   /*T_dev (overall):*/ T_dev.m_M*T_dev.m_N + /*KERNEL A*/ /*t_dev,y_dev :*/ (2*T_dev.m_M+T_dev.m_M)  + /*d_dev: */ T_dev.m_M    +   /*KERNEL B*/ /*t_dev:*/ (loops-1) * (T_dev.m_M)  + /*x*/ T_dev.m_M;
            return  readWriteNumbers * sizeof(PREC);
        case 3:
            loops = ((T_dev.m_M + ( SorProxSettings3::BlockDimKernelA -1 ) ) / SorProxSettings3::BlockDimKernelA) ;
            matrixMultplyReadWrite = (loops -1 ) *  (SorProxSettings3::BlockDimKernelA *SorProxSettings3::BlockDimKernelA) ;
            matrixMultplyReadWrite +=    (   ( T_dev.m_M -(loops-1)*SorProxSettings3::BlockDimKernelA)*( T_dev.m_M -(loops-1)*SorProxSettings3::BlockDimKernelA)) ;
            readWriteNumbers =    /*KERNEL A*/ matrixMultplyReadWrite + /*t_dev,y_dev :*/ (2*T_dev.m_M+T_dev.m_M) + /*d_dev: */ T_dev.m_M;
            return   readWriteNumbers * sizeof(PREC);
        case 4:
            loops = ((T_dev.m_M + ( SorProxSettings3::BlockDimKernelA -1 ) ) / SorProxSettings3::BlockDimKernelA) ;
            matrixMultplyReadWrite = T_dev.m_M*T_dev.m_N;
            matrixMultplyReadWrite -= (loops -1 ) *  (SorProxSettings4::BlockDimKernelA *SorProxSettings4::BlockDimKernelA) ;
            matrixMultplyReadWrite -=    (( T_dev.m_M -(loops-1)*SorProxSettings4::BlockDimKernelA)*( T_dev.m_M -(loops-1)*SorProxSettings4::BlockDimKernelA));
            readWriteNumbers =   matrixMultplyReadWrite +  /*KERNEL B*/ /*t_dev:*/ (loops-1) * (T_dev.m_M) + /*x*/ T_dev.m_M;
            return   readWriteNumbers * sizeof(PREC);
        case 5:
            loops = ((T_dev.m_M + ( SorProxSettings5::BlockDimKernelA -1 ) ) / SorProxSettings5::BlockDimKernelA) ;
            readWriteNumbers =  /*T_dev (overall):*/ T_dev.m_M*T_dev.m_N + /*KERNEL A*/ /*t_dev,y_dev :*/ (2*T_dev.m_M+T_dev.m_M) + /*d_dev: */ T_dev.m_M    +   /*KERNEL B*/ /*t_dev:*/ (loops) * (T_dev.m_M) + /*x*/ T_dev.m_M;
            return  readWriteNumbers * sizeof(PREC);
        default:
            return 0;
        }
    }


    unsigned int getTradeoff() {
        switch(VariantId) {
        case 1:
            return 400;
        case 2:
            return 400;
        case 5:
            if(SorProxSettings5::ProxPackages == 2) {
                return 550;
            } else if(SorProxSettings5::ProxPackages == 4) {
                return 450;
            } else if(SorProxSettings5::ProxPackages == 8) {
                return 350;
            } else if(SorProxSettings5::ProxPackages == 16) {
                return 300;
            } else if(SorProxSettings5::ProxPackages == 32) {
                return 260;
            } else if(SorProxSettings5::ProxPackages == 64 || SorProxSettings5::ProxPackages == 128) {
                return 200;
            } else {
                return 300;
            }
        default:
            return 200;
        }
    }


    void initialize(std::ostream* pLog, std::ofstream* pMatlab_file) {
        m_pLog = pLog;
        m_pMatlab_file = pMatlab_file;
    }

    void finalize() {

    }

    void setDeviceToUse(int device) {
        CHECK_CUDA(cudaSetDevice(device));
    }

    template<typename Derived1, typename Derived2>
    void initializeTestProblem( const Eigen::MatrixBase<Derived2> &T, const Eigen::MatrixBase<Derived1> & x_old,const Eigen::MatrixBase<Derived1> & d) {

        CHECK_CUBLAS(cublasCreate(&m_cublasHandle));
        CHECK_CUDA(cudaEventCreate(&m_startKernel));
        CHECK_CUDA(cudaEventCreate(&m_stopKernel));
        CHECK_CUDA(cudaEventCreate(&m_startCopy));
        CHECK_CUDA(cudaEventCreate(&m_stopCopy));

        size_t freeMem;
        size_t totalMem;
        CHECK_CUDA(cudaMemGetInfo (&freeMem, &totalMem));

        size_t nGPUBytes = (3*x_old.rows() + d.rows() + x_old.rows()/TConvexSet::Dimension +  T.rows()*T.cols())*sizeof(PREC);
        *m_pLog << "Will try to allocate ("<<nGPUBytes<<"/"<<freeMem<<") = " << (double)nGPUBytes/freeMem * 100.0 <<" % of global memory on GPU, Total mem: "<<totalMem<<std::endl;
        if(nGPUBytes > freeMem) {
            *m_pLog <<"Probably to little memory on GPU, try anway!..."<<std::endl;
        }

        cudaError_t error;

        CHECK_CUDA(utilCuda::releaseAndMallocMatrixDevice<alignMatrix>(T_dev, T));
        CHECK_CUDA(utilCuda::releaseAndMallocMatrixDevice<false>(d_dev, d));
        CHECK_CUDA(utilCuda::releaseAndMallocMatrixDevice<false>(x_old_dev,x_old));
        CHECK_CUDA(utilCuda::releaseAndMallocMatrixDevice<false>(t_dev,x_old));

        if(TypeTraitsHelper::IsSame<TConvexSet,ConvexSets::RPlusAndDisk>::result) {
            ASSERTMSG(x_old.rows()%TConvexSet::Dimension==0,"Wrong Dimension!");
            CHECK_CUDA(utilCuda::releaseAndMallocMatrixDevice<false>(mu_dev, x_old.rows()/TConvexSet::Dimension, x_old.cols()));
        } else if(TypeTraitsHelper::IsSame<TConvexSet,ConvexSets::RPlusAndContensouEllipsoid>::result) {
            ASSERTMSG(false, "NOT IMPLEMENTED!");
        }

        CHECK_CUDA((utilCuda::releaseAndMallocMatrixDevice<false>(x_new_dev,x_old_dev.m_M,x_old_dev.m_N)));

        if(pConvergedFlag_dev) {
            CHECK_CUDA(cudaFree(pConvergedFlag_dev));
            pConvergedFlag_dev = NULL;
        }
        CHECK_CUDA(cudaMalloc(&pConvergedFlag_dev,sizeof(bool)));

    }


    template<typename Derived1, typename Derived2,typename Derived3>
    void runCPUEquivalentProfile(Eigen::MatrixBase<Derived1> & x_newCPU, const Eigen::MatrixBase<Derived2> &T, Eigen::MatrixBase<Derived1> & x_old, const Eigen::MatrixBase<Derived1> & d, const Eigen::MatrixBase<Derived3> & mu) {

        // Do Sor Scheme!
        static const int ProxPackageSize = TConvexSet::Dimension;

        Derived1  x_temp;
        bool m_bConverged = false;

        // Do SOR Scheme!
        START_TIMER(start)

        if(VariantId < 5 ) {

            unsigned int loops =  T.cols() / ProxPackageSize;

            if(bMatchCPUToGPU) {
                // IF we match the GPU we implement the identical algorithm on the CPU!
                for(m_nIterCPU=0; m_nIterCPU< m_nMaxIterations ; m_nIterCPU++) {

                    if(bAbortIfConverged) {
                        x_temp.noalias() = x_old;
                    }

                    for(unsigned int i=0; i < loops; i++) {
                        // Normal
                        x_old((ProxPackageSize)*i) = (T.row((ProxPackageSize)*i) * x_old) + d((ProxPackageSize)*i);
                        Prox::ProxFunction<ConvexSets::RPlus>::doProxSingle(x_old((ProxPackageSize)*i));
                        // Tangential
                        x_old.template segment<ProxPackageSize-1 >((ProxPackageSize)*i+1)  = (T.block((ProxPackageSize)*i+1,0,ProxPackageSize-1,T.cols()) * x_old) + d.template segment<ProxPackageSize-1 >((ProxPackageSize)*i+1) ;
                        Prox::ProxFunction<ConvexSets::Disk>::doProxSingle(mu(i) *  x_old((ProxPackageSize)*i) , x_old.template segment<ProxPackageSize-1 >((ProxPackageSize)*i+1) );
                    }

                    if(bAbortIfConverged) {
                        // Check each nCheckConvergedFlag  the converged flag
                        if(m_nIterCPU % nCheckConvergedFlag == 0) {
                            //Calculate CancelCriteria
                            m_bConverged = Numerics::cancelCriteriaValue(x_old,x_temp, m_absTOL, m_relTOL);
                            if (m_bConverged == true) {
                                break;
                            }
                        }
                    }

                }
            } else {
                for(m_nIterCPU=0; m_nIterCPU< m_nMaxIterations ; m_nIterCPU++) {

                    if(bAbortIfConverged) {
                        x_temp.noalias() = x_old;
                    }

                    for(unsigned int i=0; i < loops; i++) {
                        // Normal and Tangential together ( is not the same as the SOR scheme on the GPU, but they converge to the same!)
                        x_old.template segment<ProxPackageSize>((ProxPackageSize)*i)  = (T.block((ProxPackageSize)*i,0,ProxPackageSize,T.cols()) * x_old) + d.template segment<ProxPackageSize>((ProxPackageSize)*i) ;
                        Prox::ProxFunction<ConvexSets::RPlusAndDisk>::doProxSingle(mu(i), x_old.template segment<ProxPackageSize>((ProxPackageSize)*i));
                    }

                    if(bAbortIfConverged) {
                        // Check each nCheckConvergedFlag  the converged flag
                        if(m_nIterCPU % nCheckConvergedFlag == 0) {
                            //Calculate CancelCriteria
                            m_bConverged = Numerics::cancelCriteriaValue(x_old,x_temp, m_absTOL, m_relTOL);
                            if (m_bConverged == true) {
                                break;
                            }
                        }
                    }

                }
            }

        } else if(VariantId == 5) {
            const int ProxPackages = SorProxSettings5::ProxPackages;
            const int BlockDimKernelA = SorProxSettings5::BlockDimKernelA;

            unsigned int loops =  (T.rows() + (SorProxSettings5::BlockDimKernelA-1)) / (SorProxSettings5::BlockDimKernelA);
            unsigned int i=0;
            unsigned int restSize,restContacts;

            for(m_nIterCPU=0; m_nIterCPU< m_nMaxIterations ; m_nIterCPU++) {

                if(bAbortIfConverged) {
                    x_temp.noalias() = x_old;
                }


                for(i = 0; i < loops-1; i++) {
                    // IF we match or not match the GPU, we implement the identical algorithm on the CPU!
                    x_old.template segment<BlockDimKernelA>((BlockDimKernelA)*i)  = (T.block((BlockDimKernelA)*i,0,BlockDimKernelA,T.cols()) * x_old) + d.template segment<BlockDimKernelA>((BlockDimKernelA)*i) ;
                    Prox::ProxFunction<ConvexSets::RPlusAndDisk>::doProxMulti(mu.template segment<ProxPackages>(ProxPackages*i), x_old.template segment<BlockDimKernelA>((BlockDimKernelA)*i));
                }
                // Do the rest
                restSize = T.cols() - (loops-1)*BlockDimKernelA;
                restContacts = restSize / ProxPackageSize;
                if(restSize !=0) {
                    x_old.segment((BlockDimKernelA)*i, restSize)  = (T.block((BlockDimKernelA)*i,0,restSize,T.cols()) * x_old) + d.segment((BlockDimKernelA)*i,restSize) ;
                    Prox::ProxFunction<ConvexSets::RPlusAndDisk>::doProxMulti(mu.segment(ProxPackages*i,restContacts), x_old.segment((BlockDimKernelA)*i,restSize));
                }

                if(bAbortIfConverged) {
                    // Check each nCheckConvergedFlag  the converged flag
                    if(m_nIterCPU % nCheckConvergedFlag == 0) {
                        //Calculate CancelCriteria
                        m_bConverged = Numerics::cancelCriteriaValue(x_old,x_temp, m_absTOL, m_relTOL);
                        if (m_bConverged == true) {
                            break;
                        }
                    }
                }

            }
        }

        // Do a swap!
        x_old.swap(x_newCPU.derived());

        STOP_TIMER_NANOSEC(count,start)

        m_cpuIterationTime = (count*1e-6 / m_nIterCPU);

        *m_pLog << " ---> CPU Sequential Iteration time: " <<  tinyformat::format("%1$8.6f ms", m_cpuIterationTime) <<std::endl;
        *m_pLog << " ---> nIterations: " << m_nIterCPU <<std::endl;
        if (m_nIterCPU == nMaxIterations) {
            *m_pLog << " ---> Not converged! Max. Iterations reached."<<std::endl;
        }
    }

    template<typename Derived1, typename Derived2,typename Derived3>
    void runCPUEquivalentPlain(Eigen::MatrixBase<Derived1> & x_newCPU, const Eigen::MatrixBase<Derived2> &T, Eigen::MatrixBase<Derived1> & x_old, const Eigen::MatrixBase<Derived1> & d, const Eigen::MatrixBase<Derived3> & mu) {

        // Do Sor Scheme!
        static const int ProxPackageSize = TConvexSet::Dimension;
        // Prox the contact

        //cout << typeid(Derived1).name() << endl << typeid(Derived1::PlainObject).name() <<endl;
        static typename Derived1::PlainObject  x_temp;

        // Do SOR Scheme!
        START_TIMER(start)

        bool m_bConverged = false;

        if(VariantId < 5 ) {

            unsigned int loops =  T.cols() / ProxPackageSize;

            if(bMatchCPUToGPU) {
                // IF we match the GPU we implement the identical algorithm on the CPU!
                for(m_nIterCPU=0; m_nIterCPU< m_nMaxIterations ; m_nIterCPU++) {

                    if(bAbortIfConverged) {
                        x_temp.noalias() = x_old;
                    }

                    for(unsigned int i=0; i < loops; i++) {
                        // Normal
                        x_old((ProxPackageSize)*i) = (T.row((ProxPackageSize)*i) * x_old) + d((ProxPackageSize)*i);
                        Prox::ProxFunction<ConvexSets::RPlus>::doProxSingle(x_old((ProxPackageSize)*i));
                        // Tangential
                        x_old.template segment<ProxPackageSize-1>((ProxPackageSize)*i+1)  = (T.template block((ProxPackageSize)*i+1,0,ProxPackageSize-1,T.cols()) * x_old) + d.template segment<ProxPackageSize-1>((ProxPackageSize)*i+1) ;
                        Prox::ProxFunction<ConvexSets::Disk>::doProxSingle(mu(i) *  x_old((ProxPackageSize)*i) , x_old.template segment<ProxPackageSize-1>((ProxPackageSize)*i+1) );
                    }

                    if(bAbortIfConverged) {
                        // Check each nCheckConvergedFlag  the converged flag
                        if(m_nIterCPU % nCheckConvergedFlag == 0) {
                            //Calculate CancelCriteria
                            m_bConverged = Numerics::cancelCriteriaValue(x_old,x_temp, m_absTOL, m_relTOL);
                            if (m_bConverged == true) {
                                break;
                            }
                        }
                    }

                }
            } else {
                for(m_nIterCPU=0; m_nIterCPU< m_nMaxIterations ; m_nIterCPU++) {

                    if(bAbortIfConverged) {
                        x_temp.noalias() = x_old;
                    }

                    for(unsigned int i=0; i < loops; i++) {
                        // Normal and Tangential together ( is not the same as the SOR scheme on the GPU, but the converge to the same!)
                        x_old.template segment<ProxPackageSize>((ProxPackageSize)*i)  = (T.template block((ProxPackageSize)*i,0,ProxPackageSize,T.cols()) * x_old) + d.template segment<ProxPackageSize>((ProxPackageSize)*i) ;
                        Prox::ProxFunction<ConvexSets::RPlusAndDisk>::doProxSingle(mu(i), x_old.template segment<ProxPackageSize>((ProxPackageSize)*i));
                    }

                    if(bAbortIfConverged) {
                        // Check each nCheckConvergedFlag  the converged flag
                        if(m_nIterCPU % nCheckConvergedFlag == 0) {
                            //Calculate CancelCriteria
                            m_bConverged = Numerics::cancelCriteriaValue(x_old,x_temp, m_absTOL, m_relTOL);
                            if (m_bConverged == true) {
                                break;
                            }
                        }
                    }

                }
            }

        } else if(VariantId == 5) {
            const int ProxPackages = SorProxSettings5::ProxPackages;
            const int BlockDimKernelA = SorProxSettings5::BlockDimKernelA;

            unsigned int loops =  (T.rows() + (SorProxSettings5::BlockDimKernelA-1)) / (SorProxSettings5::BlockDimKernelA);
            unsigned int i=0;
            unsigned int restSize,restContacts;

            for(m_nIterCPU=0; m_nIterCPU< m_nMaxIterations ; m_nIterCPU++) {

                if(bAbortIfConverged) {
                    x_temp.noalias() = x_old;
                }


                for(i = 0; i < loops-1; i++) {
                    // IF we match or not match the GPU, we implement the identical algorithm on the CPU!
                    x_old.template segment<BlockDimKernelA>((BlockDimKernelA)*i)  = (T.template block((BlockDimKernelA)*i,0,BlockDimKernelA,T.cols()) * x_old) + d.template segment<BlockDimKernelA>((BlockDimKernelA)*i) ;
                    Prox::ProxFunction<ConvexSets::RPlusAndDisk>::doProxMulti(mu.template segment<ProxPackages>(ProxPackages*i), x_old.template segment<BlockDimKernelA>((BlockDimKernelA)*i));
                }
                // Do the rest
                restSize = T.cols() - (loops-1)*BlockDimKernelA;
                restContacts = restSize / ProxPackageSize;
                if(restSize !=0) {
                    x_old.segment((BlockDimKernelA)*i, restSize)  = (T.block((BlockDimKernelA)*i,0,restSize,T.cols()) * x_old) + d.segment((BlockDimKernelA)*i,restSize) ;
                    Prox::ProxFunction<ConvexSets::RPlusAndDisk>::doProxMulti(mu.segment(ProxPackages*i,restContacts), x_old.segment((BlockDimKernelA)*i,restSize));
                }

                if(bAbortIfConverged) {
                    // Check each nCheckConvergedFlag  the converged flag
                    if(m_nIterCPU % nCheckConvergedFlag == 0) {
                        //Calculate CancelCriteria
                        m_bConverged = Numerics::cancelCriteriaValue(x_old,x_temp, m_absTOL, m_relTOL);
                        if (m_bConverged == true) {
                            break;
                        }
                    }
                }

            }
        } else if(VariantId == 11) {
            //std::cout<< "SOR VARIANT 11"<<std::endl;
            // This is a SOR Special where we iterate several time over one contact!
            // Just for test, has no implementation in GPU!!
            // IF we match the GPU we implement the identical algorithm on the CPU!
            unsigned int loops =  T.cols() / ProxPackageSize;

            for(m_nIterCPU=0; m_nIterCPU< m_nMaxIterations ; m_nIterCPU++) {

                if(bAbortIfConverged) {
                    x_temp.noalias() = x_old;
                }

                for(unsigned int i=0; i < loops; i++) {
                    // Internal loop!
                    for(unsigned int intLoop=0; intLoop < 5; intLoop++) {
                        // Normal
                        x_old((ProxPackageSize)*i) = (T.row((ProxPackageSize)*i) * x_old) + d((ProxPackageSize)*i);
                        Prox::ProxFunction<ConvexSets::RPlus>::doProxSingle(x_old((ProxPackageSize)*i));
                        // Tangential
                        x_old.template segment<ProxPackageSize-1>((ProxPackageSize)*i+1)  = (T.template block((ProxPackageSize)*i+1,0,ProxPackageSize-1,T.cols()) * x_old) + d.template segment<ProxPackageSize-1>((ProxPackageSize)*i+1) ;
                        Prox::ProxFunction<ConvexSets::Disk>::doProxSingle(mu(i) *  x_old((ProxPackageSize)*i) , x_old.template segment<ProxPackageSize-1>((ProxPackageSize)*i+1) );
                    }
                }

                if(bAbortIfConverged) {
                    // Check each nCheckConvergedFlag  the converged flag
                    if(m_nIterCPU % nCheckConvergedFlag == 0) {
                        //Calculate CancelCriteria
                        m_bConverged = Numerics::cancelCriteriaValue(x_old,x_temp, m_absTOL, m_relTOL);
                        if (m_bConverged == true) {
                            break;
                        }
                    }
                }

            }
        }

        // Do a swap!
        x_old.swap(x_newCPU.derived());

        STOP_TIMER_NANOSEC(count,start)
        m_cpuIterationTime = (count*1e-6 / m_nIterCPU);
    }


    template<typename Derived1, typename Derived2,typename Derived3>
    void runGPUProfile(Eigen::MatrixBase<Derived1> & x_newGPU, const Eigen::MatrixBase<Derived2> &T, const Eigen::MatrixBase<Derived1> & x_old, const Eigen::MatrixBase<Derived1> & d, const Eigen::MatrixBase<Derived3> & mu) {

        ASSERTMSG(x_newGPU.rows() == x_old.rows(), "Wrong Dimensions");
        ASSERTMSG(T.cols() == T.rows(), "Wrong Dimensions");
        ASSERTMSG(x_old.rows() == d.rows(), "Wrong Dimensions");
        ASSERTMSG(mu.rows() * TConvexSet::Dimension == x_old.rows(), mu.rows() * TConvexSet::Dimension << " , " << x_old.rows() << "Wrong Dimensions" );



        // Calulate t_dev and upload!
        m_t.resize(x_old.rows());
        m_t.setZero();


        //Copy Data
        CHECK_CUDA(cudaEventRecord(m_startCopy,0));
        CHECK_CUDA(copyMatrixToDevice(T_dev, T));
        CHECK_CUDA(copyMatrixToDevice(d_dev, d));
        CHECK_CUDA(copyMatrixToDevice(x_old_dev,x_old));
        CHECK_CUDA(copyMatrixToDevice(mu_dev,mu));
        CHECK_CUDA(copyMatrixToDevice(t_dev,m_t));
        CHECK_CUDA(cudaEventRecord(m_stopCopy,0));
        CHECK_CUDA(cudaEventSynchronize(m_stopCopy));



        float time;
        CHECK_CUDA( cudaEventElapsedTime(&time,m_startCopy,m_stopCopy));
        m_elapsedTimeCopyToGPU = time;
        *m_pLog << " ---> Copy time to GPU:"<< tinyformat::format("%8.6f ms" , time) <<std::endl;


        *m_pLog << " ---> Iterations started..."<<std::endl;

        m_absTOL = 1e-8;
        m_relTOL = 1e-10;

        CHECK_CUDA(cudaEventRecord(m_startKernel,0));

        runKernel();

        CHECK_CUDA(cudaEventRecord(m_stopKernel,0));
        CHECK_CUDA(cudaThreadSynchronize());

        CHECK_CUDA(cudaEventSynchronize(m_stopKernel));
        CHECK_CUDA(cudaGetLastError());


        *m_pLog<<" ---> Iterations finished" << std::endl;
        CHECK_CUDA( cudaEventElapsedTime(&time,m_startKernel,m_stopKernel));
        double average = (time/(double)m_nIterGPU);
        m_gpuIterationTime = average;

        *m_pLog << " ---> GPU Iteration time :"<< tinyformat::format("%8.6f ms",average) <<std::endl;
        *m_pLog << " ---> nIterations: " << m_nIterGPU <<std::endl;
        if (m_nIterGPU == nMaxIterations) {
            *m_pLog << " ---> Max. Iterations reached."<<std::endl;
        }

        //Sleep(500);

        // Copy results back
        CHECK_CUDA(cudaEventRecord(m_startCopy,0));
        CHECK_CUDA(copyMatrixToHost(x_newGPU,x_new_dev));
        CHECK_CUDA(cudaEventRecord(m_stopCopy,0));
        CHECK_CUDA(cudaEventSynchronize(m_stopCopy));
        CHECK_CUDA( cudaEventElapsedTime(&time,m_startCopy,m_stopCopy));
        m_elapsedTimeCopyFromGPU = time;
        *m_pLog << " ---> Copy time from GPU:"<< tinyformat::format("%8.6f ms", time) <<std::endl;
    }

    template<typename Derived1, typename Derived2, typename Derived3>
    bool runGPUPlain(Eigen::MatrixBase<Derived1> & x_newGPU, const Eigen::MatrixBase<Derived2> &T, const Eigen::MatrixBase<Derived1> & x_old, const Eigen::MatrixBase<Derived1> & d, const Eigen::MatrixBase<Derived3> & mu) {

        ASSERTMSG(x_newGPU.rows() == x_old.rows(), "Wrong Dimensions");
        ASSERTMSG(T.cols() == T.rows(), "Wrong Dimensions");
        ASSERTMSG(x_old.rows() == d.rows(), "Wrong Dimensions");
        ASSERTMSG(mu.rows() * TConvexSet::Dimension == x_old.rows(), mu.rows() * TConvexSet::Dimension << " , " << x_old.rows() << "Wrong Dimensions" );

        cudaError_t error;

        error = utilCuda::releaseAndMallocMatrixDevice<alignMatrix>(T_dev, T);
        CHECK_CUDA(error);
        error = utilCuda::releaseAndMallocMatrixDevice<false>(d_dev, d);
        CHECK_CUDA(error);
        error = utilCuda::releaseAndMallocMatrixDevice<false>(x_old_dev,x_old);
        CHECK_CUDA(error);
        error = utilCuda::releaseAndMallocMatrixDevice<false>(t_dev,x_old);
        CHECK_CUDA(error);

        if(TypeTraitsHelper::IsSame<TConvexSet,ConvexSets::RPlusAndDisk>::result) {
            ASSERTMSG(x_old.rows()%TConvexSet::Dimension==0,"Wrong Dimension!");
            error = utilCuda::releaseAndMallocMatrixDevice<false>(mu_dev, x_old.rows()/TConvexSet::Dimension, x_old.cols());
            CHECK_CUDA(error);
        } else if(TypeTraitsHelper::IsSame<TConvexSet,ConvexSets::RPlusAndContensouEllipsoid>::result) {
            ASSERTMSG(false, "NOT IMPLEMENTED!");
        }

        error = utilCuda::releaseAndMallocMatrixDevice<false>(x_new_dev,x_old_dev.m_M,x_old_dev.m_N);
        CHECK_CUDA(error);
        error = cudaMalloc(&pConvergedFlag_dev,sizeof(bool));
        CHECK_CUDA(error);


        // Calculate t_dev and upload!
        m_t.resize(x_old.rows());
        m_t.setZero();


        //Copy Data
        copyMatrixToDevice(T_dev, T);
        copyMatrixToDevice(d_dev, d);
        copyMatrixToDevice(x_old_dev,x_old);
        copyMatrixToDevice(mu_dev,mu);
        copyMatrixToDevice(t_dev,m_t);

        cudaEventCreate(&m_startKernel);
        cudaEventCreate(&m_stopKernel);

        cudaEventRecord(m_startKernel,0);
        runKernel();
        cudaEventRecord(m_stopKernel,0);


        cudaThreadSynchronize();

        float time;
        cudaEventElapsedTime(&time,m_startKernel,m_stopKernel);
        m_gpuIterationTime = (time/(double)m_nIterGPU);


        copyMatrixToHost(x_newGPU,x_new_dev);

        cudaEventDestroy(m_startKernel);
        cudaEventDestroy(m_stopKernel);

        return true;
    }

    // Partial Specialization of Member Function in template Class is not allowed.. Can do alternative struct wrapping, but this gets nasty!
    void runKernel() {

        if(VariantId == 1) {

            // Swap pointers of old and new on the device
            std::swap(x_old_dev.m_pDevice, x_new_dev.m_pDevice);

            int loops = (T_dev.m_M + ( SorProxSettings1::BlockDimKernelA -1 ) ) / SorProxSettings1::BlockDimKernelA ;

            for(m_nIterGPU=0; m_nIterGPU< m_nMaxIterations ; m_nIterGPU++) {

                // Swap pointers of old and new on the device
                std::swap(x_old_dev.m_pDevice, x_new_dev.m_pDevice);

                for(int kernelAIdx = 0; kernelAIdx < loops; kernelAIdx++) {

                    //cudaThreadSynchronize();
                    proxGPU::sorProxContactOrdered_1threads_StepA_kernelWrap<SorProxSettings1>(
                        mu_dev,x_new_dev,T_dev,d_dev,
                        t_dev,
                        kernelAIdx,
                        pConvergedFlag_dev,
                        m_absTOL,m_relTOL);

                    proxGPU::sorProx_StepB_kernelWrap<SorProxSettings1>(
                        t_dev,
                        T_dev,
                        x_new_dev,
                        kernelAIdx
                    );

                }


                if(bAbortIfConverged) {
                    // Check each nCheckConvergedFlag  the converged flag
                    if(m_nIterGPU % nCheckConvergedFlag == 0) {
                        // Check convergence
                        // First set the converged flag to 1
                        cudaMemset(pConvergedFlag_dev,1,sizeof(bool));
                        proxGPU::convergedEach_kernelWrap(x_new_dev,x_old_dev,pConvergedFlag_dev,m_absTOL,m_relTOL);

                        // Download the flag from the GPU and check
                        cudaMemcpy(&m_bConvergedFlag,pConvergedFlag_dev,sizeof(bool),cudaMemcpyDeviceToHost);
                        if(m_bConvergedFlag == true) {
                            // Converged
                            break;
                        }
                    }
                }


            }
        } else if(VariantId == 2) {

            // Swap pointers of old and new on the device
            std::swap(x_old_dev.m_pDevice, x_new_dev.m_pDevice);

            int loops = (T_dev.m_M + ( SorProxSettings2::BlockDimKernelA -1 ) ) / SorProxSettings2::BlockDimKernelA ;

            for(m_nIterGPU=0; m_nIterGPU< m_nMaxIterations ; m_nIterGPU++) {

                // Swap pointers of old and new on the device
                std::swap(x_old_dev.m_pDevice, x_new_dev.m_pDevice);

                for(int kernelAIdx = 0; kernelAIdx < loops; kernelAIdx++) {

                    proxGPU::sorProxContactOrdered_1threads_StepA_kernelWrap<SorProxSettings2>(
                        mu_dev,x_new_dev,T_dev,d_dev,
                        t_dev,
                        kernelAIdx,
                        pConvergedFlag_dev,
                        m_absTOL,m_relTOL);

                    proxGPU::sorProx_StepB_kernelWrap<SorProxSettings2>(
                        t_dev,
                        T_dev,
                        x_new_dev,
                        kernelAIdx
                    );
                }


                if(bAbortIfConverged) {
                    // Check each nCheckConvergedFlag  the converged flag
                    if(m_nIterGPU % nCheckConvergedFlag == 0) {
                        // First set the converged flag to 1
                        cudaMemset(pConvergedFlag_dev,1,sizeof(bool));
                        // Check convergence
                        proxGPU::convergedEach_kernelWrap(x_new_dev,x_old_dev,pConvergedFlag_dev,m_absTOL,m_relTOL);

                        // Download the flag from the GPU and check
                        cudaMemcpy(&m_bConvergedFlag,pConvergedFlag_dev,sizeof(bool),cudaMemcpyDeviceToHost);
                        if(m_bConvergedFlag == true) {
                            // Converged
                            break;
                        }
                    }
                }
            }
        } else if(VariantId == 3) {

            // Swap pointers of old and new on the device
            std::swap(x_old_dev.m_pDevice, x_new_dev.m_pDevice);

            int loops = (T_dev.m_M + ( SorProxSettings3::BlockDimKernelA -1 ) ) / SorProxSettings3::BlockDimKernelA ;
            for(m_nIterGPU=0; m_nIterGPU< m_nMaxIterations ; m_nIterGPU++) {

                // Swap pointers of old and new on the device
                std::swap(x_old_dev.m_pDevice, x_new_dev.m_pDevice);

                for(int kernelAIdx = 0; kernelAIdx < loops; kernelAIdx++) {

                    proxGPU::sorProxContactOrdered_1threads_StepA_kernelWrap<SorProxSettings3>(
                        mu_dev,x_new_dev,T_dev,d_dev,
                        t_dev,
                        kernelAIdx,
                        pConvergedFlag_dev,
                        m_absTOL,m_relTOL);

                    /*  proxGPU::sorProx_StepB_kernelWrap<SorProxSettings3>(
                         t_dev,
                         T_dev,
                         x_new_dev,
                         kernelAIdx
                         );*/
                }
            }
        } else if(VariantId == 4) {

            // Swap pointers of old and new on the device
            std::swap(x_old_dev.m_pDevice, x_new_dev.m_pDevice);

            int loops = (T_dev.m_M + ( SorProxSettings4::BlockDimKernelA -1 ) ) / SorProxSettings4::BlockDimKernelA ;

            for(m_nIterGPU=0; m_nIterGPU< m_nMaxIterations ; m_nIterGPU++) {

                // Swap pointers of old and new on the device
                std::swap(x_old_dev.m_pDevice, x_new_dev.m_pDevice);

                for(int kernelAIdx = 0; kernelAIdx < loops; kernelAIdx++) {

                    /* proxGPU::sorProxContactOrdered_1threads_StepA_kernelWrap<SorProxSettings4>(
                        mu_dev,x_new_dev,T_dev,x_old_dev,d_dev,
                        t_dev,
                        kernelAIdx,
                        pConvergedFlag_dev,
                        m_absTOL,m_relTOL);*/

                    proxGPU::sorProx_StepB_kernelWrap<SorProxSettings4>(
                        t_dev,
                        T_dev,
                        x_new_dev,
                        kernelAIdx
                    );
                }
            }
        } else if(VariantId == 5) {

            // Swap pointers of old and new on the device
            std::swap(x_old_dev.m_pDevice, x_new_dev.m_pDevice);

            int loops = (T_dev.m_M + ( SorProxSettings5::BlockDimKernelA -1 ) ) / SorProxSettings5::BlockDimKernelA ;

            for(m_nIterGPU=0; m_nIterGPU< m_nMaxIterations ; m_nIterGPU++) {

                // Swap pointers of old and new on the device
                std::swap(x_old_dev.m_pDevice, x_new_dev.m_pDevice);

                for(int kernelAIdx = 0; kernelAIdx < loops; kernelAIdx++) {

                    proxGPU::sorProxRelaxed_StepA_kernelWrap<SorProxSettings5>(
                        mu_dev,
                        x_new_dev,
                        t_dev,
                        d_dev,
                        kernelAIdx
                    );

                    proxGPU::sorProxRelaxed_StepB_kernelWrap<SorProxSettings5>(
                        t_dev,
                        T_dev,
                        x_new_dev,
                        kernelAIdx
                    );
                }


                if(bAbortIfConverged) {
                    // Check each nCheckConvergedFlag  the converged flag
                    if(m_nIterGPU % nCheckConvergedFlag == 0) {
                        // First set the converged flag to 1
                        cudaMemset(pConvergedFlag_dev,1,sizeof(bool));

                        // Check convergence
                        proxGPU::convergedEach_kernelWrap(x_new_dev,x_old_dev,pConvergedFlag_dev,m_absTOL,m_relTOL);

                        // Download the flag from the GPU and check
                        cudaMemcpy(&m_bConvergedFlag,pConvergedFlag_dev,sizeof(bool),cudaMemcpyDeviceToHost);
                        if(m_bConvergedFlag == true) {
                            // Converged
                            break;
                        }
                    }
                }
            }
        }

    }



    void cleanUpTestProblem() {

        CHECK_CUDA(freeMatrixDevice(T_dev));
        CHECK_CUDA(freeMatrixDevice(d_dev));
        CHECK_CUDA(freeMatrixDevice(x_old_dev));
        CHECK_CUDA(freeMatrixDevice(x_new_dev));
        CHECK_CUDA(freeMatrixDevice(t_dev));
        CHECK_CUDA(freeMatrixDevice(mu_dev));
        if(pConvergedFlag_dev) {
            CHECK_CUDA(cudaFree(pConvergedFlag_dev));
            pConvergedFlag_dev = NULL;
        }

        CHECK_CUDA(cudaEventDestroy(m_startKernel));
        CHECK_CUDA(cudaEventDestroy(m_stopKernel));
        CHECK_CUDA(cudaEventDestroy(m_startCopy));
        CHECK_CUDA(cudaEventDestroy(m_stopCopy));

    }

    double m_gpuIterationTime;
    double m_elapsedTimeCopyToGPU;
    double m_elapsedTimeCopyFromGPU;
    int m_nIterGPU;

    double m_cpuIterationTime;
    int m_nIterCPU;

private:
    Eigen::Matrix<PREC,Eigen::Dynamic,1> m_t;
    utilCuda::CudaMatrix<PREC> T_dev, d_dev, x_old_dev, x_new_dev, mu_dev, t_dev;
    bool *pConvergedFlag_dev;
    bool m_bConvergedFlag;

    unsigned int m_nMaxIterations;
    PREC m_absTOL, m_relTOL;

    cublasHandle_t m_cublasHandle;
    cudaEvent_t m_startKernel, m_stopKernel, m_startCopy,m_stopCopy;

    std::ostream* m_pLog;
    std::ofstream* m_pMatlab_file;
    Eigen::IOFormat Matlab;
};




/** @} */


#endif
