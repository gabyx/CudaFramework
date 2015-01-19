

#include <algorithm>
#include <cuda_runtime.h>

#include "CudaFramework/Kernels/ProxGPU/KernelsProx.cuh"

#include "CudaFramework/CudaModern/CudaMatrix.hpp"
#include "CudaFramework/General/AssertionDebugC.hpp"
#include "CudaFramework/General/StaticAssert.hpp"
#include "ConvexSets.hpp"
#include "CudaFramework/Kernels/ProxGPU/ProxKernelSettings.hpp"

namespace proxGPU{

   using namespace utilCuda;
   using namespace proxKernels;

   template<typename TCudaMatrix>
   __host__ void proxContactOrdered_RPlusAndDisk_2threads_kernelWrap(TCudaMatrix & mu_dev,
                                                                     TCudaMatrix & proxTerm_dev){


      ASSERTMSG_C(mu_dev.m_M *3 == proxTerm_dev.m_M, "Wrong dimensions!");
      dim3 threads(256);
      int blockdim = (proxTerm_dev.m_M + (threads.x-1) )  / threads.x;
	   dim3 blocks(blockdim);
	   proxContactOrdered_RPlusAndDisk_2threads_kernel<<< blocks, threads >>>(mu_dev,proxTerm_dev);
   }


   template<typename TConvexSet , typename TCudaMatrix>
   __host__ void proxContactOrdered_1threads_kernelWrap(TCudaMatrix & mu_dev,
                                                        TCudaMatrix & proxTerm_dev){

      STATIC_ASSERTM ((TypeTraitsHelper::IsSame<TConvexSet,ConvexSets::RPlusAndDisk>::result), ONLY_RPLUS_AND_DISK_IS_IMPLEMENTED_SO_FAR);
      static const int ProxPackageSize = TConvexSet::Dimension;


      ASSERTMSG_C(mu_dev.m_M *ProxPackageSize == proxTerm_dev.m_M, "Wrong dimensions!");

	   dim3 blocks;
      dim3 threads(128);
      blocks.x = (proxTerm_dev.m_M / ProxPackageSize + (threads.x-1) )  / threads.x;
	   proxContactOrdered_1threads_kernel<TCudaMatrix,TConvexSet><<< blocks, threads >>>(mu_dev,proxTerm_dev);

   }

    template<typename TConvexSet , typename TCudaMatrix>
   __host__ void proxContactOrderedWORSTCASE_1threads_kernelWrap(TCudaMatrix & mu_dev,
                                                                 TCudaMatrix & proxTerm_dev){

      STATIC_ASSERTM ((TypeTraitsHelper::IsSame<TConvexSet,ConvexSets::RPlusAndDisk>::result), ONLY_RPLUS_AND_DISK_IS_IMPLEMENTED_SO_FAR);
      static const int ProxPackageSize = TConvexSet::Dimension;


      ASSERTMSG_C(mu_dev.m_M *ProxPackageSize == proxTerm_dev.m_M, "Wrong dimensions!");

	   dim3 blocks;
      dim3 threads(128);
      blocks.x = (proxTerm_dev.m_M / ProxPackageSize + (threads.x-1) )  / threads.x;
	   proxContactOrderedWORSTCASE_1threads_kernel<TCudaMatrix,TConvexSet><<< blocks, threads >>>(mu_dev,proxTerm_dev);

   }


   template<typename TConvexSet , typename TCudaMatrix>
   __host__ void proxContactOrdered_1threads_kernelWrap(TCudaMatrix & mu_dev,
                                                        TCudaMatrix & proxTerm_dev,
                                                        TCudaMatrix & d_dev ){

      STATIC_ASSERTM( (TypeTraitsHelper::IsSame<TConvexSet,ConvexSets::RPlusAndDisk>::result), ONLY_RPLUS_AND_DISK_IS_IMPLEMENTED_SO_FAR)
      static const int ProxPackageSize = TConvexSet::Dimension;

      ASSERTMSG_C(mu_dev.m_M *ProxPackageSize == proxTerm_dev.m_M, "Wrong dimensions!");

      dim3 blocks;
      dim3 threads(256);
      blocks.x = (proxTerm_dev.m_M / ProxPackageSize + (threads.x-1) )  / threads.x;

	   proxContactOrdered_1threads_kernel<TCudaMatrix,TConvexSet><<< blocks, threads >>>(mu_dev,proxTerm_dev, d_dev);

   }

   template<typename TJorProxKernelSettings , typename TCudaMatrix>
   __host__ void jorProxContactOrdered_1threads_kernelWrap( TCudaMatrix & mu_dev,
                                                            TCudaMatrix & y_dev,
                                                            typename TCudaMatrix::PREC alpha,
                                                            TCudaMatrix & A_dev,
                                                            TCudaMatrix & x_dev,
                                                            typename TCudaMatrix::PREC beta,
                                                            TCudaMatrix & b_dev)
   {
      ASSERTMSG_C(A_dev.m_N == x_dev.m_M && y_dev.m_M == A_dev.m_M && b_dev.m_M == A_dev.m_M && A_dev.m_M / TJorProxKernelSettings::ProxPackageSize == mu_dev.m_M, "Matrices have wrong dimensions");
      ASSERTMSG_C(A_dev.m_M % TJorProxKernelSettings::ProxPackageSize == 0, "Vector length not a multiple of ProxPackageSize!");

      STATIC_ASSERTM( (TypeTraitsHelper::IsSame< typename TJorProxKernelSettings::TConvexSet, ConvexSets::RPlusAndDisk>::result), ONLY_RPLUS_AND_DISK_IS_IMPLEMENTED_SO_FAR )

      // Launch kernel ===========================================================
      dim3 blocks;
      blocks.x = (A_dev.m_M + (TJorProxKernelSettings::BlockDim-1)) / TJorProxKernelSettings::BlockDim; // or another blockDim

      jorProxContactOrdered_1threads_kernel<
         TCudaMatrix,
         TJorProxKernelSettings::ThreadsPerBlock,
        TJorProxKernelSettings::BlockDim,
        TJorProxKernelSettings::XElementsPerThread,
        TJorProxKernelSettings::ProxPackages,
          TJorProxKernelSettings::UnrollBlockDotProduct,
         typename TJorProxKernelSettings::TConvexSet
         >
      <<< blocks, TJorProxKernelSettings::ThreadsPerBlock >>>(mu_dev, y_dev, 1, alpha, A_dev, x_dev, 1, beta, b_dev, 1);

   }

    template<typename TSorProxKernelSettings , typename TCudaMatrix>
   __host__ void sorProxContactOrdered_1threads_StepA_kernelWrap( TCudaMatrix & mu_dev,
                                                            TCudaMatrix & x_new_dev,
                                                            TCudaMatrix & T_dev,
                                                            TCudaMatrix & d_dev,
                                                            TCudaMatrix & t_dev,
                                                            int kernelAIdx,
                                                            bool * convergedFlag_dev,
                                                            typename TCudaMatrix::PREC absTOL, typename TCudaMatrix::PREC relTOL
                                                            )
   {
      ASSERTMSG_C(d_dev.m_M == T_dev.m_M && x_new_dev.m_M == d_dev.m_M && t_dev.m_M == x_new_dev.m_M && T_dev.m_M / TSorProxKernelSettings::ProxPackageSize == mu_dev.m_M, "Matrices have wrong dimensions");
      ASSERTMSG_C(T_dev.m_M % TSorProxKernelSettings::ProxPackageSize == 0, "Vector length not a multiple of ProxPackageSize!");

      STATIC_ASSERTM( (TypeTraitsHelper::IsSame< typename TSorProxKernelSettings::TConvexSet, ConvexSets::RPlusAndDisk>::result), ONLY_RPLUS_AND_DISK_IS_IMPLEMENTED_SO_FAR )

      dim3 blocks;

      int maxNPackages = std::min(
                     (int) TSorProxKernelSettings::ProxPackages,
                     (int)((T_dev.m_M /TSorProxKernelSettings::ProxPackageSize)  -  (kernelAIdx) * (TSorProxKernelSettings::ProxPackages))
                  );

      blocks.x = 1; // Only one blocks gets launched
      sorProxContactOrdered_1threads_StepA_kernel<
         TCudaMatrix,
         TSorProxKernelSettings::ThreadsPerBlockKernelA,
         TSorProxKernelSettings::BlockDimKernelA,
         TSorProxKernelSettings::ProxPackages,
         typename TSorProxKernelSettings::TConvexSet>
      <<< blocks, TSorProxKernelSettings::ThreadsPerBlockKernelA >>>(mu_dev,x_new_dev, T_dev, d_dev, t_dev, kernelAIdx, maxNPackages,convergedFlag_dev,absTOL,relTOL);

   }

    template<typename TSorProxKernelSettings , typename TCudaMatrix>
   __host__ void sorProx_StepB_kernelWrap(   TCudaMatrix & t_dev,
                                             TCudaMatrix & T_dev,
                                             TCudaMatrix & x_new_dev,
                                             int kernelAIdx
                                             )
   {
      ASSERTMSG_C(t_dev.m_M == x_new_dev.m_M , "Matrices have wrong dimensions");
      ASSERTMSG_C(T_dev.m_M % TSorProxKernelSettings::ProxPackageSize == 0, "Vector length not a multiple of ProxPackageSize!");


      dim3 blocks;
      int maxNPackages = std::min(
                     (int) TSorProxKernelSettings::ProxPackages,
                     (int)((T_dev.m_M /TSorProxKernelSettings::ProxPackageSize)  -  (kernelAIdx) * (TSorProxKernelSettings::ProxPackages))
                  );
      int propagatingRows = (T_dev.m_M - maxNPackages*TSorProxKernelSettings::ProxPackageSize);
      blocks.x = (propagatingRows + TSorProxKernelSettings::BlockDimKernelB - 1) / TSorProxKernelSettings::BlockDimKernelB ; // Only one blocks gets launched

      if(blocks.x <= 0){
         return;
      }

      sorProx_StepB_kernel<
         TCudaMatrix,
         TSorProxKernelSettings::ThreadsPerBlockKernelB,
         TSorProxKernelSettings::BlockDimKernelB,
         TSorProxKernelSettings::BlockDimKernelA,
         TSorProxKernelSettings::XElementsPerThreadKernelB,
         TSorProxKernelSettings::UnrollBlockDotProductKernelB
      >
      <<< blocks, TSorProxKernelSettings::ThreadsPerBlockKernelB >>>(t_dev, T_dev, x_new_dev, kernelAIdx);

   }


   template<typename TRelaxedSorProxKernelSettings , typename TCudaMatrix>
   __host__ void sorProxRelaxed_StepA_kernelWrap(  TCudaMatrix & mu_dev,
                                                   TCudaMatrix & x_new_dev,
                                                   TCudaMatrix & t_dev,
                                                   TCudaMatrix & d_dev,
                                                   int kernelAIdx){

      STATIC_ASSERTM( (TypeTraitsHelper::IsSame< typename TRelaxedSorProxKernelSettings::TConvexSet , ConvexSets::RPlusAndDisk>::result), ONLY_RPLUS_AND_DISK_IS_IMPLEMENTED_SO_FAR)

      ASSERTMSG_C(mu_dev.m_M * TRelaxedSorProxKernelSettings::ProxPackageSize == x_new_dev.m_M, "Wrong dimensions!");
      ASSERTMSG_C(x_new_dev.m_M  == t_dev.m_M, "Wrong dimensions!");
      ASSERTMSG_C(d_dev.m_M  == t_dev.m_M, "Wrong dimensions!");


	   sorProxRelaxed_StepA_kernel<
         TCudaMatrix,
         TRelaxedSorProxKernelSettings::ThreadsPerBlockKernelA,
         TRelaxedSorProxKernelSettings::BlockDimKernelA,
         TRelaxedSorProxKernelSettings::ProxPackages,
         typename TRelaxedSorProxKernelSettings::TConvexSet
      >
      <<< (TRelaxedSorProxKernelSettings::ProxPackages + ( TRelaxedSorProxKernelSettings::ThreadsPerBlockKernelA-1) )  /  TRelaxedSorProxKernelSettings::ThreadsPerBlockKernelA , TRelaxedSorProxKernelSettings::ThreadsPerBlockKernelA >>>
      (mu_dev, x_new_dev, t_dev, d_dev, kernelAIdx);

   }


    template<typename TRelaxedSorProxKernelSettings , typename TCudaMatrix>
   __host__ void sorProxRelaxed_StepB_kernelWrap(  TCudaMatrix & t_dev,
                                                   TCudaMatrix & T_dev,
                                                   TCudaMatrix & x_new_dev,
                                                   int kernelAIdx){
      ASSERTMSG_C(T_dev.m_N == x_new_dev.m_M &&  t_dev.m_M == x_new_dev.m_M , "Matrices have wrong dimensions");
      ASSERTMSG_C(T_dev.m_M % TRelaxedSorProxKernelSettings::ProxPackageSize == 0, "Vector length not a multiple of ProxPackageSize!");


      dim3 blocks;
      int propagatingRows = T_dev.m_M ;
      blocks.x = (propagatingRows + TRelaxedSorProxKernelSettings::BlockDimKernelB - 1) / TRelaxedSorProxKernelSettings::BlockDimKernelB ; // Only one blocks gets launched

      sorProxRelaxed_StepB_kernel<
         TCudaMatrix,
         TRelaxedSorProxKernelSettings::ThreadsPerBlockKernelB,
         TRelaxedSorProxKernelSettings::BlockDimKernelB,
         TRelaxedSorProxKernelSettings::BlockDimKernelA,
         TRelaxedSorProxKernelSettings::XElementsPerThreadKernelB,
         TRelaxedSorProxKernelSettings::UnrollBlockDotProductKernelB
      >
      <<< blocks, TRelaxedSorProxKernelSettings::ThreadsPerBlockKernelB >>>(t_dev, T_dev, x_new_dev, kernelAIdx);

   }



    template<typename TCudaMatrix>
   __host__ void convergedEach_kernelWrap(TCudaMatrix & x_new_dev,
                                          TCudaMatrix & x_old_dev,
                                          bool * convergedFlag_dev,
                                          typename TCudaMatrix::PREC absTOL,
                                          typename TCudaMatrix::PREC relTOL){
      dim3 threads(256);
      int blockdim = (x_old_dev.m_M + (threads.x-1) )  / threads.x;
	   dim3 blocks(blockdim);
	   convergedEach_kernel<<< blocks, threads >>>(x_new_dev,x_old_dev,convergedFlag_dev,absTOL,relTOL);
   }



// Explicit instantiation:
#define TCudaMatrix CudaMatrix<float,CudaMatrixFlags::ColMajor>

   template __host__ void proxContactOrdered_RPlusAndDisk_2threads_kernelWrap(TCudaMatrix & mu_dev, TCudaMatrix & proxTerm_dev);
   template __host__ void proxContactOrdered_1threads_kernelWrap<ConvexSets::RPlusAndDisk , TCudaMatrix>(TCudaMatrix & mu_dev, TCudaMatrix & proxTerm_dev);
   template __host__ void proxContactOrderedWORSTCASE_1threads_kernelWrap<ConvexSets::RPlusAndDisk , TCudaMatrix>(TCudaMatrix & mu_dev, TCudaMatrix & proxTerm_dev);
   template __host__ void proxContactOrdered_1threads_kernelWrap<ConvexSets::RPlusAndDisk , TCudaMatrix>(TCudaMatrix &mu_dev, TCudaMatrix &proxTerm_dev, TCudaMatrix &d_dev );

   template __host__ void jorProxContactOrdered_1threads_kernelWrap<JorProxSettings3RPlusAndDisk , TCudaMatrix>(TCudaMatrix &mu_dev,TCudaMatrix &y_dev,typename TCudaMatrix::PREC alpha, TCudaMatrix &A_dev,TCudaMatrix &x_dev, typename TCudaMatrix::PREC beta, TCudaMatrix &b_dev);
   template __host__ void jorProxContactOrdered_1threads_kernelWrap<JorProxSettings4RPlusAndDisk , TCudaMatrix>(TCudaMatrix &mu_dev,TCudaMatrix &y_dev,typename TCudaMatrix::PREC alpha, TCudaMatrix &A_dev,TCudaMatrix &x_dev, typename TCudaMatrix::PREC beta, TCudaMatrix &b_dev);

   template __host__ void sorProxContactOrdered_1threads_StepA_kernelWrap<SorProxSettings1RPlusAndDisk , TCudaMatrix>(TCudaMatrix &mu_dev,
                                                                                                                    TCudaMatrix &x_new_dev,
                                                                                                                    TCudaMatrix &T_dev,
                                                                                                                    TCudaMatrix &d_dev,
                                                                                                                    TCudaMatrix &t_dev,
                                                                                                                    int kernelAIdx,
                                                                                                                    bool * convergedFlag_dev,
                                                                                                                    typename TCudaMatrix::PREC absTOL,
                                                                                                                    typename TCudaMatrix::PREC relTOL);

   template __host__ void sorProx_StepB_kernelWrap<SorProxSettings1RPlusAndDisk , TCudaMatrix>( TCudaMatrix &t_dev, TCudaMatrix &T_dev,  TCudaMatrix &x_old_dev,int kernelAIdx);
   template __host__ void sorProxContactOrdered_1threads_StepA_kernelWrap<SorProxSettings2RPlusAndDisk , TCudaMatrix>(TCudaMatrix &mu_dev,
                                                                                                                    TCudaMatrix &x_new_dev ,
                                                                                                                    TCudaMatrix &T_dev,
                                                                                                                    TCudaMatrix &d_dev,
                                                                                                                    TCudaMatrix &t_dev,
                                                                                                                    int kernelAIdx,
                                                                                                                    bool * convergedFlag_dev,
                                                                                                                    typename TCudaMatrix::PREC absTOL,
                                                                                                                    typename TCudaMatrix::PREC relTOL);
   template __host__ void sorProx_StepB_kernelWrap<SorProxSettings2RPlusAndDisk , TCudaMatrix>( TCudaMatrix &t_dev, TCudaMatrix &T_dev,  TCudaMatrix &x_old_dev,int kernelAIdx);

   template __host__ void sorProxRelaxed_StepA_kernelWrap<RelaxedSorProxSettings1RPlusAndDisk  , TCudaMatrix>(TCudaMatrix &mu_dev, TCudaMatrix &x_new_dev, TCudaMatrix &t_dev, TCudaMatrix &d_dev, int kernelAIdx);
   template __host__ void sorProxRelaxed_StepB_kernelWrap<RelaxedSorProxSettings1RPlusAndDisk  , TCudaMatrix>(TCudaMatrix &t_dev,TCudaMatrix &T_dev,  TCudaMatrix &x_new_dev,int kernelAIdx);

   template __host__ void sorProxRelaxed_StepA_kernelWrap<RelaxedSorProxKernelSettings<1,ConvexSets::RPlusAndDisk>  , TCudaMatrix>(TCudaMatrix &mu_dev, TCudaMatrix &x_new_dev, TCudaMatrix &t_dev, TCudaMatrix &d_dev, int kernelAIdx);
   template __host__ void sorProxRelaxed_StepB_kernelWrap<RelaxedSorProxKernelSettings<1,ConvexSets::RPlusAndDisk>  , TCudaMatrix>(TCudaMatrix &t_dev,TCudaMatrix &T_dev,  TCudaMatrix &x_new_dev,int kernelAIdx);
   template __host__ void sorProxRelaxed_StepA_kernelWrap<RelaxedSorProxKernelSettings<2,ConvexSets::RPlusAndDisk>  , TCudaMatrix>(TCudaMatrix &mu_dev, TCudaMatrix &x_new_dev, TCudaMatrix &t_dev, TCudaMatrix &d_dev, int kernelAIdx);
   template __host__ void sorProxRelaxed_StepB_kernelWrap<RelaxedSorProxKernelSettings<2,ConvexSets::RPlusAndDisk>  , TCudaMatrix>(TCudaMatrix &t_dev,TCudaMatrix &T_dev,  TCudaMatrix &x_new_dev,int kernelAIdx);
   template __host__ void sorProxRelaxed_StepA_kernelWrap<RelaxedSorProxKernelSettings<4,ConvexSets::RPlusAndDisk>  , TCudaMatrix>(TCudaMatrix &mu_dev, TCudaMatrix &x_new_dev, TCudaMatrix &t_dev, TCudaMatrix &d_dev, int kernelAIdx);
   template __host__ void sorProxRelaxed_StepB_kernelWrap<RelaxedSorProxKernelSettings<4,ConvexSets::RPlusAndDisk>  , TCudaMatrix>(TCudaMatrix &t_dev,TCudaMatrix &T_dev,  TCudaMatrix &x_new_dev,int kernelAIdx);
   //template __host__ void sorProxRelaxed_StepA_kernelWrap<RelaxedSorProxKernelSettings<8,ConvexSets::RPlusAndDisk>  , TCudaMatrix>(TCudaMatrix &mu_dev, TCudaMatrix &x_new_dev, TCudaMatrix &t_dev, TCudaMatrix &d_dev, int kernelAIdx);
   //template __host__ void sorProxRelaxed_StepB_kernelWrap<RelaxedSorProxKernelSettings<8,ConvexSets::RPlusAndDisk>  , TCudaMatrix>(TCudaMatrix &t_dev,TCudaMatrix &T_dev,  TCudaMatrix &x_new_dev,int kernelAIdx);
   template __host__ void sorProxRelaxed_StepA_kernelWrap<RelaxedSorProxKernelSettings<16,ConvexSets::RPlusAndDisk>  , TCudaMatrix>(TCudaMatrix &mu_dev, TCudaMatrix &x_new_dev, TCudaMatrix &t_dev, TCudaMatrix &d_dev, int kernelAIdx);
   template __host__ void sorProxRelaxed_StepB_kernelWrap<RelaxedSorProxKernelSettings<16,ConvexSets::RPlusAndDisk>  , TCudaMatrix>(TCudaMatrix &t_dev,TCudaMatrix &T_dev,  TCudaMatrix &x_new_dev,int kernelAIdx);
   template __host__ void sorProxRelaxed_StepA_kernelWrap<RelaxedSorProxKernelSettings<32,ConvexSets::RPlusAndDisk>  , TCudaMatrix>(TCudaMatrix &mu_dev, TCudaMatrix &x_new_dev, TCudaMatrix &t_dev, TCudaMatrix &d_dev, int kernelAIdx);
   template __host__ void sorProxRelaxed_StepB_kernelWrap<RelaxedSorProxKernelSettings<32,ConvexSets::RPlusAndDisk>  , TCudaMatrix>(TCudaMatrix &t_dev,TCudaMatrix &T_dev,  TCudaMatrix &x_new_dev,int kernelAIdx);
   template __host__ void sorProxRelaxed_StepA_kernelWrap<RelaxedSorProxKernelSettings<64,ConvexSets::RPlusAndDisk>  , TCudaMatrix>(TCudaMatrix &mu_dev, TCudaMatrix &x_new_dev, TCudaMatrix &t_dev, TCudaMatrix &d_dev, int kernelAIdx);
   template __host__ void sorProxRelaxed_StepB_kernelWrap<RelaxedSorProxKernelSettings<64,ConvexSets::RPlusAndDisk>  , TCudaMatrix>(TCudaMatrix &t_dev,TCudaMatrix &T_dev,  TCudaMatrix &x_new_dev,int kernelAIdx);
   template __host__ void sorProxRelaxed_StepA_kernelWrap<RelaxedSorProxKernelSettings<128,ConvexSets::RPlusAndDisk>  , TCudaMatrix>(TCudaMatrix &mu_dev, TCudaMatrix &x_new_dev, TCudaMatrix &t_dev, TCudaMatrix &d_dev, int kernelAIdx);
   template __host__ void sorProxRelaxed_StepB_kernelWrap<RelaxedSorProxKernelSettings<128,ConvexSets::RPlusAndDisk>  , TCudaMatrix>(TCudaMatrix &t_dev,TCudaMatrix &T_dev,  TCudaMatrix &x_new_dev,int kernelAIdx);

   template __host__ void convergedEach_kernelWrap(TCudaMatrix &x_new_dev, TCudaMatrix &x_old_dev, bool * convergedFlag_dev, typename TCudaMatrix::PREC absTOL, typename TCudaMatrix::PREC relTOL);


#undef TCudaMatrix


#define TCudaMatrix CudaMatrix<double,CudaMatrixFlags::ColMajor>

 template __host__ void proxContactOrdered_RPlusAndDisk_2threads_kernelWrap(TCudaMatrix & mu_dev, TCudaMatrix & proxTerm_dev);
   template __host__ void proxContactOrdered_1threads_kernelWrap<ConvexSets::RPlusAndDisk , TCudaMatrix>(TCudaMatrix & mu_dev, TCudaMatrix & proxTerm_dev);
   template __host__ void proxContactOrderedWORSTCASE_1threads_kernelWrap<ConvexSets::RPlusAndDisk , TCudaMatrix>(TCudaMatrix & mu_dev, TCudaMatrix & proxTerm_dev);
   template __host__ void proxContactOrdered_1threads_kernelWrap<ConvexSets::RPlusAndDisk , TCudaMatrix>(TCudaMatrix &mu_dev, TCudaMatrix &proxTerm_dev, TCudaMatrix &d_dev );

   template __host__ void jorProxContactOrdered_1threads_kernelWrap<JorProxSettings3RPlusAndDisk , TCudaMatrix>(TCudaMatrix &mu_dev,TCudaMatrix &y_dev,typename TCudaMatrix::PREC alpha, TCudaMatrix &A_dev,TCudaMatrix &x_dev, typename TCudaMatrix::PREC beta, TCudaMatrix &b_dev);
   template __host__ void jorProxContactOrdered_1threads_kernelWrap<JorProxSettings4RPlusAndDisk , TCudaMatrix>(TCudaMatrix &mu_dev,TCudaMatrix &y_dev,typename TCudaMatrix::PREC alpha, TCudaMatrix &A_dev,TCudaMatrix &x_dev, typename TCudaMatrix::PREC beta, TCudaMatrix &b_dev);

   template __host__ void sorProxContactOrdered_1threads_StepA_kernelWrap<SorProxSettings1RPlusAndDisk , TCudaMatrix>(TCudaMatrix &mu_dev,
                                                                                                                    TCudaMatrix &x_new_dev,
                                                                                                                    TCudaMatrix &T_dev,
                                                                                                                    TCudaMatrix &d_dev,
                                                                                                                    TCudaMatrix &t_dev,
                                                                                                                    int kernelAIdx,
                                                                                                                    bool * convergedFlag_dev,
                                                                                                                    typename TCudaMatrix::PREC absTOL,
                                                                                                                    typename TCudaMatrix::PREC relTOL);

   template __host__ void sorProx_StepB_kernelWrap<SorProxSettings1RPlusAndDisk , TCudaMatrix>( TCudaMatrix &t_dev, TCudaMatrix &T_dev,  TCudaMatrix &x_old_dev,int kernelAIdx);
   template __host__ void sorProxContactOrdered_1threads_StepA_kernelWrap<SorProxSettings2RPlusAndDisk , TCudaMatrix>(TCudaMatrix &mu_dev,
                                                                                                                    TCudaMatrix &x_new_dev ,
                                                                                                                    TCudaMatrix &T_dev,
                                                                                                                    TCudaMatrix &d_dev,
                                                                                                                    TCudaMatrix &t_dev,
                                                                                                                    int kernelAIdx,
                                                                                                                    bool * convergedFlag_dev,
                                                                                                                    typename TCudaMatrix::PREC absTOL,
                                                                                                                    typename TCudaMatrix::PREC relTOL);
   template __host__ void sorProx_StepB_kernelWrap<SorProxSettings2RPlusAndDisk , TCudaMatrix>( TCudaMatrix &t_dev, TCudaMatrix &T_dev,  TCudaMatrix &x_old_dev,int kernelAIdx);

   template __host__ void sorProxRelaxed_StepA_kernelWrap<RelaxedSorProxSettings1RPlusAndDisk  , TCudaMatrix>(TCudaMatrix &mu_dev, TCudaMatrix &x_new_dev, TCudaMatrix &t_dev, TCudaMatrix &d_dev, int kernelAIdx);
   template __host__ void sorProxRelaxed_StepB_kernelWrap<RelaxedSorProxSettings1RPlusAndDisk  , TCudaMatrix>(TCudaMatrix &t_dev,TCudaMatrix &T_dev,  TCudaMatrix &x_new_dev,int kernelAIdx);

   template __host__ void sorProxRelaxed_StepA_kernelWrap<RelaxedSorProxKernelSettings<1,ConvexSets::RPlusAndDisk>  , TCudaMatrix>(TCudaMatrix &mu_dev, TCudaMatrix &x_new_dev, TCudaMatrix &t_dev, TCudaMatrix &d_dev, int kernelAIdx);
   template __host__ void sorProxRelaxed_StepB_kernelWrap<RelaxedSorProxKernelSettings<1,ConvexSets::RPlusAndDisk>  , TCudaMatrix>(TCudaMatrix &t_dev,TCudaMatrix &T_dev,  TCudaMatrix &x_new_dev,int kernelAIdx);
   template __host__ void sorProxRelaxed_StepA_kernelWrap<RelaxedSorProxKernelSettings<2,ConvexSets::RPlusAndDisk>  , TCudaMatrix>(TCudaMatrix &mu_dev, TCudaMatrix &x_new_dev, TCudaMatrix &t_dev, TCudaMatrix &d_dev, int kernelAIdx);
   template __host__ void sorProxRelaxed_StepB_kernelWrap<RelaxedSorProxKernelSettings<2,ConvexSets::RPlusAndDisk>  , TCudaMatrix>(TCudaMatrix &t_dev,TCudaMatrix &T_dev,  TCudaMatrix &x_new_dev,int kernelAIdx);
   template __host__ void sorProxRelaxed_StepA_kernelWrap<RelaxedSorProxKernelSettings<4,ConvexSets::RPlusAndDisk>  , TCudaMatrix>(TCudaMatrix &mu_dev, TCudaMatrix &x_new_dev, TCudaMatrix &t_dev, TCudaMatrix &d_dev, int kernelAIdx);
   template __host__ void sorProxRelaxed_StepB_kernelWrap<RelaxedSorProxKernelSettings<4,ConvexSets::RPlusAndDisk>  , TCudaMatrix>(TCudaMatrix &t_dev,TCudaMatrix &T_dev,  TCudaMatrix &x_new_dev,int kernelAIdx);
   //template __host__ void sorProxRelaxed_StepA_kernelWrap<RelaxedSorProxKernelSettings<8,ConvexSets::RPlusAndDisk>  , TCudaMatrix>(TCudaMatrix &mu_dev, TCudaMatrix &x_new_dev, TCudaMatrix &t_dev, TCudaMatrix &d_dev, int kernelAIdx);
   //template __host__ void sorProxRelaxed_StepB_kernelWrap<RelaxedSorProxKernelSettings<8,ConvexSets::RPlusAndDisk>  , TCudaMatrix>(TCudaMatrix &t_dev,TCudaMatrix &T_dev,  TCudaMatrix &x_new_dev,int kernelAIdx);
   template __host__ void sorProxRelaxed_StepA_kernelWrap<RelaxedSorProxKernelSettings<16,ConvexSets::RPlusAndDisk>  , TCudaMatrix>(TCudaMatrix &mu_dev, TCudaMatrix &x_new_dev, TCudaMatrix &t_dev, TCudaMatrix &d_dev, int kernelAIdx);
   template __host__ void sorProxRelaxed_StepB_kernelWrap<RelaxedSorProxKernelSettings<16,ConvexSets::RPlusAndDisk>  , TCudaMatrix>(TCudaMatrix &t_dev,TCudaMatrix &T_dev,  TCudaMatrix &x_new_dev,int kernelAIdx);
   template __host__ void sorProxRelaxed_StepA_kernelWrap<RelaxedSorProxKernelSettings<32,ConvexSets::RPlusAndDisk>  , TCudaMatrix>(TCudaMatrix &mu_dev, TCudaMatrix &x_new_dev, TCudaMatrix &t_dev, TCudaMatrix &d_dev, int kernelAIdx);
   template __host__ void sorProxRelaxed_StepB_kernelWrap<RelaxedSorProxKernelSettings<32,ConvexSets::RPlusAndDisk>  , TCudaMatrix>(TCudaMatrix &t_dev,TCudaMatrix &T_dev,  TCudaMatrix &x_new_dev,int kernelAIdx);
   template __host__ void sorProxRelaxed_StepA_kernelWrap<RelaxedSorProxKernelSettings<64,ConvexSets::RPlusAndDisk>  , TCudaMatrix>(TCudaMatrix &mu_dev, TCudaMatrix &x_new_dev, TCudaMatrix &t_dev, TCudaMatrix &d_dev, int kernelAIdx);
   template __host__ void sorProxRelaxed_StepB_kernelWrap<RelaxedSorProxKernelSettings<64,ConvexSets::RPlusAndDisk>  , TCudaMatrix>(TCudaMatrix &t_dev,TCudaMatrix &T_dev,  TCudaMatrix &x_new_dev,int kernelAIdx);
   template __host__ void sorProxRelaxed_StepA_kernelWrap<RelaxedSorProxKernelSettings<128,ConvexSets::RPlusAndDisk>  , TCudaMatrix>(TCudaMatrix &mu_dev, TCudaMatrix &x_new_dev, TCudaMatrix &t_dev, TCudaMatrix &d_dev, int kernelAIdx);
   template __host__ void sorProxRelaxed_StepB_kernelWrap<RelaxedSorProxKernelSettings<128,ConvexSets::RPlusAndDisk>  , TCudaMatrix>(TCudaMatrix &t_dev,TCudaMatrix &T_dev,  TCudaMatrix &x_new_dev,int kernelAIdx);

   template __host__ void convergedEach_kernelWrap(TCudaMatrix &x_new_dev, TCudaMatrix &x_old_dev, bool * convergedFlag_dev, typename TCudaMatrix::PREC absTOL, typename TCudaMatrix::PREC relTOL);

#undef TCudaMatrix

}
