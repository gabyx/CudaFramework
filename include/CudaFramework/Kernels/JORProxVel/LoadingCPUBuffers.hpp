
#ifndef LoadingCPUBuffers_hpp
#define LoadingCPUBuffers_hpp

#include "GeneralStructs.hpp"
#include "AssertionDebug.hpp"


namespace LoadingCPUBuffers{



template<typename MatrixUIntType,typename MatrixType,typename ContactDataListType,typename BodyDataListType>
void loadComplete( MatrixUIntType globalBufferCPU,
                   MatrixUIntType indexSetCPU,
                   MatrixType contBufferCPU,
                   MatrixType bodyBufferCPU,
                   ContactDataListType &contactDataList,
                   BodyDataListType &bodyDataList){

     DEFINE_JORPROXVEL_GPUBUFFER_OFFSET_NAMESPACES
     unsigned int u_s1;

     u_s1=B::u1_s;

    ///  Load into storage


    unsigned int i = 0;
    unsigned int numberOfBodies=bodyBufferCPU.rows();

    for(auto & d : bodyDataList) {

        d.h_f    = bodyBufferCPU.template block<1,B::f_l>(i,B::f_s);
        d.h_m    = bodyBufferCPU.template block<1,B::tq_l>(i,B::tq_s);
        d.deltaT = 0.1;

        ASSERTMSG(d.deltaT ==0.1, " d.deltaT needs to be 0.1 for testing purposes (GPU uses same value )")

        d.mInv.Zero();
        for(int j=0;j<3;j++){

        d.mInv(j,j)  = bodyBufferCPU(i,B::mInv_s);
        d.mInv(j+3,j+3)  = bodyBufferCPU(i,B::thetaInv_s+j);
        }
        d.u = bodyBufferCPU.template block<1,B::u1_l>(i,u_s1);

        d.regM.Zero();

        for(int z=0; z<3;z++)
            {
                d.regM(z,z) =  1/bodyBufferCPU(i,B::mInv_s);
                d.regM(z+3,z+3) = 1/bodyBufferCPU(i,B::thetaInv_s+z);
               }
        d.test;
       i++;

    }

     i=0;

    for(auto & d : contactDataList) {

        ASSERTMSG(C::lambda_l ==3 , " Size of Lambda has to be 3")
        ASSERTMSG(B::u1_l ==6 , " Size of u has to be 6")
        ASSERTMSG(B::omegaOff ==3 , " Size of omega has to be 3")

        d.bodyIdx1  =  indexSetCPU(i,I::b1Idx_s);
        d.bodyIdx2  =  indexSetCPU(i,I::b2Idx_s);

        d.invR= MatrixType::Zero(C::r_l,C::r_l);

        d.mu =contBufferCPU(i,C::mu_s);
        d.alpha=contBufferCPU(i,C::alpha_s);

        d.n     = contBufferCPU.template block<1,C::n_l>(i,C::n_s);
        d.q1 = bodyBufferCPU.template block<1,B::q_l>(d.bodyIdx1,B::q_s);
        d.q2 = bodyBufferCPU.template block<1,B::q_l>(d.bodyIdx2,B::q_s);
        d.veciRsc1 = contBufferCPU.template block<1,C::rSC1_l>(i,C::rSC1_s);
        d.veciRsc2 = contBufferCPU.template block<1,C::rSC2_l>(i,C::rSC2_s);
        d.chi = contBufferCPU.template block<1,C::chi_l>(i,C::chi_s);
        d.eps = MatrixType::Zero(3,3);

        for (int z=0; z<C::eps_l; z++) {
            d.eps(z,z)=contBufferCPU(i,C::eps_s+z);
            d.invR(z,z)=contBufferCPU(i,C::r_s+z);
        }

        for(int z=0; z<3; z++) {

            d.matWbody1.template block<3,1>(0,z)=contBufferCPU.template block<1,3>(i,C::w1_s+3*z);
            d.matWbody1.template block<3,1>(3,z)=contBufferCPU.template block<1,3>(i,C::w1r_s+3*z);
        }


        d.b=contBufferCPU.template block<1,C::b_l>(i,C::b_s);
        d.lambdaOld=contBufferCPU.template block<1,C::lambda_l>(i,C::lambda_s);

        for(int z=0; z<3; z++) {
            d.matWbody2.template block<B::omegaOff,1>(B::omegaOff,z)=contBufferCPU.template block<1,B::omegaOff>(i,C::w2r_s+B::omegaOff*z);
            d.matWbody2.template block<(B::u1_l-B::omegaOff),1>(0,z)=-contBufferCPU.template block<1,3>(i,C::w1_s+(B::u1_l-B::omegaOff)*z);
        }

        i++;
    }

}



template<typename MatrixUIntType,typename MatrixType,typename ContactDataListType,typename BodyDataListType>
void loadBack(     MatrixUIntType &globalBufferCPU,
                   MatrixUIntType &indexSetCPU,
                   MatrixType &contBufferCPU,
                   MatrixType &bodyBufferCPU,
                   ContactDataListType contactDataList,
                   BodyDataListType bodyDataList){

    DEFINE_JORPROXVEL_GPUBUFFER_OFFSET_NAMESPACES

    ///  Load into storage


    unsigned int i = 0;

    for(auto & d : bodyDataList) {

      for(int z=0; z<3; z++) {

            bodyBufferCPU(i,B::mInv_s)=d.mInv(z,z);
            bodyBufferCPU(i,B::thetaInv_s+z)=d.mInv(z+3,z+3);
      }

     bodyBufferCPU.template block<1,B::u1_l>(i,B::u1_s)=d.u_2;
       i++;

    }

     i=0;

    for(auto & d : contactDataList) {

        indexSetCPU(i,I::b1Idx_s)=d.bodyIdx1;
        indexSetCPU(i,I::b2Idx_s)=d.bodyIdx2;

        contBufferCPU(i,C::mu_s)=d.mu;

        for(int z=0; z<3; z++) {
            contBufferCPU.template block<1,3>(i,C::w1_s+3*z)=d.matWbody1.template block<3,1>(0,z);
            contBufferCPU.template block<1,3>(i,C::w1r_s+3*z)=d.matWbody1.template block<3,1>(3,z);
        }
        for(int z=0; z<3; z++) {
           contBufferCPU.template block<1,3>(i,C::w2r_s+3*z)=d.matWbody2.template block<3,1>(3,z);
        }

        for (int z=0; z<3; z++) {
            contBufferCPU(i,C::r_s+z)=d.invR(z,z);
        }

        contBufferCPU.template block<1,C::b_l>(i,C::b_s)=d.b;
        contBufferCPU.template block<1,C::lambda_l>(i,C::lambda_s)=d.lambdaOld;

        i++;
    }

}

};

# endif
