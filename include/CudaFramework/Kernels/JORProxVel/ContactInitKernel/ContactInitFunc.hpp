#ifndef ContactInitFunc_hpp
#define ContactInitFunc_hpp
#include "GeneralStructs.hpp"
#include "GPUBufferOffsets.hpp"

namespace ContactInit{

template <typename MatrixType,typename VectorType>
void gettilde(MatrixType& output,VectorType input)
{
        output(0,0)=0.0;
        output(1,0)=input(2);
        output(2,0)=-input(1);
        output(0,1)=-input(2);
        output(1,1)=0.0;
        output(2,1)=input(0);
        output(0,2)=input(1);
        output(1,2)=-input(0);
        output(2,2)=0.0;

}

template <typename MatrixType,typename Quat>
void rotMatfromQuat(MatrixType& output,Quat input)
{
        output(0,0)=(1-2*(input(3)*input(3)+input(2)*input(2)));
        output(0,1)=2*input(0)*input(3)+2*input(2)*input(1);
        output(0,2)=-2*input(0)*input(2)+2*input(1)*input(3);
        output(1,0)=-2*input(0)*input(3)+2*input(2)*input(1);
        output(1,1)=(1-2*(input(3)*input(3)+input(1)*input(1)));
        output(1,2)=2*input(0)*input(1)+2*input(2)*input(3);
        output(2,0)=2*input(0)*input(2)+2*input(1)*input(3);
        output(2,1)=-2*input(0)*input(1)+2*input(2)*input(3);
        output(2,2)=(1-2*(input(1)*input(1)+input(2)*input(2)));

}

template<typename ContactDataListType,typename BodyDataListType>
void calcComCF_CPU(ContactDataListType &contactDataList,
                   BodyDataListType &bodyDataList){

        DEFINE_JORPROXVEL_GPUBUFFER_OFFSET_NAMESPACES

        bool decide;
        typedef Eigen::Matrix<double,3,3> Matrix3x3;
        typedef Eigen::Matrix<double,3,1> Vector3;

        Vector3 buffer3;
        Matrix3x3 buffer3x3;


        for (auto & d : contactDataList) {

        d.n=-d.n;
        decide=std::abs(d.n(0)) > std::abs(d.n(2));
        d.n.normalize();

        if (decide) {
            d.t1(0)=-d.n(2);
            d.t1(1)= 0;
            d.t1(2)= d.n(0);
        } else {
            d.t1(0) = 0;
            d.t1(1) = d.n(2);
            d.t1(2) =-d.n(1);
        }

        d.t1.normalize();
        d.t2=d.n.cross(d.t1);
        d.t2.normalize();

        gettilde(d.matiRscTilde1,d.veciRsc1);
        gettilde(d.matiRscTilde2,d.veciRsc2);

        d.matContFrame.template block<C::n_l,1>(0,0)=d.n;
        d.matContFrame.template block<C::n_l,1>(0,1)=d.t1;
        d.matContFrame.template block<C::n_l,1>(0,2)=d.t2;

        rotMatfromQuat(d.matAai,d.q1);
        ///A=A*R
        d.wR1=d.matAai*d.matiRscTilde1*d.matContFrame;

        rotMatfromQuat(d.matAai,d.q2);

        d.wR2=d.matAai*d.matiRscTilde2*d.matContFrame*double(-1);

        ///MatrixMultiColMaj<PREC>(array_Aai,mat_iRscTilde);
        ///A=A*B
        ///MatrixMultiColMaj<PREC>(array_Aai,contactFrameCPU_ptr+z*9);
        ///PREC* matrix_W_contact[18];

        d.matWbody1.topRows(3)=d.matContFrame;
        d.matWbody1.bottomRows(3)=d.wR1;
        d.matWbody2.topRows(3)=(-1)*d.matContFrame;
        d.matWbody2.bottomRows(3)=d.wR2;

        d.b=d.chi+d.eps*d.chi+d.eps*d.matWbody1.transpose()*bodyDataList[d.bodyIdx1].u+d.eps*d.matWbody2.transpose()*bodyDataList[d.bodyIdx2].u;

        buffer3x3=(d.matWbody1.transpose()*bodyDataList[d.bodyIdx1].mInv*d.matWbody1);
        buffer3x3+=(d.matWbody2.transpose()*bodyDataList[d.bodyIdx2].mInv*d.matWbody2);

        d.invR.Zero(C::r_l,C::r_l);

        for(int z=0; z<C::r_l; z++) {
            d.invR(z,z)= 1/buffer3x3(z,z);
        }
        if(d.invR(1,1)>d.invR(2,2)) {
            d.invR(2,2)=d.invR(1,1);
        } else {
            d.invR(1,1)=d.invR(2,2);
        }

        d.invR=d.invR*d.alpha;


    }


}

}


#endif




