// ========================================================================================
//  CudaFramework
//  Copyright (C) 2014 by Gabriel Nützi <nuetzig (at) imes (d0t) mavt (d0t) ethz (d0t) ch>
//
//  This Source Code Form is subject to the terms of the GNU GPL 3.0 licence. 
//  If a copy of the GNU GPL 3.0 was not distributed with this
//  file, you can obtain one at http://opensource.org/licenses/GPL-3.0.
// ========================================================================================


#ifndef CudaFramework_Kernels_JORProxVel_GeneralStructs_hpp
#define CudaFramework_Kernels_JORProxVel_GeneralStructs_hpp





    ///  Write into fast data structure
template<typename PREC>
struct ContactData {

        typedef Eigen::Matrix<PREC,3,1> Vector3;
        typedef Eigen::Matrix<PREC,4,1> Vector4;
        typedef Eigen::Matrix<PREC,6,1> Vector6;
        typedef Eigen::Matrix<PREC,3,3> Matrix3x3;
        typedef Eigen::Matrix<PREC,3,6> Matrix3x6;
        typedef Eigen::Matrix<PREC,6,3> Matrix6x3;


        Vector4 q1;
        Vector4 q2;

        Vector3 veciRsc1;//
        Vector3 veciRsc2;

        Vector3 chi;

        Matrix3x3 eps;

        Vector6 delta_uBody1;
        Vector6 delta_uBody2;

        Matrix6x3 matWbody1;
        Matrix6x3 matWbody2;

        Vector3 b;
        Matrix3x3 matiRscTilde1;
        Matrix3x3 matiRscTilde2;

        Matrix3x3 matAai;

        Matrix3x3 wR1;
        Matrix3x3 wR2;

        Matrix3x3 matContFrame;
        Matrix3x3 invR;

        PREC mu;
        PREC alpha;

        Vector3 lambdaOld;

        Vector3 n;
        Vector3 t1;
        Vector3 t2;

        unsigned int bodyIdx1;
        unsigned int bodyIdx2;
    };

    template<typename PREC>
    struct BodyData {

        typedef Eigen::Matrix<PREC,3,1> Vector3;
        typedef Eigen::Matrix<PREC,4,1> Vector4;
        typedef Eigen::Matrix<PREC,6,1> Vector6;
        typedef Eigen::Matrix<PREC,3,3> Matrix3x3;
        typedef Eigen::Matrix<PREC,3,6> Matrix3x6;
        typedef Eigen::Matrix<PREC,6,3> Matrix6x3;
        typedef Eigen::Matrix<PREC,6,6> Matrix6x6;

        Vector6 u;///< velocity buffer 2
        Vector3 h_f;
        Vector3 h_m;
        Matrix6x6 mInv;
        Matrix6x6 regM;
        PREC deltaT;

        Vector6 u_2;  ///< velocity buffer 2
        bool test;
    };



#endif // GPUBufferOffsets

