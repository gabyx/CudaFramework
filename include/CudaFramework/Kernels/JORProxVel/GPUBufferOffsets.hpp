
#ifndef CudaFramework_Kernels_JORProxVel_GPUBufferOffsets_hpp
#define CudaFramework_Kernels_JORProxVel_GPUBufferOffsets_hpp


#define DEFINE_JORPROXVEL_GPUBUFFER_OFFSET_NAMESPACES \
    typedef JORProxVelGPU::GPUBufferOffsets::ContBufferOffsets    C;\
    typedef JORProxVelGPU::GPUBufferOffsets::BodyBufferOffsets    B;\
    typedef JORProxVelGPU::GPUBufferOffsets::GlobalBufferOffsets  G;\
    typedef JORProxVelGPU::GPUBufferOffsets::IndexBufferOffsets   I;



namespace JORProxVelGPU{

    /** Layout for the JORProxVelGPU Buffers
    *  Abbreviations:  *_s  : offset start,
    *                  *_l  : offset length
    *  Offsets always the size of the underlying type of the corresponding buffer matrices
    */
    struct GPUBufferOffsets {




    /**  Layout for the Body Buffers */
    struct BodyBufferOffsets {

        static const unsigned   int omegaOff= 3;  ///<  omega Offset

        static const unsigned   int u1_s = 0;  ///< velocity v
        static const unsigned   int u1_l = 6;

        static const unsigned  int q_s = u1_s + u1_l ;   ///< quaternion
        static const unsigned  int q_l = 4;

        static const unsigned  int mInv_s =q_s+q_l;  ///< mass inverse
        static const unsigned  int mInv_l = 1;

        static const unsigned  int thetaInv_s = mInv_s+mInv_l;  ///< thetainverse
        static const unsigned  int thetaInv_l= 3;

        static const unsigned  int f_s = thetaInv_s+thetaInv_l;  ///< external force
        static const unsigned  int f_l = 3;

        static const unsigned  int tq_s = f_s+f_l;  ///< external moment / torque
        static const unsigned  int tq_l = 3;



        ///====================================================///

        static const unsigned  int length = u1_l+q_l+thetaInv_l+mInv_l+f_l+tq_l;

    };

    /**  Layout for the Contact Buffers */
    struct ContBufferOffsets {

        // Before Initialization =========================
        static const unsigned  int n_s = 0;  /// normal vector
        static const unsigned  int n_l = 3;

        static const unsigned  int rSC1_s = n_s+n_l;  /// vector from S to contact
        static const unsigned  int rSC1_l = 3;

        static const unsigned  int rSC2_s = rSC1_s+rSC1_l;  /// vector from S to contact Body 2
        static const unsigned  int rSC2_l = 3;
        // ===============================================

        //  After Initialization =========================
        static const unsigned  int w1_s = 0;    /// general force direction body 1 upper 3x3 block (translation)
        static const unsigned  int w1_l = 9;

        static const unsigned  int w1r_s = 9;  /// general force direction body 1 lower 3x3 block (angular velocity)
        static const unsigned  int w1r_l = 9;

        static const unsigned  int w2r_s = 18;  /// general force direction body 2 lower 3x3 block (angular velocity)
        static const unsigned  int w2r_l = 9;

        static const unsigned  int lambda_s = 27;   /// general force lambda
        static const unsigned  int lambda_l = 3;

        static const unsigned  int chi_s = 30;   /// chi which is 0 for simulated bodies
        static const unsigned  int chi_l = 3;

        static const unsigned  int eps_s = 33;   /// epsilon (for bouncing)
        static const unsigned  int eps_l = 3;

        static const unsigned  int b_s = 36;   ///  (1+eps)*chi+eps*W.T*u
        static const unsigned  int b_l = 3;

        static const unsigned  int r_s = 39;   ///  diagonal metric values of the prox
        static const unsigned  int r_l = 3;

        static const unsigned  int alpha_s =42;   /// norming alpha for R  in the prox
        static const unsigned  int alpha_l = 1;

        static const unsigned  int mu_s = 43;    /// friction mu
        static const unsigned  int mu_l = 1;
        // ==========================================================

        static const unsigned  length = w1_l+w1r_l+w2r_l+alpha_l+mu_l+lambda_l+chi_l+eps_l+b_l+r_l;

    };

    /**  Layout for the Reduction Buffers */
    struct ReductionBufferOffsets {

        static const unsigned  int u1_s = 0;   /// velocity //TODO rename to v1_s ...
        static const unsigned  int u1_l = 3;

        static const unsigned  int omega_s = 3;   /// rotational velocity
        static const unsigned  int omega_l = 3;

        static const unsigned  length = u1_l+omega_l;

    };

    /**  Layout for the Global Buffers */
    struct GlobalBufferOffsets {

        static const unsigned  int iter_s = 0;  ///  global iteration counter
        static const unsigned  int iter_l = 1;

        static const unsigned  int conv_s = 1;   /// convergence check variable   /// 0 if true  /// else 1
        static const unsigned  int conv_l = 1;

        static const unsigned  length = iter_l+conv_l;///+deltaT_l;

    };

    /**  Layout for the Global Buffers
    *    Offset lengths in size of unsigned integers
    */
    struct IndexBufferOffsets {

        static const unsigned  int redIdx_s = 0;   /// where to put the delta in
        static const unsigned  int redIdx_l = 2;

        static const unsigned  int b1Idx_s = 2;    /// body index 1
        static const unsigned  int b1Idx_l = 1;

        static const unsigned  int b2Idx_s = 3;   /// body index 2
        static const unsigned  int b2Idx_l= 1;

        static const unsigned  length = redIdx_l+b1Idx_l+b2Idx_l;

    };
};

};


#endif // GPUBufferOffsets
