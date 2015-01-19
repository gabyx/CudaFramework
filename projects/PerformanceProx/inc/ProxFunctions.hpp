/*
*  ProxFunctions.hpp
*
*  Created by Gabriel Nützi on 21.03.10.
*  Copyright 2010 ETH. All rights reserved.
*
*/

#ifndef ProxFunctions_hpp
#define ProxFunctions_hpp

#include <Eigen/Dense>
//#include "TypeDefs.hpp"

#include "CudaFramework/General/AssertionDebug.hpp"


#include "ConvexSets.hpp"



#define INLINE_PROX 1

#if INLINE_PROX == 1
#define INLINE_PROX_KEYWORD inline
#else
#define INLINE_PROX_KEYWORD
#endif

/**
* @brief Namespace for the Prox Functions.
*/
namespace Prox{

/**
* @addtogroup Inclusion
* @{
*/

   /**
   * @defgroup ProxFunctions Prox Functions
   * @brief Various proximal function onto different convex sets..
   * @{
   */


   // PROX SINGLE ====================================================================================================
   /**
   * @brief Base template for a single proximal functions on to several convex sets.
   */
   template< typename Set >
   struct ProxFunction{};

   /**
   * @brief Spezialisation for a  Prox onto \f$ C_1 = \mathcal{R}_{+} \f$ .
   */
   template<>
   struct ProxFunction<ConvexSets::RPlus>{

      /**
      * @brief Single in place Prox.
      * @param y Input/output which is proxed on to \f$ C_1\f$ .
      */
      template<typename PREC>
      static INLINE_PROX_KEYWORD void doProxSingle( PREC & y	)
      {
         using std::max;
         y = max((PREC)y,(PREC)0.0);
      }
      /**
      * @brief Single Prox.
      * @param x Input vector
      * @param y Output which is proxed on to \f$ C_1\f$ .
      */
      template<typename PREC>
      static INLINE_PROX_KEYWORD void doProxSingle( const PREC & x, PREC & y	)
      {
         using std::max;
         y = max((PREC)x,(PREC)0.0);
      }

      /**
      * @brief Multi in place Prox.
      * @param y Input/output which is proxed on to \f$ C_1\f$ .
      */
      template<typename Derived>
      static INLINE_PROX_KEYWORD void doProxMulti(const Eigen::MatrixBase<Derived> & y)
      {
         EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived);
         Eigen::MatrixBase<Derived> & y_ref =  const_cast<Eigen::MatrixBase<Derived> &>(y);

         for (unsigned int i=0; i<y_ref.rows(); i++)
         {
            using std::max;
            y_ref[i] = max(y_ref[i],0.0);
         }
      }

      /**
      * @brief Multi Prox.
      * @param x Input vector
      * @param y Output which is proxed on to \f$ C_1\f$ .
      */
      template<typename Derived, typename DerivedOther>
      static INLINE_PROX_KEYWORD void doProxMulti(const Eigen::MatrixBase<Derived> & x, const Eigen::MatrixBase<DerivedOther> & y)
      {
         EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived);
         EIGEN_STATIC_ASSERT_VECTOR_ONLY(DerivedOther);

         Eigen::MatrixBase<DerivedOther> & y_ref =  const_cast<Eigen::MatrixBase<DerivedOther> &>(y);
         ASSERTMSG( x.rows() == y_ref.rows(), "Wrong dimension!");

         for (unsigned int i=0; i<y_ref.rows(); i++)
         {
            y_ref[i] = max(x[i],0.0);
         }
      }

   };


   /**
   * @brief Spezialisation for a single Prox onto a scaled unit disk \f$ C_1 = \{ x | |x| < r \} \f$ .
   */
   template<>
   struct ProxFunction<ConvexSets::Disk>{

      /**
      * @brief Spezialisation for a single Prox onto a scaled unit disk \f$ C_1 \f$.
      * @param radius Scaling factor for the convex unit disk.
      * @param y Input/output vector which is proxed onto a scaled unit disk \f$ C_1 \f$.
      */
      template<typename PREC, typename Derived>
      static INLINE_PROX_KEYWORD void  doProxSingle(  const PREC & radius, const Eigen::MatrixBase<Derived> & y){
         // Solve the set (disc with radius mu_P_N), one r is used for the prox!
         EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived,2);
         Eigen::MatrixBase<Derived> & y_ref =  const_cast<Eigen::MatrixBase<Derived> &>(y);

         PREC absvalue;
         absvalue = y_ref.norm();
         if (absvalue > radius){
            y_ref =  y_ref / absvalue * radius;
         }
      }

      /**
      * @brief Spezialisation for a single Prox onto a scaled unit disk  \f$ C_1 \f$.
      * @param radius Scaling factor for the convex unit disk.
      * @param x Input vector.
      * @param y Output which is proxed on to \f$ C_1 \f$.
      */
      template<typename PREC, typename Derived, typename DerivedOther>
      static INLINE_PROX_KEYWORD void  doProxSingle(  const PREC & radius, const Eigen::MatrixBase<Derived> & x,  const Eigen::MatrixBase<DerivedOther> & y){
         // Solve the set (disc with radius mu_P_N), one r is used for the prox!
         EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived,2);
         EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(DerivedOther,2);
         Eigen::MatrixBase<DerivedOther> & y_ref =  const_cast<Eigen::MatrixBase<DerivedOther> &>(y);

         PREC absvalue;
         absvalue = x.norm();
         if (absvalue > radius){
            y_ref =  x / absvalue * radius;
         }
      }

      /**
      * @brief Spezialisation for a multi Prox onto a scaled unit disk \f$ C_1 \f$.
      * @param radius Scaling factor for the convex unit disk.
      * @param y Input/output vector which is proxed onto a scaled unit disk \f$ C_1 \f$.
      */
      template< typename Derived, typename DerivedOther>
      static INLINE_PROX_KEYWORD void doProxMulti(  const Eigen::MatrixBase<Derived> & radius, const Eigen::MatrixBase<DerivedOther> & y)
      {
         typedef typename Derived::Scalar PREC;
         EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived);
         EIGEN_STATIC_ASSERT_VECTOR_ONLY(DerivedOther);
         Eigen::MatrixBase<DerivedOther> & y_ref =  const_cast<Eigen::MatrixBase<DerivedOther> &>(y);
         ASSERTMSG( (2) * radius.rows() == y_ref.rows(), "Wrong dimension!");

         //// Solve the set (disc with radius mu_P_N), one r is used for the prox!
         PREC absvalue;
         for (int i=0; i<radius.rows(); i++) {
            absvalue = (y_ref.segment<2>(2*i)).norm();
            if (absvalue > radius(i,0)){
               y_ref.segment<2>(2*i) =  y_ref.segment<2>(2*i) / absvalue * radius(i,0);
            }
         }
      }

      /**
      * @brief Spezialisation for a multi Prox onto a scaled unit disk  \f$ C_1 \f$.
      * @param radius Scaling factor for the convex unit disk.
      * @param x Input vector.
      * @param y Output which is proxed on to \f$ C_1 \f$.
      */
      template< typename Derived, typename DerivedOther1, typename DerivedOther2>
      static INLINE_PROX_KEYWORD void doProxMulti(  const Eigen::MatrixBase<Derived> & radius, const Eigen::MatrixBase<DerivedOther1> & x, const Eigen::MatrixBase<DerivedOther2> & y)
      {
         typedef typename Derived::Scalar PREC;
         EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived);
         EIGEN_STATIC_ASSERT_VECTOR_ONLY(DerivedOther1);
         EIGEN_STATIC_ASSERT_VECTOR_ONLY(DerivedOther2);
         Eigen::MatrixBase<Derived> & y_ref =  const_cast<Eigen::MatrixBase<Derived> &>(y);
         ASSERTMSG( x.rows() == y_ref.rows(), "Wrong dimension!");
         ASSERTMSG( (2) * radius.rows() == y_ref.rows(), "Wrong dimension!");

         // Solve the set (disc with radius mu_P_N), one r is used for the prox!
         PREC absvalue;
         for (int i=0; i<radius.rows(); i++) {
            absvalue = (x.segment<2>(2*i)).norm();
            if (absvalue > radius(i,0)){
               y_ref.segment<2>(2*i) =  x.segment<2>(2*i) / absvalue * radius(i,0);
            }
         }
      }

   };

   /**
   * @brief Spezialisation for a single Prox onto  \f$ C_1 = \mathcal{R}_{+} \f$   and a scaled unit disk \f$ C_2 =\{ x | |x| < 1 \} \f$ . The scale factor \f$ r\f$  is the proxed value on to \f$ C_1 \f$ .
   */
   template<>
   struct ProxFunction<ConvexSets::RPlusAndDisk>{

      /** @brief Spezialisation for a single Prox onto  \f$ C_1 \f$  and \f$ C_2 \f$ .
      * @param radius The radius for scaling.
      * @param y Input/output vector, where the first value in y has been proxed onto  \f$ C_1 \f$  and the second 2 values onto the unit disk which is scaled by the first proxed value.
      */
      template<typename PREC, typename Derived>
      static INLINE_PROX_KEYWORD void doProxSingle(  const PREC & scale_factor, const Eigen::MatrixBase<Derived> & y)
      {

         EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived,3);
         Eigen::MatrixBase<Derived> & y_ref =  const_cast<Eigen::MatrixBase<Derived> &>(y);

         //// Solve the set (disc with radius mu_P_N), one r is used for the prox!
         PREC absvalue;
         // Prox normal
         using std::max;
         y_ref(0) = max((PREC)y_ref(0),(PREC)0.0);
         // Prox tangential
         absvalue = (y_ref.template segment<2>(1)).norm();
         if (absvalue > scale_factor*y_ref(0)){
            y_ref.template segment<2>(1) =  y_ref.template segment<2>(1) / absvalue * scale_factor*y_ref(0);
         }
      }

      /**
      * @brief Spezialisation for a single Prox onto  \f$ C_1 \f$  and \f$ C_2 \f$ .
      * @param radius The radius for scaling.
      * @param x Input vector.
      * @param y Output vector, where the first value in y has been proxed onto  \f$ C_1 \f$  and the second 2 values onto the unit disk which is scaled by the first proxed value.
      */
      template<typename PREC, typename Derived, typename DerivedOther>
      static INLINE_PROX_KEYWORD void doProxSingle(  const PREC & scale_factor, const Eigen::MatrixBase<Derived> & x, Eigen::MatrixBase<DerivedOther> & y)
      {
         EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived,3);
         EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(DerivedOther,3);
         Eigen::MatrixBase<Derived> & y_ref =  const_cast<Eigen::MatrixBase<Derived> &>(y);

         //// Solve the set (disc with radius mu_P_N), one r is used for the prox!
         PREC absvalue;
         // Prox normal
         using std::max;
         y_ref(0) = max(x(0),0.0);
         // Prox tangential
         absvalue = (x.segment<2>(1)).norm();
         if (absvalue > scale_factor*y_ref(0)){
            y_ref.segment<2>(1) =  x.segment<2>(1) / absvalue * scale_factor*y_ref(0);
         }
      }

      /** @brief Spezialisation for a multi Prox onto  \f$ C_1 \f$  and \f$ C_2 \f$ .
      * @param radius The radius for scaling.
      * @param y Input/output vector, where the first value in y has been proxed onto  \f$ C_1 \f$  and the second 2 values onto the unit disk which is scaled by the first proxed value.
      */
      template< typename Derived, typename DerivedOther>
      static INLINE_PROX_KEYWORD void doProxMulti( const Eigen::MatrixBase<Derived> & scale_factor, const Eigen::MatrixBase<DerivedOther> & y)
      {
         typedef typename Derived::Scalar PREC;
         EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived);
         EIGEN_STATIC_ASSERT_VECTOR_ONLY(DerivedOther);
         Eigen::MatrixBase<DerivedOther> & y_ref =  const_cast<Eigen::MatrixBase<DerivedOther> &>(y);
         ASSERTMSG( (3) * scale_factor.rows() == y_ref.rows(), "Wrong dimension!");

         //// Solve the set (disc with radius mu_P_N), one r is used for the prox!
         PREC absvalue;
         for (int i=0; i<scale_factor.rows(); i++) {
            // Rplus
            using std::max;
            y_ref(3*i) = max((PREC)y_ref(3*i),(PREC)0.0);
            // Disk
            absvalue = (y_ref.template segment<2>(3*i+1)).norm();
            if (absvalue > scale_factor(i,0)*y_ref(3*i)){
               y_ref.template segment<2>(3*i+1) =  y_ref.template segment<2>(3*i+1) / absvalue * scale_factor(i,0)*y_ref(3*i);
            }
         }
      }

      /**
      * @brief Spezialisation for a multi Prox onto  \f$ C_1 \f$  and \f$ C_2 \f$ .
      * @param radius The radius for scaling.
      * @param x Input vector.
      * @param y Output vector, where the first value in y has been proxed onto  \f$ C_1 \f$  and the second 2 values onto the unit disk which is scaled by the first proxed value.
      */
      template< typename Derived, typename DerivedOther1, typename DerivedOther2>
      static INLINE_PROX_KEYWORD void doProxMulti( const Eigen::MatrixBase<Derived> & scale_factor, const Eigen::MatrixBase<DerivedOther1> & x, const Eigen::MatrixBase<DerivedOther2> & y)
      {
         typedef typename Derived::Scalar PREC;
         EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived);
         EIGEN_STATIC_ASSERT_VECTOR_ONLY(DerivedOther1);
         EIGEN_STATIC_ASSERT_VECTOR_ONLY(DerivedOther2);
         Eigen::MatrixBase<Derived> & y_ref =  const_cast<Eigen::MatrixBase<Derived> &>(y);
         ASSERTMSG( (3) * scale_factor.rows() == x.rows(), "Wrong dimension!");
         ASSERTMSG( (3) * scale_factor.rows() == y_ref.rows(), "Wrong dimension!");

         //// Solve the set (disc with radius mu_P_N), one r is used for the prox!
         PREC absvalue;
         for (int i=0; i<scale_factor.rows(); i++) {
            // Rplus
            using std::max;
            y_ref(3*i) = max(x(3*i),0.0);
            // Disk
            absvalue = (x.segment<2>(3*i+1)).norm();
            if (absvalue > scale_factor(i,0)*y_ref(3*i)){
               y_ref.segment<2>(3*i+1) =  x.segment<2>(3*i+1) / absvalue * scale_factor(i,0)*y_ref(3*i);
            }
         }
      }

   };
   /* @} */

/** @} */
   };





   // =====================================================================================================================

   /**
   * @brief Different numerical functions.
   */
   namespace Numerics{

/**
* @addtogroup Inclusion
* @{
*/

   // CANCEL FUNCTIONS ====================================================================================================
   /**
   * @defgroup CancelationFunctions Cancelation Functions
   * @brief Cancelation function which are used to abort the iteration during the Prox iteration.
   */
   /* @{ */
   template<typename Derived>
   INLINE_PROX_KEYWORD bool cancelCriteriaVector( const Eigen::MatrixBase<Derived>& P_N_old,
      const Eigen::MatrixBase<Derived>& P_N_new,
      const Eigen::MatrixBase<Derived>& P_T_old,
      const Eigen::MatrixBase<Derived>& P_T_new ,
      double AbsTol, double RelTol)
   {
      EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived);
      typedef typename Derived::Scalar PREC;

      PREC NormP =0;
      PREC RelNormP=0;
      for(int i=0;i<P_N_old.size();i++){
         NormP += P_N_old[i]*P_N_old[i];
         RelNormP += pow(P_N_new[i]-P_N_old[i],2);
      }

      for(int i=0;i<P_T_old.size();i++){
         NormP += P_T_old[i]*P_T_old[i];
         RelNormP += pow(P_T_new[i]-P_T_old[i],2);
      }

      NormP = sqrt(NormP);
      RelNormP = sqrt(RelNormP);

      if (RelNormP < NormP * RelTol + AbsTol){
         return  true;
      }

#if CoutLevelSolverWhenContact>2
      CLEARLOG;
      logstream << "Cancel Criterion :" << RelNormP << " < " << NormP * m_Settings.m_RelTol + m_Settings.m_AbsTol;
      LOG(m_pSolverLog);
#endif

      return false;
   }


   template<typename Derived>
   INLINE_PROX_KEYWORD bool cancelCriteriaValue(    const Eigen::MatrixBase<Derived>& P_N_old,
      const Eigen::MatrixBase<Derived>& P_N_new,
      const Eigen::MatrixBase<Derived>& P_T_old,
      const Eigen::MatrixBase<Derived>& P_T_new,
      double AbsTol, double RelTol)
   {
      EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived);
      typedef typename Derived::Scalar PREC;

      using std::abs;

      for(int i=0;i<P_N_old.size();i++){
         if ( abs(P_N_new[i]-P_N_old[i]) > abs(P_N_old[i]) * RelTol + AbsTol){
            return  false;
         }
      }

      for(int i=0;i<P_T_old.size();i++){
         if ( abs(P_T_new[i]-P_T_old[i]) > abs(P_T_old[i]) * RelTol +AbsTol){
            return  false;
         }
      }
      return true;
   }

   template<typename Derived>
   INLINE_PROX_KEYWORD bool cancelCriteriaValue(   const Eigen::MatrixBase<Derived>& P_old,
                                                   const Eigen::MatrixBase<Derived>& P_new,
                                                   double AbsTol, double RelTol)
   {
      EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived);
      typedef typename Derived::Scalar PREC;

      for(int i=0;i<P_old.size();i++){
         if ( abs(P_new[i]-P_old[i]) > abs(P_old[i]) * RelTol + AbsTol){
            return  false;
         }
      }
      return true;
   }
   template<typename Derived, typename DerivedOther>
   INLINE_PROX_KEYWORD bool cancelCriteriaValue(   const Eigen::MatrixBase<Derived>& P_old,
                                                   const Eigen::MatrixBase<DerivedOther>& P_new,
                                                   double AbsTol, double RelTol,
                                                   unsigned int & counter)
   {
      EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived);
      EIGEN_STATIC_ASSERT_VECTOR_ONLY(DerivedOther);

      typedef typename Derived::Scalar PREC;

      for(int i=0;i<P_old.size();i++){
         if ( abs(P_new[i]-P_old[i]) > abs(P_old[i]) * RelTol + AbsTol){
            return  false;
         }
      }
      counter++;
      return true;
   }
   /** @} */
/** @} */
}


#endif
