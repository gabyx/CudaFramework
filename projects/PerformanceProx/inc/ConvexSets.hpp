/*
*  ConvexSets.hpp
*
*  Created by Gabriel Nützi on 21.03.10.
*  Copyright 2010 ETH. All rights reserved.
*
*/

#ifndef ConvexSets_hpp
#define ConvexSets_hpp


// CONVEX SETS

   /** @addtogroup ProxFunctions
   * @{
   */

   /**
   * @brief Definitions of all convex sets to use with a prox function.
   */
   struct ConvexSets{
      /**
      * @brief Convex set for \f$ C = \mathcal{R}_{+} \f$ .
      */
      struct RPlus{
         typedef RPlus type;
         static const int Dimension = 1; 
      };

      /**
      * @brief Convex set for a unit disk \f$ C = \{ x | |x| < 1 \} \f$ .
      */
      struct Disk{
         typedef Disk type;
         static const int Dimension = 2; 
      };

      /**
      * @brief Convex set for a Contensou ellipsoid.
      */
      struct ContensouElliposoid{
         typedef ContensouElliposoid type;
         static const int Dimension = 3; 
      };

      /**
      * @brief Convex set for  \f$ C_1 = \mathcal{R}_{+} \f$  and a unit disk \f$ C_2 = \{ x | |x| < 1 \} \f$ .
      * This function applies for triplets, the first value is proxed onto \f$ C_1 \f$  and the second to values in sequence are proxed on to \f$ C_2 \f$ .
      */
      struct RPlusAndDisk{
         typedef RPlusAndDisk type;
         static const int Dimension = 3; 
      };

      /**
      * @brief Convex set for  \f$ C_1 = \mathcal{R}_{+} \f$  and a Contensou ellipsoid.
      * This function applies for triplets, the first value is proxed onto \f$ C_1 \f$  and the second to values in sequence are proxed on to the Contensou ellipsoid.
      */
      struct RPlusAndContensouEllipsoid{
         typedef RPlusAndContensouEllipsoid type;
         static const int Dimension = 4; 
      };

   };

   /** @}*/

#endif