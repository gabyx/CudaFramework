// ========================================================================================
//  CudaFramework
//  Copyright (C) 2014 by Gabriel Nützi <nuetzig (at) imes (d0t) mavt (d0t) ethz (d0t) ch>
//
//  This Source Code Form is subject to the terms of the GNU GPL 3.0 licence.
//  If a copy of the GNU GPL 3.0 was not distributed with this
//  file, you can obtain one at http://opensource.org/licenses/GPL-3.0.
// ========================================================================================

#ifndef CudaFramework_General_StaticAssert_hpp
#define CudaFramework_General_StaticAssert_hpp


#define ASSERT_CONCAT_(a, b) a##b
#define ASSERT_CONCAT(a, b) ASSERT_CONCAT_(a, b)
/* These can't be used after statements in c89. */
#ifdef __COUNTER__
  #define STATIC_ASSERT_IMPL(e,m) \
     enum { ASSERT_CONCAT(static_assert_, __COUNTER__) = 1/(!!(e)) };
#else
  /* This can't be used twice on the same line so ensure if using in headers
   * that the headers are not included twice (by wrapping in #ifndef...#endif)
   * Note it doesn't cause an issue when used on same line of separate modules
   * compiled with gcc -combine -fwhole-program.  */
  #define STATIC_ASSERT_IMPL(e,m) \
     enum { ASSERT_CONCAT(assert_line_, __LINE__) = 1/(!!(e)) };
#endif


// XXX nvcc 2.3 can't handle STATIC_ASSERT
#if defined(__CUDACC__) /* && (CUDA_VERSION < 30)*/

    #define STATIC_ASSERT( B )        STATIC_ASSERT_IMPL( B , "no message")
    #define STATIC_ASSERTM(B,COMMENT) STATIC_ASSERT_IMPL( B ,  COMMENT )

#else

    #define STATIC_ASSERT( B )          STATIC_ASSERT_IMPL( B , "no message")
    #define STATIC_ASSERTM( B ,COMMENT) STATIC_ASSERT_IMPL( B , COMMENT )

#endif





#endif
