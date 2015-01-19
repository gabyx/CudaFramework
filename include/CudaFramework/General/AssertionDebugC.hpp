// ========================================================================================
//  CudaFramework
//  Copyright (C) 2014 by Gabriel Nützi <nuetzig (at) imes (d0t) mavt (d0t) ethz (d0t) ch>
//
//  This Source Code Form is subject to the terms of the GNU GPL 3.0 licence.
//  If a copy of the GNU GPL 3.0 was not distributed with this
//  file, you can obtain one at http://opensource.org/licenses/GPL-3.0.
// ========================================================================================

#ifndef CudaFramework_General_AssertionDebugC_hpp
#define CudaFramework_General_AssertionDebugC_hpp



// Add an Assertion Debugging for C only!

#include <stdio.h>

#ifndef NDEBUG
   #define ASSERTMSG_C(condition,message) { if(!(condition)){ fprintf(stderr, "ASSERT FAILED: %s, file %s, line %d: %s\n", #condition, __FILE__,__LINE__, #message); abort();} }
   #define WARNINGMSG_C(condition,message) { if(!(condition)){ fprintf(stderr, "WARNING: %s, file %s, line %d: %s\n", #condition, __FILE__,__LINE__, #message); } }
#else
	#define ASSERTMSG_C(condition,message) (void)0
    #define WARNINGMSG_C(condition,message) (void)0
#endif


#endif
