// ========================================================================================
//  CudaFramework
//  Copyright (C) 2014 by Gabriel Nützi <nuetzig (at) imes (d0t) mavt (d0t) ethz (d0t) ch>
//
//  This Source Code Form is subject to the terms of the GNU GPL 3.0 licence.
//  If a copy of the GNU GPL 3.0 was not distributed with this
//  file, you can obtain one at http://opensource.org/licenses/GPL-3.0.
// ========================================================================================

#ifndef CudaFramework_General_AssertionDebug_hpp
#define CudaFramework_General_AssertionDebug_hpp

// Add an Assertion Debuggin!

//#define NDEBUG
#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <typeinfo>


#include "CudaFramework/General/Exception.hpp"

#ifndef NDEBUG
// Debug!
	/**
	* @brief An Assert Macro to use within C++ code.
	* @param condition The condition which needs to be truem otherwise an assertion is thrown!
	* @param message The message in form of cout out expression like: "Variable" << i<< "has failed"
	*/
    #define ASSERTMSG(condition , message) { if(!(condition)){ std::cerr << "ASSERT FAILED: " << #condition << " : " << message << std::endl << " @ " << __FILE__ << " (" << __LINE__ << ")" << std::endl; THROWEXCEPTION("ASSERT"); abort(); } }
    #define WARNINGMSG(condition , message) { if(!(condition)){ std::cerr << "WARNING: " << #condition << " : " << message << std::endl << " @ " << __FILE__ << " (" << __LINE__ << ")" << std::endl; } }

#else
   #define ASSERTMSG(condition,message) (void)0;
   #define WARNINGMSG(condition,message) (void)0;
#endif

   #define ERRORMSG(message) { std::cerr << "ERROR: " << message << std::endl << " @ " << __FILE__ << " (" << __LINE__ << ")" << std::endl; THROWEXCEPTION("ASSERT"); abort(); }

#endif
