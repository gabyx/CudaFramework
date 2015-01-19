// ========================================================================================
//  CudaFramework
//  Copyright (C) 2014 by Gabriel Nützi <nuetzig (at) imes (d0t) mavt (d0t) ethz (d0t) ch>
//
//  This Source Code Form is subject to the terms of the GNU GPL 3.0 licence.
//  If a copy of the GNU GPL 3.0 was not distributed with this
//  file, you can obtain one at http://opensource.org/licenses/GPL-3.0.
// ========================================================================================

#ifndef CudaFramework_PerformanceTest_PerformanceTest_hpp
#define CudaFramework_PerformanceTest_PerformanceTest_hpp

/**
* @defgroup PerformanceTest Performance Test Class
* @brief The PerformanceTest class can be used to launch a spezialised Performance Test Method.
* The PerformanceTest class is used to launch a generic performance test method.
* @{
*/


/*
* @brief This is a Performance Test class.
* The template argument TMethod is the Performance Test Method.
*/
template<typename TMethod>
class PerformanceTest{
public:

   PerformanceTest(std::string name)
      : m_test_name(name)
   {}

   ~PerformanceTest(){}

   void run(){
      m_method.initialize(m_test_name);
      m_method.runTest();
      m_method.finalize();
   }

private:
   std::string m_test_name;
   TMethod m_method;
};


// Include all TestMethods implemented so far!
#include "CudaFramework/PerformanceTest/KernelTestMethod.hpp"



/** @} */





#endif
