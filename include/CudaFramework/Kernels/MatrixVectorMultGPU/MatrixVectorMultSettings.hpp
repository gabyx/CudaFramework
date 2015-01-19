// ========================================================================================
//  CudaFramework
//  Copyright (C) 2014 by Gabriel Nützi <nuetzig (at) imes (d0t) mavt (d0t) ethz (d0t) ch>
//
//  This Source Code Form is subject to the terms of the GNU GPL 3.0 licence. 
//  If a copy of the GNU GPL 3.0 was not distributed with this
//  file, you can obtain one at http://opensource.org/licenses/GPL-3.0.
// ========================================================================================
#ifndef CudaFramework_Kernels_MatrixVectorMultGPU_MatrixVectorMultSettings_hpp
#define CudaFramework_Kernels_MatrixVectorMultGPU_MatrixVectorMultSettings_hpp

#include <string>





template<typename _PREC, bool ALM, typename settings>
   struct MatrixVectorMultSettings{
      static const bool alignMatrix  = ALM;
      typedef _PREC PREC; 
      typedef settings TVariantSettings;
};

#define DEFINE_MVMULT_SETTINGS(_TSettings_)  \
   static const bool alignMatrix = _TSettings_::alignMatrix; \
   typedef typename _TSettings_::PREC PREC; \
   typedef typename _TSettings_::TVariantSettings TSettings;


template< int _minNContacts, int _stepNContacts, int _maxNContacts, int _nMaxIterations, int _nMaxTestProblems >
   struct PerformanceTestMVSettings{
      static const int minNContacts = _minNContacts;
      static const int stepNContacts = _stepNContacts;
      static const int maxNContacts = _maxNContacts;
      static const int nMaxIterations = _nMaxIterations;
      static const int maxNTestProblems = _nMaxTestProblems;
};

#define DEFINE_MV_PERFORMANCE_SETTINGS(_TSettings_)  \
      static const int minNContacts = _TSettings_::minNContacts; \
      static const int  stepNContacts = _TSettings_::stepNContacts; \
      static const int  maxNContacts = _TSettings_::maxNContacts; \
      static const int  nMaxIterations = _TSettings_::nMaxIterations; \
      static const int  maxNTestProblems = _TSettings_::maxNTestProblems;


#endif