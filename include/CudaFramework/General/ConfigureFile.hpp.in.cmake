// ========================================================================================
//  CudaFramework
//  Copyright (C) 2014 by Gabriel Nützi <nuetzig (at) imes (d0t) mavt (d0t) ethz (d0t) ch>
//
//  This Source Code Form is subject to the terms of the GNU GPL 3.0 licence.
//  If a copy of the GNU GPL 3.0 was not distributed with this
//  file, you can obtain one at http://opensource.org/licenses/GPL-3.0.
// ========================================================================================

#ifndef CudaFramework_General_ConfigureFile_hpp_in_cmake
#define CudaFramework_General_ConfigureFile_hpp_in_cmake


namespace CudaFramework{

    static const unsigned int VersionMajor =  @CudaFramework_VERSION_MAJOR@ ;
    static const unsigned int VersionMinor =  @CudaFramework_VERSION_MINOR@ ;
    static const unsigned int VersionPatch =  @CudaFramework_VERSION_PATCH@ ;
    
}

#define CUDA_VERSION @CUDA_VERSION@
#define USE_INTEL_BLAS @USE_INTEL_BLAS_FLAG@
#define BLAS_NUM_THREADS @BLAS_NUM_THREADS@
#define USE_GOTO_BLAS @USE_GOTO_BLAS_FLAG@

#endif
