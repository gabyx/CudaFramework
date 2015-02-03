// ========================================================================================
//  CudaFramework
//  Copyright (C) 2014 by Gabriel Nützi <nuetzig (at) imes (d0t) mavt (d0t) ethz (d0t) ch>
//
//  This Source Code Form is subject to the terms of the GNU GPL 3.0 licence. 
//  If a copy of the GNU GPL 3.0 was not distributed with this
//  file, you can obtain one at http://opensource.org/licenses/GPL-3.0.
// ========================================================================================


#ifndef CudaFramework_Kernels_JORProxVel_VariantLaunchSettings_hpp
#define CudaFramework_Kernels_JORProxVel_VariantLaunchSettings_hpp



 struct VariantLaunchSettings{

unsigned int numberOfThreads;
unsigned int numberOfBlocks;
unsigned int var;


///< constructor for the struct
VariantLaunchSettings(){

numberOfThreads=128;
numberOfBlocks=128;
var=1;
}


};



#endif // VariantLaunchSettings
