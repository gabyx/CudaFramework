
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
