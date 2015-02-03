
MACRO(INCLUDE_GAUSS_SEIDEL_CUDA SRC INC INCLUDE_DIRS COMMON_SOURCE_DIR)
set(${INC}
	${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/GaussSeidelGPU/GaussSeidelGPU.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/GaussSeidelGPU/GaussSeidelTestVariant.hpp
	${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/GaussSeidelGPU/KernelsGaussSeidel.cuh
)
set(${SRC}
	${COMMON_SOURCE_DIR}/src/CudaFramework/Kernels/GaussSeidelGPU/GaussSeidelGPU.cu
	${COMMON_SOURCE_DIR}/src/CudaFramework/Kernels/GaussSeidelGPU/GaussSeidelGPU.cpp
)
set(${INCLUDE_DIRS}  ${COMMON_SOURCE_DIR}/include/)
endmacro(INCLUDE_GAUSS_SEIDEL_CUDA)


MACRO(INCLUDE_PROX_CUDA SRC INC INCLUDE_DIRS COMMON_SOURCE_DIR)
set(${INC}
	${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/ProxGPU/ProxTestVariant.hpp
	${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/ProxGPU/ProxGPU.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/ProxGPU/ProxKernelSettings.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/ProxGPU/ProxSettings.hpp
	${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/ProxGPU/KernelsProx.cuh
    ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/ProxGPU/SorProxGPUVariant.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/ProxGPU/JorProxGPUVariant.hpp
)
set(${SRC}
	${COMMON_SOURCE_DIR}/src/CudaFramework/Kernels/ProxGPU/ProxGPU.cu
	${COMMON_SOURCE_DIR}/src/CudaFramework/Kernels/ProxGPU/ProxGPU.cpp
)
set(${INCLUDE_DIRS}  ${COMMON_SOURCE_DIR}/include/)
endmacro(INCLUDE_PROX_CUDA)

MACRO(INCLUDE_VECTOR_ADD_CUDA SRC INC INCLUDE_DIRS COMMON_SOURCE_DIR)
set(${INC}
	${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/VectorAddGPU/VectorAddGPU.hpp
	${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/VectorAddGPU/KernelsVectorAdd.cuh
)
set(${SRC}
	${COMMON_SOURCE_DIR}/src/CudaFramework/Kernels/VectorAddGPU/VectorAddGPU.cu
	${COMMON_SOURCE_DIR}/src/CudaFramework/Kernels/VectorAddGPU/VectorAddGPU.cpp
)
set(${INCLUDE_DIRS} ${COMMON_SOURCE_DIR}/include/)
endmacro(INCLUDE_VECTOR_ADD_CUDA)

MACRO(INCLUDE_MATRIX_MULT_CUDA SRC INC INCLUDE_DIRS COMMON_SOURCE_DIR)
set(${INC}
	${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/MatrixMultGPU/MatrixMultGPU.hpp
	${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/MatrixMultGPU/KernelsMatrixMult.cuh
)
set(${SRC}
	${COMMON_SOURCE_DIR}/src/CudaFramework/Kernels/MatrixMultGPU/MatrixMultGPU.cu
	${COMMON_SOURCE_DIR}/src/CudaFramework/Kernels/MatrixMultGPU/MatrixMultGPU.cpp
)
set(${INCLUDE_DIRS}  ${COMMON_SOURCE_DIR}/include/)
endmacro(INCLUDE_MATRIX_MULT_CUDA)

MACRO(INCLUDE_MATRIX_VECTOR_MULT_CUDA SRC INC INCLUDE_DIRS COMMON_SOURCE_DIR)
set(${INC}
   ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/MatrixVectorMultGPU/MatrixVectorMultGPU.hpp
   ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/MatrixVectorMultGPU/MatrixVectorMultTestVariant.hpp
   ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/MatrixVectorMultGPU/KernelsMatrixVectorMult.cuh
)
set(${SRC}
	${COMMON_SOURCE_DIR}/src/CudaFramework/Kernels/MatrixVectorMultGPU/MatrixVectorMultGPU.cu
	${COMMON_SOURCE_DIR}/src/CudaFramework/Kernels/MatrixVectorMultGPU/MatrixVectorMultGPU.cpp
)
set(${INCLUDE_DIRS} ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/MatrixVectorMultGPU/)
endmacro(INCLUDE_MATRIX_VECTOR_MULT_CUDA)

MACRO(INCLUDE_TESTS_CUDA SRC INC INCLUDE_DIRS COMMON_SOURCE_DIR)
set(${INC}
   	${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/TestsGPU/TestsGPU.hpp
	${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/TestsGPU/KernelsTests.cuh
)
set(${SRC}
	${COMMON_SOURCE_DIR}/src/CudaFramework/Kernels/TestsGPU/TestsGPU.cu
	${COMMON_SOURCE_DIR}/src/CudaFramework/Kernels/TestsGPU/TestsGPU.cpp
)
set(${INCLUDE_DIRS} ${COMMON_SOURCE_DIR}/include/)
endmacro(INCLUDE_TESTS_CUDA)

MACRO(INCLUDE_GENERAL_CUDA SRC INC INCLUDE_DIRS COMMON_SOURCE_DIR )
set(${INC}
	${PROJECT_BINARY_DIR}/include/CudaFramework/General/ConfigureFile.hpp
	${COMMON_SOURCE_DIR}/include/CudaFramework/General/Utilities.hpp
	${COMMON_SOURCE_DIR}/include/CudaFramework/General/FloatingPointType.hpp
	${COMMON_SOURCE_DIR}/include/CudaFramework/General/GPUMutex.hpp
	${COMMON_SOURCE_DIR}/include/CudaFramework/General/TypeTraitsHelper.hpp
	${COMMON_SOURCE_DIR}/include/CudaFramework/General/StaticAssert.hpp
	${COMMON_SOURCE_DIR}/include/CudaFramework/General/FlopsCounting.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/General/AssertionDebug.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/General/AssertionDebugC.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/CudaModern/CudaEvent.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/CudaModern/CudaTimer.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/CudaModern/CudaAlloc.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/CudaModern/CudaRefcounting.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/CudaModern/CudaDeviceMemory.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/CudaModern/CudaTypeDefs.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/CudaModern/CudaDeviceGroup.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/CudaModern/CudaContextGroup.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/CudaModern/CudaContext.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/CudaModern/CudaDevice.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/CudaModern/CudaException.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/CudaModern/CudaMemSupport.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/CudaModern/CudaMatrixSupport.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/CudaModern/CudaUtilities.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/General/Exception.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/CudaModern/CudaError.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/CudaModern/CudaMatrix.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/CudaModern/CudaPrint.hpp
    
    ${COMMON_SOURCE_DIR}/include/CudaFramework/PerformanceTest/PerformanceTest.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/PerformanceTest/KernelTestMethod.hpp
    
    ${COMMON_SOURCE_DIR}/external/pugixml/PugiXmlInclude.hpp
    ${COMMON_SOURCE_DIR}/external/tinyformat/tinyformat.h
)
set(${SRC}
    ${COMMON_SOURCE_DIR}/src/CudaFramework/CudaModern/CudaCompilerVersion.cu
    ${COMMON_SOURCE_DIR}/src/CudaFramework/CudaModern/CudaMemSupport.cpp
    ${COMMON_SOURCE_DIR}/src/CudaFramework/CudaModern/CudaDevice.cpp
    ${COMMON_SOURCE_DIR}/src/CudaFramework/CudaModern/CudaDeviceGroup.cpp
    ${COMMON_SOURCE_DIR}/src/CudaFramework/CudaModern/CudaContextGroup.cpp
    ${COMMON_SOURCE_DIR}/src/CudaFramework/CudaModern/CudaContext.cpp
    ${COMMON_SOURCE_DIR}/src/CudaFramework/CudaModern/CudaAlloc.cpp
    ${COMMON_SOURCE_DIR}/src/CudaFramework/CudaModern/CudaUtilities.cpp
    ${COMMON_SOURCE_DIR}/src/CudaFramework/CudaModern/CudaEvent.cpp
    ${COMMON_SOURCE_DIR}/src/CudaFramework/CudaModern/CudaTimer.cpp
    ${COMMON_SOURCE_DIR}/src/CudaFramework/CudaModern/CudaPrint.cpp
	${COMMON_SOURCE_DIR}/src/CudaFramework/General/Utilities.cpp
    
    ${COMMON_SOURCE_DIR}/external/pugixml/src/pugixml.cpp
)
set(${INCLUDE_DIRS} 
    ${COMMON_SOURCE_DIR}/include/
    ${COMMON_SOURCE_DIR}/external/
)
endmacro(INCLUDE_GENERAL_CUDA)

MACRO(INCLUDE_GENERAL_EXTERN_CUDA SRC INC INCLUDE_DIRS COMMON_SOURCE_DIR )
set(${INC}
	${COMMON_SOURCE_DIR}/include/CudaFramework/General/Utilities.hpp
	${COMMON_SOURCE_DIR}/include/CudaFramework/General/FloatingPointType.hpp
	${COMMON_SOURCE_DIR}/include/CudaFramework/General/TypeTraitsHelper.hpp
	${COMMON_SOURCE_DIR}/include/CudaFramework/General/GPUMutex.hpp
	${COMMON_SOURCE_DIR}/include/CudaFramework/General/StaticAssert.hpp
	${COMMON_SOURCE_DIR}/include/CudaFramework/General/FlopsCounting.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/CudaModern/CudaEvent.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/CudaModern/CudaTimer.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/CudaModern/CudaAlloc.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/CudaModern/CudaRefcounting.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/CudaModern/CudaDeviceMemory.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/CudaModern/CudaDeviceMatrix.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/CudaModern/CudaTypeDefs.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/CudaModern/CudaDeviceGroup.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/CudaModern/CudaContextGroup.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/CudaModern/CudaContext.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/CudaModern/CudaDevice.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/CudaModern/CudaException.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/CudaModern/CudaMemSupport.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/CudaModern/CudaMatrixSupport.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/CudaModern/CudaUtilities.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/General/Exception.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/CudaModern/CudaError.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/CudaModern/CudaMatrix.hpp
    
    ${COMMON_SOURCE_DIR}/include/CudaFramework/PerformanceTest/PerformanceTest.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/PerformanceTest/KernelTestMethod.hpp
    
    #${COMMON_SOURCE_DIR}/external/pugixml/src/CudaFramework/pugixml.hpp
)
set(${SRC}
    ${COMMON_SOURCE_DIR}/src/CudaFramework/CudaModern/CudaCompilerVersion.cu
    ${COMMON_SOURCE_DIR}/src/CudaFramework/CudaModern/CudaMemSupport.cpp
    ${COMMON_SOURCE_DIR}/src/CudaFramework/CudaModern/CudaDevice.cpp
    ${COMMON_SOURCE_DIR}/src/CudaFramework/CudaModern/CudaDeviceGroup.cpp
    ${COMMON_SOURCE_DIR}/src/CudaFramework/CudaModern/CudaContextGroup.cpp
    ${COMMON_SOURCE_DIR}/src/CudaFramework/CudaModern/CudaContext.cpp
    ${COMMON_SOURCE_DIR}/src/CudaFramework/CudaModern/CudaAlloc.cpp
    ${COMMON_SOURCE_DIR}/src/CudaFramework/CudaModern/CudaUtilities.cpp
    ${COMMON_SOURCE_DIR}/src/CudaFramework/CudaModern/CudaEvent.cpp
    ${COMMON_SOURCE_DIR}/src/CudaFramework/CudaModern/CudaTimer.cpp
	${COMMON_SOURCE_DIR}/src/CudaFramework/General/Utilities.cpp
    
    #${COMMON_SOURCE_DIR}/external/pugixml/src/CudaFramework/pugixml.cpp
)
set(${INCLUDE_DIRS} 
    ${COMMON_SOURCE_DIR}/include/
)
endmacro(INCLUDE_GENERAL_EXTERN_CUDA)


# Macro to include BodyInitKernel
MACRO(INCLUDE_BODY_INIT_KERNEL_CUDA SRC INC INCLUDE_DIRS COMMON_SOURCE_DIR)

set(${INC}
      ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/JORProxVel/BodyInitKernel/BodyInit.cuh
      ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/JORProxVel/BodyInitKernel/BodyInitKernelWrap.hpp
      ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/JORProxVel/BodyInitKernel/BodyInit.hpp
      ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/JORProxVel/BodyInitKernel/BodyInit.icc
      ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/JORProxVel/BodyInitKernel/BodyInitFunc.hpp

)
set(${SRC}
      ${COMMON_SOURCE_DIR}/src/CudaFramework/Kernels/JORProxVel/BodyInitKernel/BodyInit.cu
)
set(${INCLUDE_DIRS} ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/JORProxVel/BodyInitKernel/)
endmacro(INCLUDE_BODY_INIT_KERNEL_CUDA)


MACRO(INCLUDE_JOR_PROX_VEL_COMMONS_CUDA SRC INC INCLUDE_DIRS COMMON_SOURCE_DIR)

set(${INC}

      ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/JORProxVel/GPUBufferLoadStore.hpp
      ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/JORProxVel/GPUBufferOffsets.hpp
      ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/JORProxVel/JORProxVelocityGPUModule.hpp
      ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/JORProxVel/DeviceIntrinsics.cuh
      ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/JORProxVel/GeneralStructs.hpp
      ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/JORProxVel/GenRandomContactGraphClass.hpp
      ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/JORProxVel/GPUBufferOffsets.hpp
      ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/JORProxVel/UtilitiesMatrixVector.cuh
      ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/JORProxVel/VariantLaunchSettings.hpp
      ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/JORProxVel/LoadingCPUBuffers.hpp
)
set(${SRC}
)
set(${INCLUDE_DIRS} ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/JORProxVel/)
endmacro(INCLUDE_JOR_PROX_VEL_COMMONS_CUDA)

# Macro to include ContactInitKernel
MACRO(INCLUDE_CONTACT_INIT_KERNEL_CUDA SRC INC INCLUDE_DIRS COMMON_SOURCE_DIR)
set(${INC}
      ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/JORProxVel/ContactInitKernel/ContactInit.cuh
      ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/JORProxVel/ContactInitKernel/ContactInitFunc.hpp
      ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/JORProxVel/ContactInitKernel/ContactInitKernelWrap.hpp
      ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/JORProxVel/ContactInitKernel/ContactInit.hpp
      ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/JORProxVel/ContactInitKernel/ContactInit.icc
)
set(${SRC}
      ${COMMON_SOURCE_DIR}/src/CudaFramework/Kernels/JORProxVel/ContactInitKernel/ContactInit.cu

)
set(${INCLUDE_DIRS} ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/JORProxVel/ContactInitKernel/)
endmacro(INCLUDE_CONTACT_INIT_KERNEL_CUDA)


# Macro to include ContactIterationKernel
MACRO(INCLUDE_CONTACT_ITERATION_KERNEL_CUDA SRC INC INCLUDE_DIRS COMMON_SOURCE_DIR)
set(${INC}

      ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/JORProxVel/ContactIterationKernel/ContactIteration.cuh
      ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/JORProxVel/ContactIterationKernel/ContactIterationKernelWrap.hpp
      ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/JORProxVel/ContactIterationKernel/ContactIteration.hpp
      ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/JORProxVel/ContactIterationKernel/ContactIteration.icc
      ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/JORProxVel/ContactIterationKernel/ContactIterationFunc.hpp
)
set(${SRC}
      ${COMMON_SOURCE_DIR}/src/CudaFramework/Kernels/JORProxVel/ContactIterationKernel/ContactIteration.cu

)
set(${INCLUDE_DIRS} ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/JORProxVel/ContactIterationKernel/)
endmacro(INCLUDE_CONTACT_ITERATION_KERNEL_CUDA)





# Macro to include ConvergenceCheckKernel
MACRO(INCLUDE_CONVERGENCE_CHECK_KERNEL_CUDA SRC INC INCLUDE_DIRS COMMON_SOURCE_DIR)
set(${INC}
	#${COMMON_SOURCE_DIR}/inc/BodyInitKernel/....hpp
      ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/JORProxVel/ConvergenceCheckKernel/ConvergenceCheck.cuh
      ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/JORProxVel/ConvergenceCheckKernel/ConvergenceCheckKernelWrap.hpp
      ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/JORProxVel/ConvergenceCheckKernel/ConvergenceCheck.hpp
      ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/JORProxVel/ConvergenceCheckKernel/ConvergenceCheckFunc.hpp
      ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/JORProxVel/ConvergenceCheckKernel/ConvergenceCheck.icc
)
set(${SRC}
      ${COMMON_SOURCE_DIR}/src/CudaFramework/Kernels/JORProxVel/ConvergenceCheckKernel/ConvergenceCheck.cu

)
set(${INCLUDE_DIRS} ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/JORProxVel/ConvergenceCheckKernel/)
endmacro(INCLUDE_CONVERGENCE_CHECK_KERNEL_CUDA)


# Macro to include ReductionKernel
MACRO(INCLUDE_REDUCTION_KERNEL_CUDA SRC INC INCLUDE_DIRS COMMON_SOURCE_DIR)
set(${INC}
	#${COMMON_SOURCE_DIR}/inc/BodyInitKernel/....hpp

      ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/JORProxVel/ReductionKernel/ReductionTestVariant.hpp
      ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/JORProxVel/ReductionKernel/ReductionTestVariant.icc

      ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/JORProxVel/ReductionKernel/DeviceUtilities/EnumsDevice.cuh
      ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/JORProxVel/ReductionKernel/DeviceUtilities/Instantiations.cuh
      ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/JORProxVel/ReductionKernel/DeviceUtilities/DeviceOp.cuh
      ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/JORProxVel/ReductionKernel/DeviceUtilities/GPUDefines.cuh

      ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/JORProxVel/ReductionKernel/DeviceUtilities/Loadstore.cuh
      ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/JORProxVel/ReductionKernel/DeviceUtilities/Intrinsics.cuh

      ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/JORProxVel/ReductionKernel/KernelsWrappers/PartitionKernel.cuh
      ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/JORProxVel/ReductionKernel/KernelsWrappers/ReductionKernel.cuh

      ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/JORProxVel/ReductionKernel/DeviceFunctions/BinarySearch.cuh
      ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/JORProxVel/ReductionKernel/Reduction.hpp

      ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/JORProxVel/ReductionKernel/HostUtilities/Tuning.hpp
      ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/JORProxVel/ReductionKernel/HostUtilities/TuningFunctions.hpp
      ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/JORProxVel/ReductionKernel/HostUtilities/Static.hpp

      ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/JORProxVel/ReductionKernel/DeviceFunctions/CTAReduce.cuh
      ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/JORProxVel/ReductionKernel/ReductionKernelWrap.hpp
      ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/JORProxVel/ReductionKernel/Enum.h
      ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/JORProxVel/ReductionKernel/KernelsWrappers/ReduceSpineKernels.cuh
)
set(${SRC}
      ${COMMON_SOURCE_DIR}/src/CudaFramework/Kernels/JORProxVel/ReductionKernel/Reduction.cu
)
set(${INCLUDE_DIRS}
    ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/JORProxVel/ReductionKernel/
    ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/JORProxVel/ReductionKernel/HostUtilities/
    ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/JORProxVel/ReductionKernel/DeviceUtilities/
    ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/JORProxVel/ReductionKernel/DeviceFunctions/
    ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/JORProxVel/ReductionKernel/KernelsWrappers/
)
endmacro(INCLUDE_REDUCTION_KERNEL_CUDA)



# Macro to include JORProxVelKernel
MACRO(INCLUDE_JOR_PROX_VEL_KERNEL_CUDA SRC INC INCLUDE_DIRS COMMON_SOURCE_DIR)
set(${INC}
      ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/JORProxVel/JORProxVelKernel/JORProxVelKernelWrap.hpp
      ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/JORProxVel/JORProxVelKernel/JORProxVel.hpp
      ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/JORProxVel/JORProxVelKernel/JORProxVel.icc
      ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/JORProxVel/JORProxVelKernel/JORProxVelGPU.hpp
)
set(${SRC}
    ${COMMON_SOURCE_DIR}/src/CudaFramework/Kernels/JORProxVel/JORProxVelKernel/BodyUpdate.cu
)
set(${INCLUDE_DIRS}
${${INCLUDE_DIRS}}
${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/JORProxVel/JORProxVelKernel/
)
endmacro(INCLUDE_JOR_PROX_VEL_KERNEL_CUDA)




# Macros for the JOR Prox Velocity Module (includes the JORProxVelKernel)
MACRO(INCLUDE_JORPROX_VELOCITY_MODULE_EXTERN_CUDA SRC INC INCLUDE_DIRS COMMON_SOURCE_DIR )

 # Include all kernels
    INCLUDE_BODY_INIT_KERNEL_CUDA(BodyInitKernel_SRC BodyInitKernel_INC BodyInitKernel_INCLUDE_DIRS ${COMMON_SOURCE_DIR} )
    INCLUDE_CONTACT_INIT_KERNEL_CUDA(ContactInitKernel_SRC ContactInitKernel_INC ContactInitKernel_INCLUDE_DIRS ${COMMON_SOURCE_DIR} )
    INCLUDE_REDUCTION_KERNEL_CUDA(ReductionKernel_SRC ReductionKernel_INC ReductionKernel_INCLUDE_DIRS ${COMMON_SOURCE_DIR} )
    INCLUDE_JOR_PROX_VEL_KERNEL_CUDA(JORProxVelKernel_SRC JORProxVelKernel_INC JORProxVelKernel_INCLUDE_DIRS ${COMMON_SOURCE_DIR} )
    INCLUDE_CONVERGENCE_CHECK_KERNEL_CUDA(ConvergenceCheckKernel_SRC ConvergenceCheckKernel_INC ConvergenceCheckKernel_INCLUDE_DIRS ${COMMON_SOURCE_DIR} )
    INCLUDE_CONTACT_ITERATION_KERNEL_CUDA(ContactIterationKernel_SRC ContactIterationKernel_INC ContactIterationKernel_INCLUDE_DIRS ${COMMON_SOURCE_DIR} )
    INCLUDE_JOR_PROX_VEL_COMMONS_CUDA(JORProxVelCommon_SRC JORProxVelCommon_INC JORProxVelCommon_INCLUDE_DIRS ${COMMON_SOURCE_DIR} )

set(${SRC}

      ${JORProxVelKernel_SRC}
      ${BodyInitKernel_SRC}
      ${ContactInitKernel_SRC}
      ${ConvergenceCheckKernel_SRC}
      ${ReductionKernel_SRC}
      ${ContactIterationKernel_SRC}
      ${JORProxVelCommon_SRC}
)
set(${INC}
    ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/JORProxVel/JORProxVelocityGPUModule.hpp
    ${COMMON_SOURCE_DIR}/include/CudaFramework/Kernels/JORProxVel/GPUBufferOffsets.hpp
    ${JORProxVelKernel_INC}
    ${BodyInitKernel_INC}
    ${ContactInitKernel_INC}
    ${ConvergenceCheckKernel_INC}
    ${ReductionKernel_INC}
    ${ContactIterationKernel_INC}
    ${JORProxVelCommon_INC}
)
set(${INCLUDE_DIRS}

    ${COMMON_SOURCE_DIR}/inc/JORProxVel

    ${BodyInitKernel_INCLUDE_DIRS}
    ${ContactInitKernel_INCLUDE_DIRS}
    ${ConvergenceCheckKernel_INCLUDE_DIRS}
    ${ReductionKernel_INCLUDE_DIRS}
    ${JORProxVelKernel_INCLUDE_DIRS}
    ${ContactIterationKernel_INCLUDE_DIRS}
    ${JORProxVelCommon_INCLUDE_DIRS}

)
ENDMACRO(INCLUDE_JORPROX_VELOCITY_MODULE_EXTERN_CUDA)
